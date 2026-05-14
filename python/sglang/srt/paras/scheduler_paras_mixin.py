import abc
from collections import deque
from types import SimpleNamespace
from typing import List, Any, Optional
import torch
import logging
import time
import os

from sglang.srt.managers.io_struct import (
    ParaSAutoSwitchReq,
    ParaSConfigureReqInput,
    ParaSConfigureReqOutput,
    ParaSConfigureReqType,
)
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info, get_attention_tp_group
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args

from sglang.srt.paras.utils import paras_func, paras_profile_func
from sglang.srt.paras.gather_manager import ParaSReqGatherManager
from sglang.srt.paras.scatter_manager import ParaSReqScatterManager
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.managers.utils import SenderWrapper


logger = logging.getLogger(__name__)

class TimeReporter:
    def __init__(self, op_name: str):
        self.op_name = op_name
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        cost_ms = (end_time - self.start_time) * 1000
        logger.info(f"Time taken to {self.op_name}: {cost_ms} ms")

class ParasAutoSwitchPolicy(abc.ABC):
    """Base class for ParaS auto-switch policies.

    Subclasses define `observation_for_batch` to (a) filter iterations the
    policy cares about and (b) compute the per-iteration global metric value.
    The base handles windowing, cooldown, and the cross-threshold decision.
    """

    def __init__(self, threshold: int, window: int, cooldown_sec: float):
        self.threshold = threshold
        self.window: deque = deque(maxlen=window)
        self.cooldown_sec = cooldown_sec
        self.cooldown_until: float = 0.0

    @abc.abstractmethod
    def observation_for_batch(self, batch: ScheduleBatch) -> Optional[int]:
        """Return this iteration's global metric, or None to skip the iteration."""

    def observe(self, batch: ScheduleBatch, now: float) -> None:
        value = self.observation_for_batch(batch)
        if value is None or value <= 0:
            return
        self.window.append(value)

    def pick_target(self, current_mode: str, now: float) -> Optional[str]:
        if now < self.cooldown_until:
            return None
        if len(self.window) < self.window.maxlen:
            return None
        avg = sum(self.window) / len(self.window)
        target: Optional[str] = None
        if current_mode == "EP" and avg < self.threshold:
            target = "TP"
        elif current_mode == "TP" and avg > self.threshold:
            target = "EP"
        if target is not None:
            logger.info(
                f"ParaS [{type(self).__name__}] policy fired: "
                f"{current_mode} -> {target} at t={now:.3f} | "
                f"observations={list(self.window)} avg={avg:.2f} "
                f"threshold={self.threshold} window_maxlen={self.window.maxlen} "
                f"cooldown_sec={self.cooldown_sec}"
            )
            self.cooldown_until = now + self.cooldown_sec
            self.window.clear()
        return target


class PrefillAutoSwitchPolicy(ParasAutoSwitchPolicy):
    """Observes pure prefill (EXTEND) iterations; metric is global prefill tokens."""

    def observation_for_batch(self, batch: ScheduleBatch) -> Optional[int]:
        if batch.forward_mode != ForwardMode.EXTEND:
            return None
        if batch.global_num_tokens:
            return int(sum(batch.global_num_tokens))
        return sum(req.seqlen for req in batch.reqs)


class DecodeAutoSwitchPolicy(ParasAutoSwitchPolicy):
    """Observes every iteration; metric is global in-flight token / request count.

    The metric is `sum(batch.global_num_tokens)` in EP+DP-attention mode (the
    all-gathered per-DP token count, summed across all DP ranks) and
    `len(batch.reqs)` in TP-only mode. Forward mode is intentionally NOT
    filtered: rank 0 may run an idle batch (`forward_mode = IDLE`) when other
    DP ranks hold the work, but its `batch.global_num_tokens` still carries
    the true global state via the MLP all-gather. Skipping idle batches would
    silently strand the policy whenever round-robin routes light-load
    requests to a non-zero DP rank.
    """

    def observation_for_batch(self, batch: ScheduleBatch) -> Optional[int]:
        if batch.global_num_tokens:
            return int(sum(batch.global_num_tokens))
        return len(batch.reqs)


class HybridAutoSwitchPolicy(ParasAutoSwitchPolicy):
    """Mixed prefill+decode batches. Not yet implemented; raises at construction."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Hybrid (mixed prefill+decode) auto-switch policy is not implemented. "
            "Use 'prefill' or 'decode'. ParaS disables chunked prefill so "
            "ForwardMode.MIXED should not occur in practice."
        )

    def observation_for_batch(self, batch: ScheduleBatch) -> Optional[int]:
        raise NotImplementedError  # unreachable: __init__ raises


_PARAS_AUTO_SWITCH_POLICY_CLASSES = {
    "prefill": PrefillAutoSwitchPolicy,
    "decode": DecodeAutoSwitchPolicy,
    "hybrid": HybridAutoSwitchPolicy,
}


class SchedulerParasMixin:
    """
    This class implements the parallel configuration logic for Scheduler.
    """
    
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    
    def init_paras_config(self):
        # Always initialize so non-ParaS schedulers can no-op the event-loop hook
        # with a single `if self._paras_auto_policy is not None:` check.
        self._paras_auto_policy: Optional[ParasAutoSwitchPolicy] = None

        if not self.server_args.enable_paras_moe:
            return

        # ParaS config
        self.paras_tp_size = self.server_args.paras_tp_size
        self.paras_tp_rank = self.tp_rank % self.paras_tp_size
        self.paras_dp_size = self.tp_size // self.paras_tp_size
        self.paras_dp_rank = self.tp_rank // self.paras_tp_size
        self.paras_tp_group = self.tp_worker.get_paras_tp_group()
        self.paras_tp_cpu_group = self.paras_tp_group.cpu_group
        self.paras_tp_attn_tp_group = self.paras_tp_group
        self.paras_tp_attn_tp_cpu_group = self.paras_tp_group.cpu_group

        if self.paras_tp_rank == 0:
            self.tp_recv_from_tokenizer = self.recv_from_tokenizer
            self.tp_send_to_tokenizer = self.send_to_tokenizer
            self.tp_send_to_detokenizer = self.send_to_detokenizer
            self.tp_recv_from_rpc = self.recv_from_rpc
        else:
            self.tp_recv_from_tokenizer = None
            self.tp_recv_from_rpc = None
            self.tp_send_to_tokenizer = SenderWrapper(None)
            self.tp_send_to_detokenizer = SenderWrapper(None)

        self.paras_ep_size = self.tp_size
        self.paras_ep_rank = self.tp_rank
        self.paras_ep_group = self.tp_group
        self.paras_ep_cpu_group = self.tp_cpu_group
        # Use the singleton _ATTN_TP_GROUP (built at init with enable_dp_attention=True)
        # rather than self.tp_group (all ranks). In EP mode attn_tp_size==1, so this
        # group should be a single-rank group matching baseline EP semantics.
        self.paras_ep_attn_tp_group = get_attention_tp_group()
        self.paras_ep_attn_tp_cpu_group = get_attention_tp_group().cpu_group

        self.ep_recv_from_tokenizer = self.recv_from_tokenizer
        self.ep_recv_from_rpc = self.recv_from_rpc
        self.ep_send_to_tokenizer = self.send_to_tokenizer
        self.ep_send_to_detokenizer = self.send_to_detokenizer

        self.paras_parallelism_config = "EP"

        sa = self.server_args
        if sa.paras_auto_switch:
            policy_cls = _PARAS_AUTO_SWITCH_POLICY_CLASSES[sa.paras_auto_switch_policy]
            self._paras_auto_policy = policy_cls(
                threshold=sa.paras_auto_switch_threshold,
                window=sa.paras_auto_switch_window,
                cooldown_sec=sa.paras_auto_switch_cooldown_sec,
            )

    def paras_configure_helper(self):
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            _,
            _,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if self.is_hybrid:
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )
        
    def paras_check(self):
        # Canary: log if running_batch ever contains a req with output_ids=[].
        # This state was thought unreachable in normal mode (all reqs entering
        # running_batch traverse last_batch first; process_batch_result populates
        # output_ids before last_batch is set). The previous guard from
        # b6d0b9665 inspected this state. If this log fires in production, the
        # guard was needed and should be reinstated.
        if self.running_batch is not None and any(
            len(req.output_ids) == 0 for req in self.running_batch.reqs
        ):
            logger.warning(
                "paras_check: running_batch contains a req with output_ids=[] "
                "(thought unreachable in normal mode); investigate before "
                "relying on this assumption"
            )
        return True

    def paras_auto_observe(self, batch: Optional[ScheduleBatch]) -> None:
        assert self._paras_auto_policy is not None
        if batch is None:
            return
        self._paras_auto_policy.observe(batch, time.time())

    def _paras_auto_clear_window_on_switch(self) -> None:
        assert self._paras_auto_policy is not None
        policy = self._paras_auto_policy
        policy.window.clear()
        policy.cooldown_until = max(
            policy.cooldown_until, time.time() + policy.cooldown_sec
        )

    def paras_auto_pick_signal(self) -> Optional[ParaSAutoSwitchReq]:
        assert self._paras_auto_policy is not None
        target = self._paras_auto_policy.pick_target(
            self.paras_parallelism_config, time.time()
        )
        if target is None:
            return None
        req_type = (
            ParaSConfigureReqType.CONFIGURE_TP
            if target == "TP"
            else ParaSConfigureReqType.CONFIGURE_EP
        )
        logger.info(
            f"ParaS auto-switch policy fired: {self.paras_parallelism_config} -> {target}"
        )
        return ParaSAutoSwitchReq(target=req_type)
    
    def paras_get_req_seqlens(self, reqs: List[Req]):
        seqlens = []
        for req in reqs:
            seqlens.append(req.seqlen)
        return seqlens
    
    def paras_get_local_reqs(self):
        # Merge the last batch into the running batch, now every request is in the decode status
        self.merge_last_batch()
        return self.running_batch.reqs
    
    @paras_func
    def paras_configure_tp(self):
        if self.paras_parallelism_config == "TP":
            logger.warning("paras_configure_tp called but already in TP mode; skipping")
            return
        if not self.paras_check():
            return

        assert self.server_args.enable_paras_moe, "ParaS parallelism is not enabled."
        assert not self.enable_overlap, "Overlap schedule is not supported currently in ParaS."
        torch.cuda.synchronize()

        if self._paras_auto_policy is not None:
            self._paras_auto_clear_window_on_switch()

        # switch from EP to DP x TP
        self.paras_parallelism_config = "TP"
        self.server_args.enable_dp_attention = False
        # Store the string value (matching ServerArgs dataclass), not the Enum.
        # `require_attn_tp_gather` compares `server_args.moe_a2a_backend != "none"`
        # and only the string form compares equal; storing the Enum here caused
        # the scheduler to believe a2a was still enabled and to pad the prefill
        # batch to a multiple of attn_tp_size via `prepare_mlp_sync_batch`,
        # injecting uninitialized padding tokens whose qkv_proj output
        # overflows the BF16 attention softmax on some TP ranks.
        self.server_args.moe_a2a_backend = MoeA2ABackend.NONE.value
        self.server_args.dp_size = 1
        self.server_args.ep_size = 1
        moe_utils.MOE_A2A_BACKEND = MoeA2ABackend.NONE
        
        self.paras_start_profile("/tmp/paras_configure_profile")
        self.tree_cache.reset()
        local_reqs = self.paras_get_local_reqs()
        local_waiting_reqs = list(self.waiting_queue)

        paras_gather_manager = ParaSReqGatherManager(
            local_reqs,
            self.paras_tp_group,
            self.req_to_token_pool, 
            self.token_to_kv_pool_allocator,
            peer_ctx=getattr(
                self.tp_worker.model_runner.model,
                '_fused_peer_access_ctx',
                None,
            ),
            method=os.environ.get("PARAS_KV_TRANSFER_METHOD", "nccl"),
            local_waiting_reqs=local_waiting_reqs,
            layer_specs=getattr(
                self.tp_worker.model_runner.model,
                'paras_layer_specs',
                None,
            ),
        )
        
        start_time = time.time()
        
        with TimeReporter("gather_global_reqs"):
            paras_gather_manager.gather_global_reqs()
        
        with TimeReporter("reorchestrate_cache"):
            paras_gather_manager.reorchestrate_cache()
        
        with TimeReporter("gather_cache"):
            paras_gather_manager.gather_cache()
        
        self.running_batch = paras_gather_manager.get_new_running_batch(
            self.tokenizer,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor
        )
        self.waiting_queue = paras_gather_manager.get_new_waiting_queue(
            self.paras_tp_rank
        )
        # paras_gather_manager.update_running_batch_inplace(self.running_batch)

        with TimeReporter("transfer_weights"):
            self.tp_worker.paras_configure_tp(self.paras_tp_size, self.paras_tp_rank)
        
        end_time = time.time()
        cost_ms = (end_time - start_time) * 1000
        logger.info(f"Time taken to configure TP: {cost_ms} ms")

        self.paras_stop_profile()

        # drop-in replacement for scheduler tp configs 
        self.tp_size = self.paras_tp_size
        self.tp_rank = self.paras_tp_rank
        self.attn_tp_group = self.paras_tp_group
        self.attn_tp_cpu_group = self.paras_tp_cpu_group
        self.tp_group = self.paras_tp_group
        self.tp_cpu_group = self.paras_tp_cpu_group

        # NOTE(shaoyuw): attn_dp_rank should be dealt with more carefully. 
        #                But now it seems to be used only a few times.
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            self.paras_tp_rank,
            self.paras_tp_size,
            self.paras_dp_rank,
        )
        self.attn_tp_group = self.paras_tp_attn_tp_group
        self.attn_tp_cpu_group = self.paras_tp_attn_tp_cpu_group

        self.recv_from_tokenizer = self.tp_recv_from_tokenizer
        self.send_to_tokenizer = self.tp_send_to_tokenizer
        self.send_to_detokenizer = self.tp_send_to_detokenizer
        self.recv_from_rpc = self.tp_recv_from_rpc

        # Drop the pre-switch batch reference: its req_pool_idx points into the
        # destroyed EP pool layout, and merge_last_batch already absorbed its
        # reqs into the new TP running_batch via paras_get_local_reqs().
        self.last_batch = None
        torch.cuda.synchronize()

    @paras_func
    def paras_configure_ep(self):
        # Entry guards
        if self.paras_parallelism_config != "TP":
            logger.warning("paras_configure_ep called but not in TP mode")
            return
        if not self.paras_check():
            return
        assert self.server_args.enable_paras_moe, "ParaS parallelism is not enabled."
        assert not self.enable_overlap, "Overlap schedule is not supported currently in ParaS."
        assert self.paras_dp_size == 1, "paras_configure_ep only supports dp_size==1"
        torch.cuda.synchronize()

        if self._paras_auto_policy is not None:
            self._paras_auto_clear_window_on_switch()

        self.paras_start_profile("/tmp/paras_configure_profile")

        # Phase 1: Prepare — reset tree cache, merge batches, build global req list
        self.tree_cache.reset()
        self.merge_last_batch()
        global_reqs = list(self.running_batch.reqs) if self.running_batch else []
        local_waiting_reqs = list(self.waiting_queue)

        # Phase 2: Scatter — partition reqs, shrink pools, scatter KV cache
        paras_scatter_manager = ParaSReqScatterManager(
            global_reqs=global_reqs,
            scatter_group=self.paras_tp_group,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            peer_ctx=getattr(
                self.tp_worker.model_runner.model,
                '_fused_peer_access_ctx',
                None,
            ),
            paras_tp_rank=self.paras_tp_rank,
            paras_tp_size=self.paras_tp_size,
            local_waiting_reqs=local_waiting_reqs,
            layer_specs=getattr(
                self.tp_worker.model_runner.model,
                'paras_layer_specs',
                None,
            ),
        )

        start_time = time.time()

        with TimeReporter("partition_requests"):
            paras_scatter_manager.partition_requests()

            with TimeReporter("reorchestrate_cache"):
                paras_scatter_manager.reorchestrate_cache()

        with TimeReporter("scatter_cache"):
            paras_scatter_manager.scatter_cache()

        self.running_batch = paras_scatter_manager.get_new_running_batch(
            self.tokenizer,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
        )
        self.waiting_queue = paras_scatter_manager.get_new_waiting_queue()

        # Phase 3: Model switch (weights + attention)
        with TimeReporter("transfer_weights"):
            self.tp_worker.paras_configure_ep()

        end_time = time.time()
        cost_ms = (end_time - start_time) * 1000
        logger.info(f"Time taken to configure EP: {cost_ms} ms")

        self.paras_stop_profile()

        # Phase 4: Update scheduler config and restore tokenizer
        # switch from TP to EP
        self.paras_parallelism_config = "EP"
        self.server_args.enable_dp_attention = True
        self.server_args.moe_a2a_backend = MoeA2ABackend.DEEPEP.value
        self.server_args.dp_size = self.paras_ep_size
        self.server_args.ep_size = self.paras_ep_size
        moe_utils.MOE_A2A_BACKEND = MoeA2ABackend.DEEPEP

        # drop-in replacement for scheduler ep configs
        self.tp_size = self.paras_ep_size
        self.tp_rank = self.paras_ep_rank
        self.tp_group = self.paras_ep_group
        self.tp_cpu_group = self.paras_ep_cpu_group
        # equals to:
        # self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = 0, 1, self.tp_rank
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                self.server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )
        self.attn_tp_group = self.paras_ep_attn_tp_group
        self.attn_tp_cpu_group = self.paras_ep_attn_tp_cpu_group

        self.recv_from_tokenizer = self.ep_recv_from_tokenizer
        self.send_to_tokenizer = self.ep_send_to_tokenizer
        self.send_to_detokenizer = self.ep_send_to_detokenizer
        self.recv_from_rpc = self.ep_recv_from_rpc

        # See paras_configure_tp's matching reset: drop pre-switch batch ref.
        self.last_batch = None
        torch.cuda.synchronize()

    def paras_configure_handle(self, recv_req: ParaSConfigureReqInput):
        if recv_req.type == ParaSConfigureReqType.CONFIGURE_TP:
            self.paras_configure_tp()
        elif recv_req.type == ParaSConfigureReqType.CONFIGURE_EP:
            self.paras_configure_ep()
        else:
            raise ValueError(f"Unrecognized ParaSConfigureReqType: {recv_req.type}")
        return ParaSConfigureReqOutput()
    
    def paras_start_profile(self, output_dir: str = "/tmp/paras_configure_profile"):
        import os
        output_dir = os.environ.get("PARAS_PROFILE_DIR", output_dir)
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                output_dir,
                worker_name=f"rank{self.tp_rank}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.start()
        
    def paras_stop_profile(self):
        self.profiler.stop()
        self.profiler = None