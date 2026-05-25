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
from sglang.srt.paras.gather_manager import ParaSReqGatherManager, recover_request
from sglang.srt.paras.scatter_manager import ParaSReqScatterManager
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.managers.utils import SenderWrapper
from sglang.srt.utils.common import freeze_gc


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
    policy cares about and (b) compute the per-iteration global metric value
    using whatever scheduler state they need. The base handles windowing,
    cooldown, and the cross-threshold decision.
    """

    def __init__(self, threshold: int, window: int, cooldown_sec: float):
        self.threshold = threshold
        self.window: deque = deque(maxlen=window)
        self.cooldown_sec = cooldown_sec
        self.cooldown_until: float = 0.0

    @abc.abstractmethod
    def observation_for_batch(
        self, scheduler: Any, batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        """Return this iteration's global metric, or None to skip the iteration."""

    def observe(
        self, scheduler: Any, batch: Optional[ScheduleBatch], now: float
    ) -> None:
        value = self.observation_for_batch(scheduler, batch)
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

    def observation_for_batch(
        self, scheduler: Any, batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        if batch is None or batch.forward_mode != ForwardMode.EXTEND:
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

    def observation_for_batch(
        self, scheduler: Any, batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        if batch is None:
            return None
        if batch.global_num_tokens:
            return int(sum(batch.global_num_tokens))
        return len(batch.reqs)


class HybridAutoSwitchPolicy(ParasAutoSwitchPolicy):
    """Two-signal auto-switch: pending reqs (rollout-style) OR prefill tokens.

    Motivation: the rollout signal undercounts during prefill spikes because
    reqs being prefilled are in `cur_batch`, not in `running_batch.reqs` and
    not in `waiting_queue`. During pure-prefill burst onset rollout often
    sees running+waiting=0 and the base-class `observe()` skips the iteration
    (value <= 0 filter), leaving the window partial and `pick_target` blind.
    This policy adds a prefill-tokens window so prefill spikes can drive the
    TP -> EP transition even when the req-count signal is idle.

    Switching rules:
      * TP -> EP: EITHER window full AND its avg > threshold (catch the burst
        on whichever signal trips first; prefill-tokens fires faster than
        running+waiting during burst onset).
      * EP -> TP: BOTH signals "cold". A partial/empty window counts as cold,
        so the policy can drop back during pure-decode quiet periods when
        prefill_window has no fresh samples.

    Window state is cleared on every fire so successive switches are isolated.
    """

    def __init__(
        self,
        threshold: int,
        window: int,
        cooldown_sec: float,
        prefill_threshold: int,
    ):
        super().__init__(threshold, window, cooldown_sec)
        self.prefill_threshold = prefill_threshold
        self.prefill_window: deque = deque(maxlen=window)

    def observation_for_batch(
        self, scheduler: Any, batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        if scheduler.paras_parallelism_config == "EP":
            if batch is None:
                return None
            running_field = batch.global_running_reqs
            waiting_field = batch.global_waiting_reqs
            if running_field is None or waiting_field is None:
                return None
            return int(sum(running_field)) + int(sum(waiting_field))
        rb = scheduler.running_batch
        running = len(rb.reqs) if rb is not None else 0
        waiting = len(scheduler.waiting_queue)
        return running + waiting

    def _prefill_observation(
        self, scheduler: Any, batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        if batch is None or batch.forward_mode != ForwardMode.EXTEND:
            return None
        if batch.global_num_tokens:
            return int(sum(batch.global_num_tokens))
        return sum(req.seqlen for req in batch.reqs)

    def observe(
        self, scheduler: Any, batch: Optional[ScheduleBatch], now: float
    ) -> None:
        req_value = self.observation_for_batch(scheduler, batch)
        if req_value is not None and req_value > 0:
            self.window.append(req_value)
        prefill_value = self._prefill_observation(scheduler, batch)
        if prefill_value is not None and prefill_value > 0:
            self.prefill_window.append(prefill_value)

    def pick_target(self, current_mode: str, now: float) -> Optional[str]:
        if now < self.cooldown_until:
            return None
        req_full = len(self.window) >= self.window.maxlen
        prefill_full = len(self.prefill_window) >= self.prefill_window.maxlen
        if not req_full and not prefill_full:
            return None

        req_avg = sum(self.window) / len(self.window) if self.window else 0.0
        prefill_avg = (
            sum(self.prefill_window) / len(self.prefill_window)
            if self.prefill_window else 0.0
        )
        req_hot = req_full and req_avg > self.threshold
        prefill_hot = prefill_full and prefill_avg > self.prefill_threshold
        req_cold = (not req_full) or req_avg < self.threshold
        prefill_cold = (not prefill_full) or prefill_avg < self.prefill_threshold

        target: Optional[str] = None
        trigger = "?"
        if current_mode == "TP" and (req_hot or prefill_hot):
            target = "EP"
            trigger = "BOTH" if (req_hot and prefill_hot) else ("REQ" if req_hot else "PREFILL")
        elif current_mode == "EP" and req_cold and prefill_cold:
            target = "TP"
            trigger = "BOTH_COLD"

        if target is not None:
            logger.info(
                f"ParaS [HybridAutoSwitchPolicy] policy fired: "
                f"{current_mode} -> {target} at t={now:.3f} | trigger={trigger} | "
                f"req_avg={req_avg:.2f} req_thresh={self.threshold} "
                f"req_window={list(self.window)} | "
                f"prefill_avg={prefill_avg:.2f} prefill_thresh={self.prefill_threshold} "
                f"prefill_window={list(self.prefill_window)} | "
                f"cooldown_sec={self.cooldown_sec}"
            )
            self.cooldown_until = now + self.cooldown_sec
            self.window.clear()
            self.prefill_window.clear()
        return target


class RolloutAutoSwitchPolicy(ParasAutoSwitchPolicy):
    """Observes every iteration; metric is global pending request count.

    Mode-aware source selection:

    * EP mode: rank 0 holds only its DP slice, so the per-rank running batch
      and waiting queue underrepresent the global state. We sum the MLP-sync
      all-gather output (``batch.global_running_reqs`` / ``global_waiting_reqs``)
      across DP ranks. Skip iterations where the all-gather hasn't populated
      yet (idle / boot).
    * TP mode: unified data plane, every rank holds the same running batch
      after the EP->TP switch. Rank 0's local view IS the global view, so we
      read ``scheduler.running_batch`` and ``scheduler.waiting_queue``
      directly. The all-gather is unreliable in TP mode post-switch (see the
      matching workaround in ParasMetricsSampler), so we must NOT consult
      ``batch.global_*`` here.
    """

    def observation_for_batch(
        self, scheduler: Any, batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        if scheduler.paras_parallelism_config == "EP":
            if batch is None:
                return None
            running_field = batch.global_running_reqs
            waiting_field = batch.global_waiting_reqs
            if running_field is None or waiting_field is None:
                return None
            return int(sum(running_field)) + int(sum(waiting_field))
        rb = scheduler.running_batch
        running = len(rb.reqs) if rb is not None else 0
        waiting = len(scheduler.waiting_queue)
        return running + waiting


_PARAS_AUTO_SWITCH_POLICY_CLASSES = {
    "prefill": PrefillAutoSwitchPolicy,
    "decode": DecodeAutoSwitchPolicy,
    "hybrid": HybridAutoSwitchPolicy,
    "rollout": RolloutAutoSwitchPolicy,
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
            policy_kwargs = dict(
                threshold=sa.paras_auto_switch_threshold,
                window=sa.paras_auto_switch_window,
                cooldown_sec=sa.paras_auto_switch_cooldown_sec,
            )
            if sa.paras_auto_switch_policy == "hybrid":
                policy_kwargs["prefill_threshold"] = sa.paras_auto_switch_prefill_threshold
            self._paras_auto_policy = policy_cls(**policy_kwargs)

        # Freeze the long-lived serving heap (weights, KV pool layouts, sampler
        # state, radix tree, attention metadata buffers) so subsequent gen-2 GC
        # cycles do not walk it. Without this, the ~2k Req objects deserialized
        # by paras_tp_group_all_gather_reqs on every EP<->TP switch trip the
        # gen-2 threshold and the resulting heap walk stalls the GIL for
        # ~1.5-2 s (root-caused in INVESTIGATION_paras_gather_reqs_slow.md).
        # Called on every rank: gc.freeze is per-process and idempotent. By
        # this point model load + cuda graph capture inside TpModelWorker.__init__
        # have completed (and their own freeze_gc context manager has exited),
        # so the captured heap reflects the post-warmup steady state.
        freeze_gc("paras_serving_warmup")

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

        from sglang.srt.paras.paras_memory_manager import (
            get_global_paras_memory_manager,
        )

        _paras_mgr = get_global_paras_memory_manager()
        if _paras_mgr is not None:
            if self.paras_parallelism_config == "TP":
                self.max_running_requests = _paras_mgr.get_tp_max_running_requests()
            else:
                self.max_running_requests = _paras_mgr.get_ep_max_running_requests()

        # Refresh pp_max_micro_batch_size to track the post-switch per-mode cap.
        # Without this, the scheduler's admission gate
        # (Scheduler.get_num_allocatable_reqs in managers/scheduler.py, which
        # returns pp_max_micro_batch_size - running_bs) keeps using the
        # boot-time EP per-rank value (max_running_requests // dp_size) and
        # silently caps paras-tp's running batch at the EP slice even after
        # configure_tp updates self.max_running_requests to the global cap.
        get_global_server_args().pp_max_micro_batch_size = max(
            self.max_running_requests // max(self.server_args.pp_size, 1), 1
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
        if self._paras_auto_policy is None:
            return
        self._paras_auto_policy.observe(self, batch, time.time())

    def _paras_drain_overlap_pipeline(self) -> None:
        # event_loop_overlap builds batch N+1 on CPU while GPU runs batch N
        # and stores (batch_copy, result) tuples in self.result_queue for
        # processing in the NEXT iteration. A paras EP<->TP switch destroys
        # mode-dependent state (KV pool layout, attention metadata, expert
        # routing, future_map indices), so any in-flight queued batch MUST
        # be fully processed before the switch body runs. Otherwise the
        # post-switch process_batch_result would dereference stale GPU
        # tensors or pool indices and either crash or corrupt outputs.
        if not getattr(self, "enable_overlap", False):
            return
        result_queue = getattr(self, "result_queue", None)
        if result_queue is None:
            return
        while result_queue:
            tmp_batch, tmp_result = result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
        # process_batch_result_prefill populates output_ids[0] on the
        # newly-prefilled reqs but does NOT add them to running_batch.
        # In normal overlap flow the next iter's get_next_batch_to_run
        # calls merge_last_batch to absorb them; the switch path skips
        # that iter and reads running_batch.reqs directly for global
        # gather, so without an explicit merge here the just-prefilled
        # reqs are silently dropped (causing the 37-req loss + apparent
        # hang reproduced in the 20260516 smoke). Merge against the
        # existing self.last_batch (the original batch from the previous
        # iter, with intact sampling_info), NOT the popped tmp_batch:
        # ScheduleBatch.copy() omits sampling_info so merge_batch on the
        # copy crashes at SamplingBatchInfo.merge_batch:315 on a NoneType
        # `other`. The copy and self.last_batch share the reqs list, so
        # process_batch_result's output_ids mutations are visible here.
        self.merge_last_batch()
        # process_batch_result above already called cache_finished_req on reqs
        # that finished during the drain, freeing their old KV/req-pool slots.
        # But those reqs remain in running_batch.reqs until the next iter's
        # update_running_batch -> filter_batch removes them. The paras switch
        # path skips that iter and reads running_batch.reqs straight into the
        # gather/scatter manager, so without an explicit filter here those
        # finished reqs get fresh KV slots allocated for them in the new pool
        # via reorchestrate_cache, then orphaned on the post-switch iter --
        # producing the ~4105-token leak first observed in paras-t256
        # (~16 finished reqs * ~256 tokens, tripping the strict-mode
        # check_memory assertion in scheduler_runtime_checker_mixin).
        if self.running_batch is not None:
            self.running_batch.filter_batch()
        self.last_batch = None
        self.cur_batch = None

    def _paras_auto_clear_window_on_switch(self) -> None:
        assert self._paras_auto_policy is not None
        policy = self._paras_auto_policy
        policy.window.clear()
        policy.cooldown_until = max(
            policy.cooldown_until, time.time() + policy.cooldown_sec
        )

    def _paras_restore_tp_state_after_aborted_ep_switch(self) -> None:
        # Roll back the TP->EP preamble in paras_configure_ep when the
        # capacity precheck aborts the switch. Asymmetric vs the EP->TP
        # restore: paras_configure_ep defers all server_args /
        # parallelism_config / MOE_A2A_BACKEND mutations until AFTER
        # reorchestrate_cache succeeds, so the preamble has only two
        # things to undo (in reverse order):
        #   1. prune_request mutations applied by the ParaSReqScatterManager
        #      constructor on local_waiting_reqs (via
        #      paras_tp_group_all_gather_reqs). Rank 0's waiting_queue items
        #      are pruned in place; other ranks pass an empty list so are
        #      no-ops. running_batch.reqs are NOT pruned by the scatter
        #      ctor, but tree_cache.reset() ran earlier so their stale
        #      last_host_node / last_node references must be re-linked to
        #      the new (empty) root via recover_request -- defensively
        #      iterating over them too costs nothing and matches the EP->TP
        #      restore's symmetry.
        #   2. profiler started by paras_start_profile.
        # tree_cache.reset() and merge_last_batch() are NOT reverted: the
        # radix tree rebuilds on subsequent prefills, and the merged
        # running_batch is the correct TP-mode resume state.
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                recover_request(req, self.tree_cache, self.tokenizer)
        for req in self.waiting_queue:
            recover_request(req, self.tree_cache, self.tokenizer)

        self.paras_stop_profile()

    def _paras_restore_ep_state_after_aborted_tp_switch(self) -> None:
        # Roll back the EP->TP preamble in paras_configure_tp when the
        # capacity precheck aborts the switch. Undoes (in reverse order):
        #   1. prune_request mutations on running_batch.reqs and waiting_queue
        #      applied by paras_tp_group_all_gather_reqs (it nulls
        #      tokenizer / radix-tree linkage on every local req to keep
        #      pickle off GPU tensors). Without this restore the scheduler
        #      would dereference None on the next prefill/decode iteration.
        #   2. profiler started by paras_start_profile (no destructive side
        #      effect, but leaving it running leaks a profiler context).
        #   3. server_args + global moe_a2a_backend mutations -- mirrors the
        #      EP epilogue in paras_configure_ep so subsequent EP iterations
        #      see a coherent config.
        # tree_cache.reset() and merge_last_batch() are NOT reverted: the
        # radix tree rebuilds naturally on subsequent prefills, and the
        # merged running_batch is already the correct EP-mode state.
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                recover_request(req, self.tree_cache, self.tokenizer)
        for req in self.waiting_queue:
            recover_request(req, self.tree_cache, self.tokenizer)

        self.paras_stop_profile()

        self.paras_parallelism_config = "EP"
        self.server_args.enable_dp_attention = True
        self.server_args.moe_a2a_backend = MoeA2ABackend.DEEPEP.value
        self.server_args.dp_size = self.paras_ep_size
        self.server_args.ep_size = self.paras_ep_size
        moe_utils.MOE_A2A_BACKEND = MoeA2ABackend.DEEPEP

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
        self._paras_drain_overlap_pipeline()
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

        # Capacity precheck. For models with num_kv_heads < paras_tp_size
        # (e.g. Qwen3-235B: num_kv_heads=4 at paras_tp_size=8 -> KV heads
        # replicated 2x across ranks), the TP-mode KV pool is roughly half
        # the EP-mode pool. Under heavy in-flight load an EP->TP switch can
        # overflow the smaller TP pool. Detect that here and abort the
        # switch gracefully -- staying in EP, restoring preamble state --
        # instead of letting the asserts in reorchestrate_cache crash the
        # scheduler (observed in the N=2048 paras-t128 rollout sweep on
        # 2026-05-16). The autoswitch policy's cooldown was already
        # extended by _paras_auto_clear_window_on_switch above, so the
        # natural retry spacing applies.
        precheck_ok, precheck_reason, precheck_details = (
            paras_gather_manager.precheck_tp_capacity()
        )
        if not precheck_ok:
            logger.warning(
                "ParaS EP->TP switch rejected by precheck: %s | "
                "num_global_tokens=%d new_cache_size=%d "
                "num_reqs=%d new_req_pool_size=%d. "
                "Staying in EP; autoswitch will retry after cooldown.",
                precheck_reason,
                precheck_details["num_global_tokens"],
                precheck_details["new_cache_size"],
                precheck_details["num_reqs"],
                precheck_details["new_req_pool_size"],
            )
            self._paras_restore_ep_state_after_aborted_tp_switch()
            return True

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
        self.waiting_queue = paras_gather_manager.get_new_waiting_queue()
        # paras_gather_manager.update_running_batch_inplace(self.running_batch)

        with TimeReporter("transfer_weights"):
            self.tp_worker.paras_configure_tp(self.paras_tp_size, self.paras_tp_rank)

        sampler = self.tp_worker.model_runner.sampler
        sampler.tp_sync_group = self.paras_tp_group.device_group
        sampler.force_sync_token_ids = True

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
        assert self.paras_dp_size == 1, "paras_configure_ep only supports dp_size==1"
        self._paras_drain_overlap_pipeline()
        torch.cuda.synchronize()

        if self._paras_auto_policy is not None:
            self._paras_auto_clear_window_on_switch()

        self.paras_start_profile("/tmp/paras_configure_profile")

        # Phase 1: Prepare — reset tree cache, merge batches, build global req list
        self.tree_cache.reset()
        self.merge_last_batch()
        global_reqs = list(self.running_batch.reqs) if self.running_batch else []
        local_waiting_reqs = (
            list(self.waiting_queue) if self.paras_tp_rank == 0 else []
        )

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

            # Capacity precheck (TP->EP direction). Defensive parity with
            # the EP->TP precheck: although the current Qwen3+TP=8 config
            # makes the EP pool >= the TP pool, a different model / memory
            # config could make EP pool axes smaller and the per-rank
            # asserts in the scatter manager's reorchestrate_cache would
            # then crash the scheduler the same way the gather path did.
            # Checks the worst-loaded rank using the deterministic
            # partition so every rank arrives at the same accept/reject
            # decision without a collective. The autoswitch policy's
            # cooldown was already extended by
            # _paras_auto_clear_window_on_switch above, so natural retry
            # spacing applies.
            ep_precheck_ok, ep_precheck_reason, ep_precheck_details = (
                paras_scatter_manager.precheck_ep_capacity()
            )
            if not ep_precheck_ok:
                logger.warning(
                    "ParaS TP->EP switch rejected by precheck: %s | "
                    "max_tokens_per_rank=%d new_ep_cache_size=%d "
                    "max_reqs_per_rank=%d new_req_pool_size=%d. "
                    "Staying in TP; autoswitch will retry after cooldown.",
                    ep_precheck_reason,
                    ep_precheck_details["max_tokens_per_rank"],
                    ep_precheck_details["new_ep_cache_size"],
                    ep_precheck_details["max_reqs_per_rank"],
                    ep_precheck_details["new_req_pool_size"],
                )
                self._paras_restore_tp_state_after_aborted_ep_switch()
                return True

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

        sampler = self.tp_worker.model_runner.sampler
        sampler.tp_sync_group = self.paras_tp_attn_tp_group.device_group
        sampler.force_sync_token_ids = False

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
        # A successful switch swaps self.send_to_tokenizer to the post-switch
        # variant (tp_send_to_tokenizer or ep_send_to_tokenizer) BEFORE
        # process_input_requests auto-emits this handler's return value, and
        # that swap silently drops the emission on non-designated ranks via
        # SenderWrapper(None) so the reply count matches the cardinality
        # TokenizerManager.paras_configure_{tp,ep} pre-set on the communicator
        # (fan_out = paras_dp_size for TP-target; fan_out = server_args.dp_size
        # for EP-target). A precheck-rejected switch returns BEFORE that swap,
        # so the auto-emit would fire from every rank under the current
        # (pre-switch) sender topology and overflow the communicator's
        # _result_values=None state after the first reply lands -- crashing
        # TokenizerManager at tokenizer_communicator_mixin.py:163 with
        # AttributeError on .append. To mirror the success path's
        # send-suppression on the rejection path, emit the reply ourselves
        # via the POST-target-mode sender (whose null-wrapper distribution
        # naturally gates emissions to the expected count) and return None so
        # process_input_requests does not auto-emit a second time.
        if recv_req.type == ParaSConfigureReqType.CONFIGURE_TP:
            rejected = self.paras_configure_tp()
            if rejected:
                self.tp_send_to_tokenizer.send_output(
                    ParaSConfigureReqOutput(), recv_req
                )
                return None
        elif recv_req.type == ParaSConfigureReqType.CONFIGURE_EP:
            rejected = self.paras_configure_ep()
            if rejected:
                self.ep_send_to_tokenizer.send_output(
                    ParaSConfigureReqOutput(), recv_req
                )
                return None
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