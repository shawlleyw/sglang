import logging
from http import HTTPStatus
from typing import Optional, List, Union, Dict

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
import time
import torch
import numpy as np
import os
import pickle
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            logger.error(error_msg)
            req.finished_reason = FINISH_ABORT(
                error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
            )
            return error_msg

    return None

@dataclass
class StepMetrics:
    batch_size: int
    attention_elapse: List[float]
    all_gather_elapse: List[float]
    moe_elapse: List[float]
    moe_num_tokens_per_local_expert: List[List[int]]
    
@dataclass
class TokenStepMetrics:
    """For tokenizer and detokenizer"""
    batch_size: int
    t_elapse: float  # For tokenizer, this includes sampling & sending request; for detokenizer, this includes `batch_decode` & `send_pyobj`
    t_wait: Optional[float] # tokenizer only, the time for `Wait for all requests`
        
class StepRecorder:
    
    def __init__(self, batch_size):
        
        self.batch_size = batch_size
        self.attention_start_timestamp: List[torch.cuda.Event] = []
        self.attention_end_timestamp: List[torch.cuda.Event] = []
        
        self.all_gather_start_timestamp: List[torch.cuda.Event] = []
        self.all_gather_end_timestamp: List[torch.cuda.Event] = []
        
        self.moe_start_timestamp: List[torch.cuda.Event] = []
        self.moe_end_timestamp: List[torch.cuda.Event] = []
        self.moe_num_tokens_per_local_expert: List[torch.Tensor] = []
        
    def mark_attention_start(self):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.attention_start_timestamp.append(start)
        
    def mark_attention_end(self):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.attention_end_timestamp.append(end)
        
    def mark_moe_start(self):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.moe_start_timestamp.append(start)
        
    def mark_moe_end(self):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.moe_end_timestamp.append(end)
        
    def mark_all_gather_start(self):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.all_gather_start_timestamp.append(start)
        
    def mark_all_gather_end(self):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.all_gather_end_timestamp.append(end)
        
    def record_moe_num_tokens_per_local_expert(self, num_tokens_per_local_expert: torch.Tensor):
        self.moe_num_tokens_per_local_expert.append(num_tokens_per_local_expert)
    
    def post_process(self) -> StepMetrics:
        torch.cuda.synchronize()
        attn_elapse = [start.elapsed_time(end) for start, end in zip(self.attention_start_timestamp, self.attention_end_timestamp)]
        all_gather_elapse = [start.elapsed_time(end) for start, end in zip(self.all_gather_start_timestamp, self.all_gather_end_timestamp)]
        moe_elapse = [start.elapsed_time(end) for start, end in zip(self.moe_start_timestamp, self.moe_end_timestamp)]
        moe_num_tokens_per_local_expert = [x.view(-1).tolist() for x in self.moe_num_tokens_per_local_expert]
        return StepMetrics(self.batch_size, attn_elapse, all_gather_elapse, moe_elapse, moe_num_tokens_per_local_expert)
        
cur_step_runtime_recorder: StepRecorder = None

metrics_list: List[Union[StepMetrics, TokenStepMetrics]] = None

_last_detoken_time: Dict[int, int] = {}

detoken_itl: Dict[int, List[float]] = {}

def start_metrics():
    global metrics_list, detoken_itl
    metrics_list = []
    detoken_itl = {}

def stop_metrics(rank, name="scheduler"):
    global metrics_list, detoken_itl
    sglang_metrics_dir = os.getenv("SGLANG_METRICS_DIR", ".")
    if name == "scheduler":
        # align previous behavior, won't change file name
        fn = f"{sglang_metrics_dir}/sglang_metrics_rank{rank}"
    else:
        # tokenizer & detokenizer, no rank
        fn = f"{sglang_metrics_dir}/sglang_metrics_{name}"
    with open(f"{fn}.pickle", "wb") as f:
        pickle.dump(metrics_list, f)
    if name == "detokenizer":
        # save detoken_itl
        with open(f"{fn}_detoken_itl.pickle", "wb") as f:
            pickle.dump(detoken_itl, f)
    metrics_list = None
    detoken_itl = None
    
def append_one_metric(metric):
    # print("append one metric:", metric)
    global metrics_list
    if metrics_list is not None:
        metrics_list.append(metric)

def append_detok_rids(rids: List[int]):
    global _last_detoken_time, metrics_list, detoken_itl
    if metrics_list is None:
        return
    st = time.perf_counter()
    for rid in rids:
        if rid not in _last_detoken_time:
            _last_detoken_time[rid] = st
        else:
            dt = st - _last_detoken_time[rid]
            if rid not in detoken_itl:
                detoken_itl[rid] = []
            detoken_itl[rid].append(dt)
            _last_detoken_time[rid] = st