# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import time
from abc import ABC
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.distributed

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

_OutputMode = Literal["file", "object"]

# ---------------------------------------------------------------------------
# Global GPU buffer for local-token counting.
#
# FusedMoE.forward() writes to this buffer using **pure tensor ops** so the
# writes are captured inside torch.compile / CUDA-graph regions.  After each
# model-forward step the model-runner calls ``capture_step`` on the recorder
# which async-copies the buffer to CPU and appends it to the per-step list.
# ---------------------------------------------------------------------------
_local_tokens_gpu_buffer: Optional[torch.Tensor] = None  # shape [num_layers], int32


def get_local_tokens_gpu_buffer() -> Optional[torch.Tensor]:
    return _local_tokens_gpu_buffer


def init_local_tokens_gpu_buffer(num_layers: int, device: str = "cuda"):
    global _local_tokens_gpu_buffer
    _local_tokens_gpu_buffer = torch.zeros(num_layers, dtype=torch.int32, device=device)


class MoEKernelBalanceRecorder(ABC):
    """Records MoE kernel execution time per layer per forward step.

    Usage from each MoE layer's forward method::

        recorder = get_global_moe_kernel_balance_recorder()
        recorder.record_start(self.layer_id)
        output = self.run_moe_core(...)
        recorder.record_end(self.layer_id)

    Step boundaries are detected automatically: ``layer_idx == 0`` signals the
    start of a new forward step (finalizing the previous one).

    ``set_forward_mode`` should be called once per forward pass (e.g. from
    the model runner) so that dump-time filtering to decode-only steps works.

    ``capture_step`` should be called from the model runner **after** each
    forward pass.  It async-copies the GPU local-tokens buffer to CPU and
    stores the snapshot for the current step.

    At dump time, gathers data from all ranks, filters to steps where all ranks
    are in decode mode, and produces a 3D tensor of shape
    [#all_decode_steps, #layers, #EP_ranks].
    """

    @staticmethod
    def init_new(
        num_layers: int,
        rank: int,
        world_size: int,
        enabled: bool = False,
    ) -> "MoEKernelBalanceRecorder":
        if enabled:
            return _MoEKernelBalanceRecorderReal(num_layers, rank, world_size)
        return _MoEKernelBalanceRecorderNoop()

    def set_forward_mode(self, forward_mode: ForwardMode):
        pass

    def record_start(self):
        pass

    def record_end(self):
        pass

    def capture_step(self, batch_size: int = 0):
        """Called from model_runner after each forward to snapshot the GPU buffer."""
        pass

    def start_record(self):
        pass

    def stop_record(self):
        pass

    def dump(self, output_mode: _OutputMode = "file"):
        pass

    @property
    def recording(self):
        return False


class _MoEKernelBalanceRecorderNoop(MoEKernelBalanceRecorder):
    pass


class _MoEKernelBalanceRecorderReal(MoEKernelBalanceRecorder):
    def __init__(self, num_layers: int, rank: int, world_size: int):
        self._num_layers = num_layers
        self._rank = rank
        self._world_size = world_size
        self._recording = False

        self._forward_modes: List[int] = []
        self._batch_sizes: List[int] = []
        self._timestamps: List[float] = []  # wall-clock time.time() per step
        self._local_tokens_per_step: List[torch.Tensor] = []  # CPU int32 [num_layers]
        self._step_events: List[tuple] = []  # (start_event, end_event) per step

        self._current_forward_mode_value: int = -1
        self._pending_start_event: Optional[torch.cuda.Event] = None

    def set_forward_mode(self, forward_mode: ForwardMode):
        if self._recording:
            self._current_forward_mode_value = forward_mode.value

    def record_start(self):
        if not self._recording:
            return
        self._timestamps.append(time.time())
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._pending_start_event = start_event

    def record_end(self):
        if not self._recording:
            return
        if self._pending_start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._step_events.append((self._pending_start_event, end_event))
        self._pending_start_event = None

    def capture_step(self, batch_size: int = 0):
        """Snapshot the GPU local-tokens buffer after each forward pass.

        Called from model_runner.forward() AFTER the model forward completes.
        The GPU buffer was written by FusedMoE.forward() tensor ops (CUDA-graph
        safe).  We do a D2H copy here (outside the graph).
        """
        if not self._recording:
            return
        gpu_buf = get_local_tokens_gpu_buffer()
        if gpu_buf is None:
            return
        # D2H copy — non_blocking since we only need the value at dump time
        cpu_snapshot = gpu_buf.to("cpu", non_blocking=True)
        self._forward_modes.append(self._current_forward_mode_value)
        self._batch_sizes.append(batch_size)
        self._local_tokens_per_step.append(cpu_snapshot)

    def start_record(self):
        self._recording = True
        self._reset()

    def stop_record(self):
        self._recording = False

    def _reset(self):
        self._forward_modes.clear()
        self._batch_sizes.clear()
        self._timestamps.clear()
        self._local_tokens_per_step.clear()
        self._step_events.clear()
        self._current_forward_mode_value = -1
        self._pending_start_event = None

    @property
    def recording(self):
        return self._recording

    def dump(self, output_mode: _OutputMode = "file"):
        num_steps = len(self._forward_modes)
        device = "cuda"

        torch.cuda.synchronize()

        # --- Per-step timing from events ---
        # record_start / record_end are called from model_runner.forward()
        # (outside the compiled / CUDA-graph region) so every step has an
        # event pair.  They are 1:1 aligned with capture_step entries.
        local_times_cpu = torch.zeros(num_steps, dtype=torch.float32)
        num_event_steps = min(len(self._step_events), num_steps)
        for step_idx in range(num_event_steps):
            start_evt, end_evt = self._step_events[step_idx]
            local_times_cpu[step_idx] = start_evt.elapsed_time(end_evt)

        # --- Gather across ranks ---
        local_num_steps = torch.tensor([num_steps], dtype=torch.int64, device=device)
        all_num_steps_list = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_num_steps_list, local_num_steps)
        max_steps = int(max(s.item() for s in all_num_steps_list))

        if max_steps == 0:
            logger.warning(
                "MoEKernelBalanceRecorder: no steps recorded, skipping dump."
            )
            return

        local_modes = torch.full(
            (max_steps,), fill_value=-1, dtype=torch.int32, device=device
        )
        local_times = torch.zeros(max_steps, dtype=torch.float32, device=device)
        local_bsz = torch.zeros((max_steps,), dtype=torch.int32, device=device)
        local_ltok = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.int32, device=device
        )
        local_ts = torch.zeros(max_steps, dtype=torch.float64, device=device)

        if num_steps > 0:
            local_modes[:num_steps] = torch.tensor(
                self._forward_modes, dtype=torch.int32, device=device
            )
            local_times[:num_steps] = local_times_cpu.to(device)
            local_bsz[:num_steps] = torch.tensor(
                self._batch_sizes, dtype=torch.int32, device=device
            )
            # Wall-clock timestamps (time.time() recorded at record_start)
            num_ts = min(len(self._timestamps), num_steps)
            if num_ts > 0:
                local_ts[:num_ts] = torch.tensor(
                    self._timestamps[:num_ts], dtype=torch.float64, device=device
                )
            # Stack CPU snapshots from capture_step (already CPU tensors)
            if self._local_tokens_per_step:
                ltok_stacked = torch.stack(self._local_tokens_per_step)  # [steps, layers]
                local_ltok[:num_steps] = ltok_stacked.to(device=device, dtype=torch.int32)

        all_modes_list = [
            torch.zeros_like(local_modes) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_modes_list, local_modes)
        all_modes = torch.stack(all_modes_list)

        decode_value = ForwardMode.DECODE.value
        all_decode_mask = (all_modes == decode_value).all(dim=0)

        all_times_list = [
            torch.zeros_like(local_times) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_times_list, local_times)
        all_times = torch.stack(all_times_list)  # [world_size, max_steps]

        all_bsz_list = [
            torch.zeros_like(local_bsz) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_bsz_list, local_bsz)
        all_bsz = torch.stack(all_bsz_list)

        all_ltok_list = [
            torch.zeros_like(local_ltok) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_ltok_list, local_ltok)
        all_ltok = torch.stack(all_ltok_list)

        all_ts_list = [
            torch.zeros_like(local_ts) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_ts_list, local_ts)
        all_ts = torch.stack(all_ts_list)  # [world_size, max_steps]

        # moe_times: [decode_steps, world_size]  (per-step total forward time)
        decode_times = all_times[:, all_decode_mask]  # [world_size, decode_steps]
        result = decode_times.permute(1, 0).contiguous()  # [decode_steps, world_size]
        decode_bsz = all_bsz[:, all_decode_mask].permute(1, 0).contiguous()
        decode_ltok = all_ltok[:, all_decode_mask, :]
        decode_ltok = decode_ltok.permute(1, 2, 0).contiguous()
        decode_ts = all_ts[:, all_decode_mask].permute(1, 0).contiguous()
        # decode_ts: [decode_steps, world_size] — wall-clock time per rank per step

        output = dict(
            rank=self._rank,
            moe_times=result,
            batch_sizes=decode_bsz,
            local_token_counts=decode_ltok,
            timestamps=decode_ts,
            num_total_steps=max_steps,
            num_decode_steps=result.shape[0],
        )

        self._reset()

        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"moe_kernel_balance_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError


def _dump_to_file(name: str, data):
    save_dir = Path(envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.get())
    path_output = save_dir / name
    logger.info(f"Write MoE kernel balance data to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


_global_moe_kernel_balance_recorder: Optional[MoEKernelBalanceRecorder] = (
    _MoEKernelBalanceRecorderNoop()
)


def get_global_moe_kernel_balance_recorder() -> MoEKernelBalanceRecorder:
    return _global_moe_kernel_balance_recorder


def set_global_moe_kernel_balance_recorder(value: MoEKernelBalanceRecorder):
    global _global_moe_kernel_balance_recorder
    _global_moe_kernel_balance_recorder = value
