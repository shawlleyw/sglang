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


class MoEKernelBalanceRecorder(ABC):
    """Records MoE kernel execution time per layer per forward step.

    Usage from each MoE layer's forward method::

        recorder = get_global_moe_kernel_balance_recorder()
        recorder.record_start(self.layer_id)
        output = self.run_moe_core(...)
        recorder.record_end(self.layer_id)

    Step boundaries are detected automatically: ``layer_idx == 0`` signals the
    start of a new forward step (finalising the previous one).

    ``set_forward_mode`` should be called once per forward pass (e.g. from
    the model runner) so that dump-time filtering to decode-only steps works.

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
    ) -> MoEKernelBalanceRecorder:
        if enabled:
            return _MoEKernelBalanceRecorderReal(num_layers, rank, world_size)
        return _MoEKernelBalanceRecorderNoop()

    def set_forward_mode(self, forward_mode: ForwardMode):
        pass

    def record_start(self, layer_idx: int):
        pass

    def record_end(self, layer_idx: int):
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
        # Each entry is a list of num_layers elements, where each element is
        # either None (layer not recorded) or a (start_event, end_event) tuple.
        self._step_events: List[List] = []

        self._current_forward_mode_value: int = -1
        self._current_events: Optional[List] = None
        self._pending_start_event: Optional[torch.cuda.Event] = None

    def set_forward_mode(self, forward_mode: ForwardMode):
        if self._recording:
            self._current_forward_mode_value = forward_mode.value

    def record_start(self, layer_idx: int):
        if not self._recording:
            return
        if layer_idx == 0:
            self._finalize_step()
            self._current_events = [None] * self._num_layers
        if self._current_events is None:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._pending_start_event = start_event

    def record_end(self, layer_idx: int):
        if not self._recording or self._current_events is None:
            return
        if self._pending_start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._current_events[layer_idx] = (self._pending_start_event, end_event)
        self._pending_start_event = None

    def _finalize_step(self):
        if self._current_events is not None:
            self._forward_modes.append(self._current_forward_mode_value)
            self._step_events.append(self._current_events)
            self._current_events = None

    def start_record(self):
        self._recording = True
        self._reset()

    def stop_record(self):
        self._finalize_step()
        self._recording = False

    def _reset(self):
        self._forward_modes.clear()
        self._step_events.clear()
        self._current_forward_mode_value = -1
        self._current_events = None
        self._pending_start_event = None

    @property
    def recording(self):
        return self._recording

    def dump(self, output_mode: _OutputMode = "file"):
        self._finalize_step()
        num_steps = len(self._forward_modes)
        device = "cuda"

        # Single sync to ensure all recorded CUDA events have completed
        torch.cuda.synchronize()

        # Compute elapsed times (ms) from stored events -- cheap after sync
        local_times_cpu = torch.zeros(
            (num_steps, self._num_layers), dtype=torch.float32
        )
        for step_idx, events in enumerate(self._step_events):
            for layer_idx, event_pair in enumerate(events):
                if event_pair is not None:
                    start_evt, end_evt = event_pair
                    local_times_cpu[step_idx, layer_idx] = (
                        start_evt.elapsed_time(end_evt)
                    )

        # Synchronize step counts across ranks
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

        # Pad local data to max_steps for uniform all_gather
        local_modes = torch.full(
            (max_steps,), fill_value=-1, dtype=torch.int32, device=device
        )
        local_times = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.float32, device=device
        )

        if num_steps > 0:
            local_modes[:num_steps] = torch.tensor(
                self._forward_modes, dtype=torch.int32, device=device
            )
            local_times[:num_steps] = local_times_cpu.to(device)

        # Gather forward modes from all ranks -> [world_size, max_steps]
        all_modes_list = [
            torch.zeros_like(local_modes) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_modes_list, local_modes)
        all_modes = torch.stack(all_modes_list)

        # Steps where ALL ranks are in DECODE mode
        decode_value = ForwardMode.DECODE.value
        all_decode_mask = (all_modes == decode_value).all(dim=0)

        # Gather MoE times from all ranks -> [world_size, max_steps, num_layers]
        all_times_list = [
            torch.zeros_like(local_times) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_times_list, local_times)
        all_times = torch.stack(all_times_list)

        # Filter to all-decode steps and reshape to [#decode_steps, num_layers, world_size]
        decode_times = all_times[:, all_decode_mask, :]
        result = decode_times.permute(1, 2, 0).contiguous()

        output = dict(
            rank=self._rank,
            moe_times=result,
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
