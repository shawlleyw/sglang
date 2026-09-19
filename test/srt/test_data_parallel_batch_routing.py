import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
)


class BatchRoutingTest(unittest.TestCase):
    def controller(self, workers=8):
        c = DataParallelController.__new__(DataParallelController)
        c.server_args = SimpleNamespace(enable_trace=False, disaggregation_mode="null")
        c.workers = [Mock() for _ in range(workers)]
        c.control_message_step = 8
        c.round_robin_counter = 0
        c.dispatching = c.round_robin_scheduler
        c.init_dispatcher()
        return c

    def test_generate_and_embedding_batches_balance_across_all_ranks(self):
        for kind in (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput):
            with self.subTest(kind=kind):
                c = self.controller()
                reqs = [
                    SimpleNamespace(data_parallel_rank=None, rid=str(i))
                    for i in range(512)
                ]
                c._request_dispatcher(kind(batch=reqs))
                self.assertEqual([w.send_pyobj.call_count for w in c.workers], [64] * 8)
                for rank, worker in enumerate(c.workers):
                    self.assertEqual(
                        [call.args[0].rid for call in worker.send_pyobj.call_args_list],
                        [str(i) for i in range(rank, 512, 8)],
                    )

    def test_explicit_rank_and_tp_mode(self):
        c = self.controller()
        reqs = [
            SimpleNamespace(data_parallel_rank=7, rid="explicit"),
            SimpleNamespace(data_parallel_rank=None, rid="balanced"),
        ]
        c._request_dispatcher(BatchTokenizedGenerateReqInput(batch=reqs))
        c.workers[7].send_pyobj.assert_called_once_with(reqs[0])
        c.workers[0].send_pyobj.assert_called_once_with(reqs[1])
        c = self.controller(workers=1)
        c._request_dispatcher(BatchTokenizedGenerateReqInput(batch=[reqs[1]] * 8))
        self.assertEqual(c.workers[0].send_pyobj.call_count, 8)


if __name__ == "__main__":
    unittest.main()
