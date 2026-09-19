"""Streaming must retain every delta when consumers run behind producers."""

import asyncio
import dataclasses
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.managers.io_struct import BatchStrOutput, BatchTokenIDOutput
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager


def manager():
    m = TokenizerManager.__new__(TokenizerManager)
    m.server_args = SimpleNamespace(
        weight_version="test",
        stream_output=False,
        speculative_algorithm=None,
        enable_lora=False,
        skip_tokenizer_init=False,
    )
    m.rid_to_state = {}
    m.log_requests = m.enable_metrics = False
    m.dump_requests_folder = m.crash_dump_folder = None
    return m


def state(m, rid="a", stream=True):
    obj = SimpleNamespace(
        rid=rid,
        stream=stream,
        background=False,
        return_logprob=False,
        log_metrics=False,
    )
    s = ReqState([], False, asyncio.Event(), obj, time.time())
    m.rid_to_state[rid] = s
    return s


def emit(m, s, text, ids, final=False, raw=False):
    cls = BatchTokenIDOutput if raw else BatchStrOutput
    b = cls.__new__(cls)
    for field in dataclasses.fields(cls):
        setattr(b, field.name, None)
    b.rids = [s.obj.rid]
    b.finished_reasons = [{"type": "length"} if final else None]
    b.prompt_tokens = [3]
    b.retraction_counts = b.cached_tokens = [0]
    b.output_strs = [text]
    b.output_ids = [ids]
    b.completion_tokens = [len(s.output_ids) + len(ids)]
    m._handle_batch_output(b)


class TestStreamingBackpressure(unittest.IsolatedAsyncioTestCase):
    async def test_coalesced_text_and_speculative_tokens(self):
        m = manager()
        s = state(m)
        emit(m, s, "Hello", [1, 2])
        emit(m, s, " 世界", [3, 4])
        self.assertEqual(len(s.out_list), 1)
        gen = m._wait_one_response(s.obj, s)
        first = await anext(gen)
        self.assertEqual(first["text"], "Hello 世界")
        self.assertEqual(first["output_ids"], [1, 2, 3, 4])
        emit(m, s, "!", [5], final=True)
        last = await anext(gen)
        self.assertEqual(last["text"], "!")
        self.assertEqual(last["output_ids"], [5])
        self.assertEqual(last["meta_info"]["completion_tokens"], 5)
        with self.assertRaises(StopAsyncIteration):
            await anext(gen)
        self.assertEqual(first["output_ids"], [1, 2, 3, 4])

    async def test_final_arrives_before_first_read(self):
        m = manager()
        s = state(m)
        emit(m, s, "a", [1])
        emit(m, s, "b", [2], final=True)
        gen = m._wait_one_response(s.obj, s)
        out = await anext(gen)
        self.assertEqual((out["text"], out["output_ids"]), ("ab", [1, 2]))
        await gen.aclose()

    async def test_nonstream_still_returns_full_response(self):
        m = manager()
        s = state(m, stream=False)
        emit(m, s, "a", [1])
        emit(m, s, "b", [2], final=True)
        gen = m._wait_one_response(s.obj, s)
        out = await anext(gen)
        self.assertEqual((out["text"], out["output_ids"]), ("ab", [1, 2]))
        await gen.aclose()

    async def test_raw_ids_delta_and_cumulative_modes(self):
        for delta in [False, True]:
            m = manager()
            m.server_args.stream_output = delta
            s = state(m)
            gen = m._wait_one_response(s.obj, s)
            emit(m, s, "", [1], raw=True)
            self.assertEqual((await anext(gen))["output_ids"], [1])
            emit(m, s, "", [2], raw=True)
            emit(m, s, "", [3], final=True, raw=True)
            self.assertEqual(
                (await anext(gen))["output_ids"], [2, 3] if delta else [1, 2, 3]
            )
            await gen.aclose()

    async def test_abort_flushes_pending_without_repeating_delivered_tokens(self):
        m = manager()
        s = state(m)
        gen = m._wait_one_response(s.obj, s)
        emit(m, s, "a", [1])
        await anext(gen)
        emit(m, s, "b", [2])
        m._handle_abort_req(
            SimpleNamespace(rid="a", abort_message="stop", finished_reason=None)
        )
        out = await anext(gen)
        self.assertEqual((out["text"], out["output_ids"]), ("b", [2]))
        self.assertEqual(out["meta_info"]["finish_reason"]["type"], "abort")
        await gen.aclose()

    async def test_raw_abort_preserves_cumulative_contract(self):
        m = manager()
        m.server_args.skip_tokenizer_init = True
        s = state(m)
        gen = m._wait_one_response(s.obj, s)
        emit(m, s, "", [1], raw=True)
        await anext(gen)
        emit(m, s, "", [2], raw=True)
        m._handle_abort_req(
            SimpleNamespace(rid="a", abort_message="stop", finished_reason=None)
        )
        out = await anext(gen)
        self.assertNotIn("text", out)
        self.assertEqual(out["output_ids"], [1, 2])
        await gen.aclose()

    async def test_batch_ready_queue_order_finish_and_cleanup(self):
        m = manager()
        objs = [
            SimpleNamespace(
                rid=str(i),
                stream=True,
                background=False,
                return_logprob=False,
                log_metrics=False,
            )
            for i in range(3)
        ]

        class Batch:
            batch_size = 3
            stream = True
            background = False
            parallel_sample_num = 1

            def __getitem__(self, i):
                return objs[i]

        states = []
        m._should_use_batch_tokenization = lambda *args: True
        m._batch_tokenize_and_process = AsyncMock(return_value=objs)

        def send(*args):
            states.extend(state(m, str(i)) for i in range(3))

        m._send_batch_request = send
        gen = m._handle_batch_request(Batch())

        async def consume():
            return [out async for out in gen]

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        for i in [2, 0, 1]:
            emit(m, states[i], "a", [1])
            emit(m, states[i], "b", [2], final=True)
        results = await asyncio.wait_for(task, timeout=1)
        self.assertEqual([r["index"] for r in results], [2, 0, 1])
        self.assertTrue(
            all(r["text"] == "ab" and r["output_ids"] == [1, 2] for r in results)
        )
        self.assertTrue(all(s.response_queue is None for s in states))

    async def test_cancelled_batch_releases_ready_queue(self):
        m = manager()

        class Batch:
            batch_size = 1
            stream = True
            background = False
            parallel_sample_num = 1

            def __getitem__(self, index):
                return SimpleNamespace(rid="a")

        states = []
        m._should_use_batch_tokenization = lambda *args: True
        m._batch_tokenize_and_process = AsyncMock(return_value=[])

        def send(*args):
            states.append(state(m))

        m._send_batch_request = send
        gen = m._handle_batch_request(Batch())
        task = asyncio.create_task(anext(gen))
        await asyncio.sleep(0)
        self.assertIsNotNone(states[0].response_queue)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(states[0].response_queue)
        await gen.aclose()

    async def test_ready_queue_coalesces_and_requeues(self):
        m = manager()
        s = state(m)
        s.response_queue = asyncio.Queue()
        s.response_index = 7
        emit(m, s, "a", [1])
        emit(m, s, "b", [2])
        self.assertEqual(s.response_queue.qsize(), 1)
        self.assertEqual(s.response_queue.get_nowait(), 7)
        gen = m._wait_one_response(s.obj, s)
        await anext(gen)
        emit(m, s, "c", [3], final=True)
        self.assertEqual(s.response_queue.get_nowait(), 7)
        self.assertEqual((await anext(gen))["text"], "c")
        await gen.aclose()


class TestNativeBatchDecode(unittest.TestCase):
    def tokenizer(self, cls=None, cleanup=False):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "hello": 1, "world": 2, ".": 3, "[PAD]": 4},
                unk_token="[UNK]",
            )
        )
        return (cls or PreTrainedTokenizerFast)(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            clean_up_tokenization_spaces=cleanup,
        )

    def check(self, tokenizer):
        from sglang.srt.managers.detokenizer_manager import DetokenizerManager

        m = DetokenizerManager.__new__(DetokenizerManager)
        m.tokenizer = tokenizer
        inputs = [[], [1], [1, 2, 3], [4, 1, 0]]
        for skip in [False, True]:
            expected = tokenizer.batch_decode(
                inputs, skip_special_tokens=skip, spaces_between_special_tokens=True
            )
            self.assertEqual(m.batch_decode(inputs, skip, True), expected)

    def test_native_preserves_special_tokens_and_cleanup(self):
        for cleanup in [False, True]:
            self.check(self.tokenizer(cleanup=cleanup))

    def test_custom_decode_falls_back(self):
        from transformers import PreTrainedTokenizerFast

        class Custom(PreTrainedTokenizerFast):
            def _decode(self, *args, **kwargs):
                return "custom:" + super()._decode(*args, **kwargs)

        self.check(self.tokenizer(cls=Custom))

    def test_custom_public_decode_falls_back(self):
        from transformers import PreTrainedTokenizerFast

        class Custom(PreTrainedTokenizerFast):
            def decode(self, *args, **kwargs):
                return "public:" + super().decode(*args, **kwargs)

        self.check(self.tokenizer(cls=Custom))

    def test_custom_batch_decode_falls_back(self):
        from transformers import PreTrainedTokenizerFast

        class Custom(PreTrainedTokenizerFast):
            def batch_decode(self, sequences, **kwargs):
                return ["custom batch"] * len(sequences)

        self.check(self.tokenizer(cls=Custom))


if __name__ == "__main__":
    unittest.main()
