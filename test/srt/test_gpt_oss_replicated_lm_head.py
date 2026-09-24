"""CPU checks for opt-in head replication and unchanged default layouts."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.logits_processor as logits
import sglang.srt.layers.vocab_parallel_embedding as embedding
import sglang.srt.models.gpt_oss as gpt_oss


class ReplicatedLMHeadTest(unittest.TestCase):
    def test_weight_loading_and_projection_across_eight_ranks(self):
        generator = torch.Generator().manual_seed(42)
        weights = torch.randn(512, 16, generator=generator, dtype=torch.bfloat16)
        hidden = torch.randn(3, 16, generator=generator, dtype=torch.bfloat16)
        shards = []
        for rank in range(8):
            with (
                self.subTest(rank=rank),
                patch.object(
                    embedding, "get_tensor_model_parallel_world_size", return_value=8
                ),
                patch.object(
                    embedding, "get_tensor_model_parallel_rank", return_value=rank
                ),
            ):
                normal = embedding.ParallelLMHead(512, 16, params_dtype=torch.bfloat16)
                replicated = embedding.ParallelLMHead(
                    512, 16, params_dtype=torch.bfloat16, enable_tp=False
                )
                normal.weight_loader(normal.weight, weights)
                replicated.weight_loader(replicated.weight, weights)
                self.assertEqual(normal.weight.shape, (64, 16))
                self.assertEqual(replicated.weight.shape, (512, 16))
                torch.testing.assert_close(
                    normal.weight, weights[rank * 64 : (rank + 1) * 64], rtol=0, atol=0
                )
                torch.testing.assert_close(replicated.weight, weights, rtol=0, atol=0)
                shards.append(hidden @ normal.weight.T)
                torch.testing.assert_close(
                    hidden @ replicated.weight.T, hidden @ weights.T, rtol=0, atol=0
                )
        torch.testing.assert_close(
            torch.cat(shards, dim=-1), hidden @ weights.T, rtol=0, atol=0
        )

    def model_context(self, env=None, dp=False, dp_head=False):
        stack = ExitStack()
        self.addCleanup(stack.close)
        env_values = dict(os.environ)
        env_values.pop("SGLANG_GPTOSS_REPLICATED_LM_HEAD", None)
        if env is not None:
            env_values["SGLANG_GPTOSS_REPLICATED_LM_HEAD"] = env
        stack.enter_context(patch.dict(os.environ, env_values, clear=True))
        args = SimpleNamespace(
            enable_dp_attention=dp, enable_dp_lm_head=dp_head, enable_fp32_lm_head=False
        )
        stack.enter_context(
            patch.object(gpt_oss, "get_global_server_args", return_value=args)
        )
        stack.enter_context(
            patch.object(logits, "get_global_server_args", return_value=args)
        )
        stack.enter_context(patch.object(gpt_oss, "get_pp_group"))
        for module in [embedding, logits]:
            stack.enter_context(
                patch.object(
                    module, "get_tensor_model_parallel_world_size", return_value=8
                )
            )
            stack.enter_context(
                patch.object(
                    module, "get_attention_tp_size", return_value=1 if dp else 8
                )
            )
        stack.enter_context(
            patch.object(embedding, "get_tensor_model_parallel_rank", return_value=0)
        )
        stack.enter_context(
            patch.object(embedding, "get_attention_tp_rank", return_value=0)
        )
        stack.enter_context(
            patch.object(logits, "get_attention_dp_size", return_value=8 if dp else 1)
        )
        constructor = stack.enter_context(
            patch.object(gpt_oss, "GptOssModel", return_value=torch.nn.Module())
        )
        return constructor

    def test_model_default_and_explicitly_disabled_keep_sharding(self):
        for value in [None, "0"]:
            with self.subTest(value=value):
                self.model_context(env=value)
                model = gpt_oss.GptOssForCausalLM(
                    SimpleNamespace(vocab_size=512, hidden_size=16)
                )
                self.assertEqual(model.lm_head.weight.shape, (64, 16))
                self.assertTrue(model.logits_processor.do_tensor_parallel_all_gather)

    def test_model_opt_in_loads_full_head_without_gather(self):
        self.model_context(env="1")
        model = gpt_oss.GptOssForCausalLM(
            SimpleNamespace(vocab_size=512, hidden_size=16)
        )
        self.assertEqual(model.lm_head.weight.shape, (512, 16))
        self.assertFalse(model.logits_processor.do_tensor_parallel_all_gather)

    def test_existing_dp_head_configuration_is_unchanged(self):
        self.model_context(dp=True, dp_head=True)
        model = gpt_oss.GptOssForCausalLM(
            SimpleNamespace(vocab_size=512, hidden_size=16)
        )
        self.assertEqual(model.lm_head.weight.shape, (512, 16))
        self.assertTrue(model.lm_head.enable_tp)
        self.assertTrue(model.logits_processor.use_attn_tp_group)
        self.assertFalse(model.logits_processor.do_tensor_parallel_all_gather)

    def test_opt_in_rejects_unsupported_modes_before_model_allocation(self):
        for dp, dp_head, quant in [
            (True, False, None),
            (False, True, None),
            (False, False, object()),
        ]:
            with self.subTest(dp=dp, dp_head=dp_head, quantized=quant is not None):
                constructor = self.model_context(env="1", dp=dp, dp_head=dp_head)
                with self.assertRaisesRegex(
                    ValueError, "SGLANG_GPTOSS_REPLICATED_LM_HEAD requires"
                ):
                    gpt_oss.GptOssForCausalLM(
                        SimpleNamespace(vocab_size=512, hidden_size=16),
                        quant_config=quant,
                    )
                constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
