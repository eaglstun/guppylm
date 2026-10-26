"""Checks the optional cache against the ordinary sliding-window path."""

import unittest

import torch

from guppylm.config import GuppyConfig
from guppylm.model import GuppyLM


class CachedGenerationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = GuppyConfig(vocab_size=61, max_seq_len=12, d_model=32,
                                  n_layers=2, n_heads=4, ffn_hidden=64,
                                  dropout=0.0, eos_id=-1)
        self.model = GuppyLM(self.config).eval()

    def test_prefill_and_incremental_logits(self):
        tokens = torch.randint(3, self.config.vocab_size, (2, 9))
        with torch.no_grad():
            for cut in (1, 3, 6):
                logits, past = self.model.forward_cached(tokens[:, :cut])
                expected, _ = self.model(tokens[:, :cut])
                torch.testing.assert_close(logits, expected, atol=2e-5, rtol=2e-5)
                for end in range(cut + 1, tokens.size(1) + 1):
                    logits, past = self.model.forward_cached(tokens[:, end - 1:end], past)
                    expected, _ = self.model(tokens[:, :end])
                    torch.testing.assert_close(logits[:, -1], expected[:, -1],
                                               atol=2e-5, rtol=2e-5)

    def test_generation_after_window_overflow(self):
        prompt = torch.tensor([[4, 5, 6, 7]])
        with torch.no_grad():
            regular, _ = self.model.generate(prompt, max_new_tokens=20, top_k=1)
            cached, _ = self.model.generate(prompt, max_new_tokens=20, top_k=1,
                                             use_cache=True)
        torch.testing.assert_close(cached, regular, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
