import unittest

import mlx.core as mx
from mlx_lm.sample_utils import (
    BeamSearchSampler,
    min_p_sampling,
    top_k_sampling,
    top_p_sampling,
)


class TestSampleUtils(unittest.TestCase):
    def test_top_p_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        temperature = 1.0

        token = top_p_sampling(logits, 0.3, temperature).item()
        self.assertEqual(token, 0)

        token = top_p_sampling(logits, 0.95, temperature).item()
        self.assertTrue(token in (0, 3))

        probs = mx.array([0.0, 0.5, 0.4, 0.1])[None]
        logits = mx.log(probs)

        token = top_p_sampling(logits, 0.4, temperature).item()
        self.assertEqual(token, 1)

        token = top_p_sampling(logits, 0.6, temperature).item()
        self.assertTrue(token in (1, 2))

        token = top_p_sampling(logits, 0.95, temperature).item()
        self.assertTrue(token in (1, 2, 3))

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)
        tokens = top_p_sampling(logits, 0.5, temperature)
        self.assertEqual(tokens.tolist(), [0, 1])

    def test_min_p_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        temperature = 1.0
        token = min_p_sampling(logits, 0.8)
        self.assertEqual(token, 0)

        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        temperature = 1.0
        for _ in range(5):
            token = min_p_sampling(logits, 0.05)
            self.assertTrue(token in (0, 3))

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)
        tokens = min_p_sampling(logits, 0.7)
        self.assertEqual(tokens.tolist(), [0, 1])

    def test_top_k_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        token = top_k_sampling(logits, 1).item()
        self.assertEqual(token, 0)

        probs = mx.array([0.5, 0.0, 0.0, 0.5])[None]
        tokens = set()
        for _ in range(100):
            token = top_k_sampling(logits, 2)
            tokens.add(token.item())
        self.assertEqual(tokens, {0, 3})

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)

        tokens = top_k_sampling(logits, 1)
        self.assertEqual(tokens.tolist(), [0, 1])


class TestBeamSearchSampler(unittest.TestCase):
    def test_beam_search(self):
        beams = 3
        beam_sampler = BeamSearchSampler(beams=beams)
        
        batch_size = 2
        vocab_size = 5
        
        # Corrected logits shape (batch_size * 1, vocab_size) for initial beams=1
        logits = mx.array([
            [0.1, 0.2, 0.3, 0.05, 0.15],  # First batch element
            [0.7, 0.08, 0.12, 0.05, 0.05]  # Second batch element
        ])
        
        # First step - initial beams=1, so seq_weights shape (batch_size * 1,)
        seq_weights = mx.zeros((batch_size,))
        
        logprobs = mx.log(mx.softmax(logits, axis=-1))
        
        # First call to beam sampler expands to beams=3
        next_tokens, ancestors, scores = beam_sampler(
            logprobs, seq_weights, None
        )
        
        # Check outputs after first step
        self.assertEqual(next_tokens.shape, (batch_size * beams, 1))
        self.assertEqual(ancestors.shape, (batch_size * beams,))
        self.assertEqual(scores.shape, (batch_size * beams,))
        
        # Second step uses previous scores and new logits for expanded beams
        new_logits = mx.array([
            [0.8, 0.1, 0.05, 0.025, 0.025],
            [0.2, 0.3, 0.3, 0.1, 0.1],
            [0.5, 0.2, 0.15, 0.1, 0.05],
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.3, 0.3, 0.2, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25, 0.0]
        ])  # Shape (6,5) for batch_size * beams=3
        
        logprobs_2 = mx.log(mx.softmax(new_logits, axis=-1))
        next_tokens_2, ancestors_2, scores_2 = beam_sampler(
            logprobs_2, scores, None
        )
        
        # Validate subsequent step
        self.assertEqual(next_tokens_2.shape, (batch_size * beams, 1))
        self.assertEqual(ancestors_2.shape, (batch_size * beams,))


if __name__ == "__main__":
    unittest.main()
