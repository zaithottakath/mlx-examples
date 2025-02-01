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
        
        # Test sampler initialization
        beam_sampler = BeamSearchSampler(beams=beams)
        self.assertEqual(beam_sampler.samples, beams)
        
        # Test basic beam search step
        batch_size = 2
        vocab_size = 5
        logits = mx.array([[
            [0.1, 0.2, 0.3, 0.05, 0.15],
            [0.7, 0.08, 0.12, 0.05, 0.05]
        ]])
        
        # First step - sequence weights start at 0
        seq_weights = mx.zeros((batch_size * beams,))
        
        # First step generates beam candidates
        logprobs = mx.log(mx.softmax(logits, axis=-1))
        next_tokens, ancestors, scores = beam_sampler(
            logprobs, seq_weights, None
        )
        
        # Validate first step outputs
        self.assertEqual(next_tokens.shape, (batch_size * beams, 1))
        self.assertEqual(ancestors.shape, (batch_size * beams,))
        self.assertEqual(scores.shape, (batch_size * beams,))
        
        # After first step scores should be logprobs of selected tokens
        self.assertTrue(mx.allclose(scores, mx.array([0.3, 0.2, 0.15, 0.7, 0.08, 0.05])))
        
        # Test second step with accumulated scores
        new_logits = mx.array([[
            [0.8, 0.1, 0.05, 0.025, 0.025],
            [0.2, 0.3, 0.3, 0.1, 0.1]
        ]]).repeat(beams, axis=0)
        
        logprobs_2 = mx.log(mx.softmax(new_logits, axis=-1))
        next_tokens_2, ancestors_2, scores_2 = beam_sampler(
            logprobs_2, scores, None
        )
        
        # Validate beam tracking
        self.assertEqual(next_tokens_2.shape, (batch_size * beams, 1))
        self.assertEqual(ancestors_2.shape, (batch_size * beams,))
        
        # Verify score accumulation combines previous scores with new logprobs
        expected_scores = mx.array([
            # First batch beams - top 3 of [0.3+0.8, 0.3+0.1, ...]
            1.1, 0.4, 0.3+0.05,
            # Second batch beams - top 3 of [0.7+0.2, 0.7+0.3, 0.7+0.3, ...]
            1.0, 1.0, 0.7+0.1
        ])
        self.assertTrue(mx.allclose(scores_2, expected_scores, atol=1e-3))
        
        # Verify ancestors point to correct original beams
        self.assertEqual(ancestors_2.tolist(), [0,0,0,3,3,3])


if __name__ == "__main__":
    unittest.main()
