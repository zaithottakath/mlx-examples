import unittest

import mlx.core as mx
from mlx_lm.sample_utils import min_p_sampling, top_k_sampling, top_p_sampling


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


    def test_beam_search_sampler(self):
        from mlx_lm.sample_utils import BeamSearchSampler
        beams = 2
        temperature = 1.0
        # Create a deterministic case: batch size = 1, beams = 2, vocab_size = 5.
        # next_token_logits shape: (2, 5)
        # Row 0 has a clear maximum at index 0; row 1 is uniform.
        next_token_logits = mx.array([
            [10, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        # Use zero sequence weights.
        sequence_weights = mx.zeros((2,), dtype=mx.float32)
        
        sampler = BeamSearchSampler(beams=beams, temperature=temperature)
        next_token_ids, beam_indices, beam_scores = sampler(next_token_logits, sequence_weights, None)
        
        # Expected output:
        # When re-shaping, the two rows become concatenated to form a (1,10) array:
        #   row0: logsoftmax => highest value at index 0, others much lower.
        #   row1: uniform => each value equal.
        # Top 2 indices should be 0 (from row0) and 5 (from row1),
        # yielding beam_indices: [0, 1] and next_token_ids: [[0], [0]].
        expected_next_token_ids = mx.array([[0], [0]])
        expected_beam_indices = mx.array([0, 1])
        
        print("next_token_ids:", next_token_ids)
        print("expected_next_token_ids:", expected_next_token_ids)
        print("beam_indices:", beam_indices)
        print("expected_beam_indices:", expected_beam_indices)
        print("next_token_ids shape:", next_token_ids.shape)
        print("beam_indices shape:", beam_indices.shape)
        
        self.assertTrue(mx.array_equal(next_token_ids, expected_next_token_ids))
        self.assertTrue(mx.array_equal(beam_indices, expected_beam_indices))
        self.assertEqual(next_token_ids.shape, (2, 1))
        self.assertEqual(beam_indices.shape, (2,))
        
if __name__ == "__main__":
    unittest.main()
