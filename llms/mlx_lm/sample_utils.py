# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial
from typing import Callable, Dict, Optional

import mlx.core as mx
import os
import logging


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = -1,
    beam: int = 1,
) -> Callable:
    """
    Make a sampler function for use with ``generate_step``.

    Args:
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        top_k (int, optional): The top k tokens ranked by probability to constrain
          the sampling to.
        beam (int, optional): Number of beams for beam search. If beam > 1, a beam search sampler is returned.

    Returns:
        Callable:
            A sampler function which takes log-probabilities and returns tokens.
            If beam > 1, a BeamSearchSampler is returned.
    """
    if beam > 1:
        return BeamSearchSampler(beams=beam, temperature=temp)
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)
    elif top_p > 0 and top_p < 1.0:
        return lambda x: top_p_sampling(x, top_p, temp)
    elif min_p != 0.0:
        return lambda x: min_p_sampling(x, min_p, min_tokens_to_keep, temp)
    elif top_k > 0:
        return lambda x: top_k_sampling(x, top_k, temp)
    else:
        return lambda x: categorical_sampling(x, temp)


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
):
    """
    Make logits processors for use with ``generate_step``.

    Args:
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        logit_bias (dictionary, optional): Additive logit bias.

    Returns:
        List[Callable[[mx.array, mx.array], mx.array]]:
            A list of logits processors. Each processor in the list is a
            callable which takes an array of tokens and an array of logits
            and returns the updated logits.
    """
    logits_processors = []
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            logits[:, indices] += values
            return logits

        logits_processors.append(logit_bias_processor)

    if repetition_penalty and repetition_penalty != 0.0:
        logits_processors.append(
            make_repetition_penalty(repetition_penalty, repetition_context_size)
        )
    return logits_processors


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logprobs: mx.array,
    top_k: int,
    temperature=1.0,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.

    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    logprobs = logprobs * (1 / temperature)
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return mx.random.categorical(masked_logprobs, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def min_p_sampling(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
    temperature=1.0,
) -> mx.array:
    """
    Apply min-p sampling to the logprobs.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logprobs: A vector of log probabilities.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.

    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )
    # reference implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L531-L605

    logprobs = logprobs * (1 / temperature)

    # Indices sorted in decreasing order
    sorted_indices = mx.argsort(-logprobs, axis=-1)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

    # Top probability
    top_logprobs = sorted_logprobs[:, 0:1]

    # Calculate the min_p threshold
    scaled_min_p = top_logprobs + math.log(min_p)

    # Mask tokens that have a probability less than the scaled min_p
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    # Create pool of tokens with probability less than scaled min_p
    selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)

    # Return sampled tokens
    sorted_tokens = mx.random.categorical(selected_logprobs, axis=-1)[:, None]
    return mx.take_along_axis(sorted_indices, sorted_tokens, axis=-1).squeeze(1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        0,
    )

    sorted_tokens = mx.random.categorical(mx.log(top_probs), axis=-1)[:, None]
    return mx.take_along_axis(sorted_indices, sorted_tokens, axis=-1).squeeze(1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))


def make_repetition_penalty(penalty: float, context_size: int = 20):
    """
    Make repetition penalty processor.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        penalty (float): The repetition penalty factor to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]:
            The repetition penalty processor.
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor

class BeamSearchSampler:
    """
    Beam search sampler for sequence generation.
    This sampler selects the top-B beam tokens for each item.
    It returns a tuple of (next_token_ids, beam_indices, beam_scores)
    where next_token_ids is of shape (batch * beams, 1).
    """
    def __init__(self, beams: int = 3, temperature: float = 1.0):
        self.beams = beams
        if temperature <= 0.0:
            self.temperature = 1.0
        else:
            self.temperature = temperature

    def __call__(self, next_token_logits: mx.array, sequence_weights: mx.array, _):
        # next_token_logits: shape (batch * beams, vocab_size)
        if next_token_logits.shape[0] != sequence_weights.shape[0]:
            next_token_logits = next_token_logits[:sequence_weights.shape[0]]
        # Compute numerically stable log probabilities without using mx.log_softmax.
        scaled_logits = next_token_logits / self.temperature
        m = mx.max(scaled_logits, axis=-1, keepdims=True)
        logprobs = scaled_logits - m - mx.log(mx.sum(mx.exp(scaled_logits - m), axis=-1, keepdims=True))
        if mx.sum(mx.isnan(logprobs)).item() > 0:
            print("BeamSearchSampler: Detected NaN values in logprobs. scaled_logits: {}, m: {}".format(scaled_logits, m))
        # Add previous cumulative sequence weights.
        combined_scores = mx.reshape(sequence_weights, (-1, 1)) + logprobs  # shape: (batch*beams, vocab_size)
        batch = sequence_weights.shape[0] // self.beams
        if sequence_weights.shape[0] != batch * self.beams:
            print("DEBUG: Adjusting sequence_weights shape from", sequence_weights.shape, "to", (batch * self.beams,))
            sequence_weights = sequence_weights[:batch * self.beams]
            logprobs = logprobs[:batch * self.beams]
        vocab_size = next_token_logits.shape[-1]
        # Reshape combined scores to (batch, beams, vocab_size)
        combined_scores = mx.reshape(combined_scores, (batch, self.beams, vocab_size))
        # Flatten scores to (batch, beams*vocab_size)
        flat_scores = mx.reshape(combined_scores, (batch, self.beams * vocab_size))
        # Add tie-breaking bias.
        total_dim = self.beams * vocab_size
        bias = -mx.arange(total_dim, dtype=flat_scores.dtype)
        bias = mx.reshape(bias, (1, total_dim))
        flat_scores = flat_scores + bias * 1e-6
        # Perform global candidate selection with diversity constraint using vectorized operations:
        total_dim = self.beams * vocab_size
        sorted_idx = mx.topk(flat_scores, k=total_dim, axis=1).astype(mx.int32)  # shape (batch, total_dim)
        sorted_scores = mx.take_along_axis(flat_scores, sorted_idx, axis=1)  # shape (batch, total_dim)
        sorted_beam_ids = sorted_idx // vocab_size  # shape (batch, total_dim)
        sorted_token_ids = sorted_idx % vocab_size   # shape (batch, total_dim)
        positions = mx.arange(total_dim, dtype=mx.int32).reshape((1, total_dim))
        positions = mx.broadcast_to(positions, (batch, total_dim))  # shape (batch, total_dim)
        beam_range = mx.arange(self.beams, dtype=mx.int32).reshape((1, 1, self.beams))  # shape (1,1,self.beams)
        sorted_beam_ids_exp = mx.expand_dims(sorted_beam_ids, axis=-1)  # shape (batch, total_dim, 1)
        mask = mx.equal(sorted_beam_ids_exp, beam_range)  # shape (batch, total_dim, self.beams)
        positions_extended = mx.expand_dims(positions, axis=-1)  # shape (batch, total_dim, 1)
        positions_masked = mx.where(mask, positions_extended, mx.full(positions_extended.shape, total_dim, dtype=positions_extended.dtype))
        first_occurrence = mx.min(positions_masked, axis=1)  # shape (batch, self.beams)
        batch_indices = mx.arange(batch, dtype=mx.int32).reshape((batch, 1))
        batch_indices = mx.broadcast_to(batch_indices, (batch, self.beams))  # shape (batch, self.beams)
        gather_indices = mx.stack([batch_indices, first_occurrence], axis=-1)  # shape (batch, self.beams, 2)
        selected_token_ids = mx.gather_nd(sorted_token_ids, gather_indices)  # shape (batch, self.beams)
        selected_beam_ids = mx.gather_nd(sorted_beam_ids, gather_indices)  # shape (batch, self.beams)
        selected_scores = mx.gather_nd(sorted_scores, gather_indices)  # shape (batch, self.beams)
        next_token_ids = mx.reshape(selected_token_ids, (batch * self.beams, 1))
        beam_indices = mx.reshape(selected_beam_ids, (batch * self.beams,))
        beam_scores = mx.reshape(selected_scores, (batch * self.beams,))
        return next_token_ids, beam_indices, beam_scores
