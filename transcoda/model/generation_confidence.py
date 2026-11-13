"""Length-normalized confidence metrics for generated sequences."""

from __future__ import annotations

import torch


def compute_greedy_transition_scores(
    *,
    sequences: torch.Tensor,
    scores: tuple[torch.Tensor, ...],
    batch_index: int = 0,
) -> torch.Tensor:
    """Fallback selected-token logprobs for non-beam generation."""
    selected_logprobs: list[torch.Tensor] = []
    sequence = sequences[batch_index]
    for step_index, step_scores in enumerate(scores):
        token_index = step_index + 1
        if token_index >= sequence.numel():
            break
        token_id = int(sequence[token_index].item())
        step_logprobs = torch.log_softmax(
            step_scores[batch_index].detach().float().cpu(), dim=-1
        )
        selected_logprobs.append(step_logprobs[token_id])
    if not selected_logprobs:
        return torch.empty(0, dtype=torch.float32)
    return torch.stack(selected_logprobs)


def _summarize_selected_logprobs(
    selected_logprobs: torch.Tensor,
    *,
    token_ids: list[int],
    i2w: dict[int, str],
    pad_token_id: int | None,
    eos_token_id: int | None,
    finalized,
    max_length: int,
    use_grammar: bool,
    use_spine_structure_constraints: bool = False,
) -> dict[str, object]:
    scored_token_ids = token_ids[1 : 1 + int(selected_logprobs.numel())]
    if len(scored_token_ids) < int(selected_logprobs.numel()):
        selected_logprobs = selected_logprobs[: len(scored_token_ids)]
    keep_values = [
        token_id != pad_token_id and i2w.get(token_id) != "<bos>"
        for token_id in scored_token_ids
    ]
    keep_mask = torch.tensor(keep_values, dtype=torch.bool)
    if keep_mask.numel() == selected_logprobs.numel():
        selected_logprobs = selected_logprobs[keep_mask]
        scored_token_ids = [
            token_id
            for token_id, keep in zip(scored_token_ids, keep_values, strict=True)
            if keep
        ]

    tokens = [
        "" if eos_token_id is not None and token_id == eos_token_id else i2w.get(token_id, "")
        for token_id in scored_token_ids
    ]

    if selected_logprobs.numel() == 0:
        mean_logprob = None
        mean_prob = None
        min_prob = None
        p05_prob = None
        p10_prob = None
    else:
        probs = selected_logprobs.exp()
        mean_logprob = float(selected_logprobs.mean().item())
        mean_prob = float(probs.mean().item())
        min_prob = float(probs.min().item())
        p05_prob = float(torch.quantile(probs, 0.05).item())
        p10_prob = float(torch.quantile(probs, 0.10).item())

    return {
        "num_scored_tokens": int(selected_logprobs.numel()),
        "mean_logprob": mean_logprob,
        "mean_prob": mean_prob,
        "min_prob": min_prob,
        "p05_prob": p05_prob,
        "p10_prob": p10_prob,
        "token_logprobs": [round(float(value), 4) for value in selected_logprobs.tolist()],
        "tokens": tokens,
        "saw_eos": bool(
            getattr(finalized, "saw_eos", False)
            or (eos_token_id is not None and eos_token_id in token_ids)
        ),
        "hit_max_length": bool(
            getattr(finalized, "hit_max_length", False)
            or len(token_ids) >= int(max_length)
        ),
        "confidence_kind": (
            "grammar_constrained_generation_scores"
            if use_grammar
            else (
                "spine_constrained_generation_scores"
                if use_spine_structure_constraints
                else "post_processor_generation_scores"
            )
        ),
    }


def _batch_transition_scores(model, generation_output) -> torch.Tensor | None:
    """Selected-token logprobs for the whole batch, or None to use the fallback."""
    scores = tuple(getattr(generation_output, "scores", ()) or ())
    if not scores:
        return None
    beam_indices = getattr(generation_output, "beam_indices", None)
    try:
        transition_scores = model.compute_transition_scores(
            generation_output.sequences,
            scores,
            beam_indices=beam_indices,
            normalize_logits=True,
        )
    except Exception:
        return None
    return transition_scores.detach().float().cpu()


def _selected_logprobs_for_row(
    model,
    generation_output,
    transition_scores: torch.Tensor | None,
    batch_index: int,
) -> torch.Tensor:
    if transition_scores is not None:
        return transition_scores[batch_index]
    scores = tuple(getattr(generation_output, "scores", ()) or ())
    if not scores:
        return torch.empty(0, dtype=torch.float32)
    return compute_greedy_transition_scores(
        sequences=generation_output.sequences,
        scores=scores,
        batch_index=batch_index,
    )


def compute_generation_confidence(
    *,
    model,
    generation_output,
    token_ids: list[int],
    i2w: dict[int, str],
    pad_token_id: int | None,
    eos_token_id: int | None,
    finalized,
    max_length: int,
    use_grammar: bool,
    use_spine_structure_constraints: bool = False,
) -> dict[str, object]:
    """Compute confidence metrics for a single-sample generation.

    Besides summary statistics, the result carries the full per-token logprob
    array (``token_logprobs``) so future acquisition functions can be computed
    without re-running generation.
    """
    transition_scores = _batch_transition_scores(model, generation_output)
    selected_logprobs = _selected_logprobs_for_row(
        model, generation_output, transition_scores, batch_index=0
    )
    return _summarize_selected_logprobs(
        selected_logprobs,
        token_ids=token_ids,
        i2w=i2w,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        finalized=finalized,
        max_length=max_length,
        use_grammar=use_grammar,
        use_spine_structure_constraints=use_spine_structure_constraints,
    )


def compute_batch_generation_confidences(
    *,
    model,
    generation_output,
    batch_token_ids: list[list[int]],
    i2w: dict[int, str],
    pad_token_id: int | None,
    eos_token_id: int | None,
    finalized_list: list,
    max_length: int,
    use_grammar: bool,
    use_spine_structure_constraints: bool = False,
) -> list[dict[str, object]]:
    """Per-item confidence metrics for a batched generation.

    Transition scores are computed once for the whole batch; rows that
    finished early are padded by ``generate`` and the pad steps are dropped
    by the per-row token mask.
    """
    if len(batch_token_ids) != len(finalized_list):
        raise ValueError("batch_token_ids and finalized_list must have equal length")
    transition_scores = _batch_transition_scores(model, generation_output)
    confidences: list[dict[str, object]] = []
    for batch_index, (token_ids, finalized) in enumerate(
        zip(batch_token_ids, finalized_list, strict=True)
    ):
        selected_logprobs = _selected_logprobs_for_row(
            model, generation_output, transition_scores, batch_index
        )
        confidences.append(
            _summarize_selected_logprobs(
                selected_logprobs,
                token_ids=token_ids,
                i2w=i2w,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                finalized=finalized,
                max_length=max_length,
                use_grammar=use_grammar,
                use_spine_structure_constraints=use_spine_structure_constraints,
            )
        )
    return confidences
