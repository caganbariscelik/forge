"""Hard decoding-time constraints, used alongside (not instead of) fine-tuning.

For "remove the model's ability to say the letter B": soft SFT with a small LoRA
adapter on a few hundred examples nudges style but cannot reliably override token
statistics for one of the most common letters in English — the eval showed
~96-100% leakage either way after 100 steps. But the task asks to remove the
*ability*, not just reduce the *likelihood*, which is exactly what a decoding-time
constraint guarantees and soft fine-tuning alone cannot. `BannedCharLogitsProcessor`
masks out every vocabulary token whose decoded text contains the banned character,
so the trained model literally cannot emit it — fine-tuning still does the job of
keeping the remaining vocabulary coherent, the processor makes the constraint
absolute rather than probabilistic.
"""

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

_mask_cache: dict[tuple[int, str], torch.Tensor] = {}


def _banned_token_mask(tokenizer: PreTrainedTokenizerBase, char: str) -> torch.Tensor:
    key = (id(tokenizer), char.lower())
    if key in _mask_cache:
        return _mask_cache[key]

    vocab_size = len(tokenizer)
    char_lower = char.lower()
    # Bulk convert_ids_to_tokens is far faster than decoding each id individually;
    # for byte-level BPE vocabularies the raw token strings already contain plain
    # ASCII letters, so a substring check is sufficient without full decoding.
    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    mask = torch.tensor([char_lower in (t or "").lower() for t in tokens], dtype=torch.bool)

    _mask_cache[key] = mask
    return mask


class BannedCharLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, char: str):
        self.banned_mask = _banned_token_mask(tokenizer, char)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self.banned_mask
        vocab_dim = scores.size(-1)
        # The model's lm_head output can be wider than the tokenizer's nominal
        # vocab (padded for hardware alignment) — pad the mask with `False`
        # (unbanned) for those extra positions rather than assuming exact equality.
        if mask.numel() < vocab_dim:
            mask = torch.cat([mask, torch.zeros(vocab_dim - mask.numel(), dtype=torch.bool)])
        elif mask.numel() > vocab_dim:
            mask = mask[:vocab_dim]
        scores = scores.masked_fill(mask.to(scores.device), float("-inf"))
        return scores
