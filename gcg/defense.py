"""Perplexity filtering (Jain et al., 2023, arXiv:2309.00614) against the suffixes GCG finds.

Adversarial suffixes are token soup, so their perplexity under a language model should sit far
above natural text. We score whole prompts (prefix + suffix) and their worst 10-token window, and
measure how cleanly a threshold separates GCG prompts from prompts ending in natural sentences.
"""

import re

import numpy as np
import torch
import torch.nn.functional as F

from gcg.attack import DEVICE

WINDOW = 10


@torch.no_grad()
def token_nll(model, tok, text):
    ids = torch.tensor([tok.encode(text)], device=DEVICE)
    logits = model(ids).logits[0, :-1].float()
    return F.cross_entropy(logits, ids[0, 1:], reduction="none").cpu().numpy()


def scores(model, tok, texts):
    out = []
    for text in texts:
        nll = token_nll(model, tok, text)
        windows = [nll[i : i + WINDOW].mean() for i in range(max(1, len(nll) - WINDOW + 1))]
        out.append({"log_ppl": float(nll.mean()), "max_window_log_ppl": float(max(windows))})
    return out


def detokenize(text):
    """WikiText stores text tokenized ("word , word 's"); undo that so it reads naturally."""
    text = re.sub(r" ([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[$]) ", r"\1", text)
    text = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    return re.sub(r'" (.*?) "', r'"\1"', text)


def natural_sentences(n=300, seed=0, min_words=8, max_words=24):
    from datasets import load_dataset

    lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    sents = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("="):
            continue
        for s in re.split(r"(?<=[.!?])\s+", detokenize(line)):
            if min_words <= len(s.split()) <= max_words and "@" not in s:
                sents.append(s)
    rng = np.random.default_rng(seed)
    return [sents[i] for i in rng.choice(len(sents), size=min(n, len(sents)), replace=False)]
