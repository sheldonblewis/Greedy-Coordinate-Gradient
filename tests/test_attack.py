import pytest
import torch

from gcg.attack import DEVICE, init_suffix
from gcg.setup import load, problem


@pytest.fixture(scope="module")
def gpt2():
    return load("gpt2")


def greedy_continuation(model, ids, n):
    with torch.no_grad():
        out = model.generate(ids[None], max_new_tokens=n, do_sample=False, pad_token_id=0)
    return out[0, ids.shape[0] :]


def test_exact_flag_matches_greedy_decoding(gpt2):
    model, tok, allowed = gpt2
    p = problem("gpt2", model, tok, allowed, "Paris is in France.")
    suffix = init_suffix(tok, 20)
    prompt = torch.cat([p.before, suffix, p.after])
    p.target = greedy_continuation(model, prompt, 6)
    loss, exact = p.evaluate(suffix[None])
    assert exact.item()
    p.target = p.target.clone()
    p.target[0] = (p.target[0] + 1) % model.config.vocab_size
    _, exact = p.evaluate(suffix[None])
    assert not exact.item()


def test_gradient_ranks_helpful_swaps_above_random(gpt2):
    model, tok, allowed = gpt2
    p = problem("gpt2", model, tok, allowed, "The capital of France is Berlin.")
    suffix = init_suffix(tok, 20)
    base, _ = p.evaluate(suffix[None])
    grad = p.token_gradients(suffix)
    grad[:, ~allowed.to(DEVICE)] = float("inf")
    g = torch.Generator().manual_seed(0)
    pos = torch.randint(0, 20, (256,), generator=g).to(DEVICE)

    guided = suffix.repeat(256, 1)
    top = (-grad).topk(64, dim=1).indices
    guided[torch.arange(256), pos] = top[pos, torch.randint(0, 64, (256,), generator=g).to(DEVICE)]
    ids = allowed.nonzero().squeeze(1).to(DEVICE)
    random = suffix.repeat(256, 1)
    random[torch.arange(256), pos] = ids[torch.randint(0, len(ids), (256,), generator=g).to(DEVICE)]

    guided_loss, _ = p.evaluate(guided)
    random_loss, _ = p.evaluate(random)
    assert (guided_loss - base).mean() < (random_loss - base).mean()
    assert (guided_loss < base).float().mean() > (random_loss < base).float().mean()


def test_retokenization_filter(gpt2):
    model, tok, allowed = gpt2
    p = problem("gpt2", model, tok, allowed, "Cats can fly.")
    assert p.retokenizes(init_suffix(tok, 20)[None]).all()
    pieces = torch.tensor([tok.encode("he")[0], tok.encode("llo")[0]], device=DEVICE)
    assert tok.encode(tok.decode(pieces)) != pieces.tolist()
    assert not p.retokenizes(pieces[None]).item()


def test_allowed_tokens_are_printable_ascii(gpt2):
    _, tok, allowed = gpt2
    for i in allowed.nonzero().squeeze(1)[:: max(1, int(allowed.sum()) // 500)].tolist():
        s = tok.decode([i])
        assert s.isascii() and s.isprintable() and s.strip()
