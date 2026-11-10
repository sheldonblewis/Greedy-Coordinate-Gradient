"""Greedy Coordinate Gradient (Zou et al., 2023, arXiv:2307.15043) and the baselines it is
compared against.

The task is target forcing: find a suffix of L tokens such that, given [prefix][suffix][postfix],
the model's greedy continuation is exactly the target string. GCG uses the gradient of the target
loss with respect to a one-hot encoding of the suffix to propose top-k token swaps per position,
evaluates a batch of single-swap candidates exactly, and greedily keeps the best one.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def allowed_tokens(tok, vocab_size):
    """Printable-ascii tokens that decode cleanly; everything else is excluded from the search."""
    ok = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(tok.all_special_ids)
    for i in range(min(vocab_size, len(tok))):
        if i in special:
            continue
        s = tok.decode([i])
        if s and s.isascii() and s.isprintable() and s.strip():
            ok[i] = True
    return ok


@dataclass
class Problem:
    """Token layout: [before][suffix][after][target]. Loss is teacher-forced CE on target."""

    model: object
    tok: object
    before: torch.Tensor
    after: torch.Tensor
    target: torch.Tensor
    allowed: torch.Tensor
    chunk: int = 128
    embed: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.embed = self.model.get_input_embeddings().weight

    def _hidden(self, ids=None, embeds=None):
        out = self.model.base_model(input_ids=ids, inputs_embeds=embeds)
        return out.last_hidden_state

    def _target_logits(self, hidden, L):
        start = len(self.before) + L + len(self.after)
        h = hidden[:, start - 1 : start - 1 + len(self.target)]
        return self.model.lm_head(h).float()

    @torch.no_grad()
    def evaluate(self, suffixes):
        """Per-candidate target loss and whether greedy decoding would produce the target exactly."""
        B, L = suffixes.shape
        losses, exact = [], []
        for i in range(0, B, self.chunk):
            s = suffixes[i : i + self.chunk]
            n = len(s)
            ids = torch.cat([self.before.expand(n, -1), s, self.after.expand(n, -1), self.target.expand(n, -1)], 1)
            logits = self._target_logits(self._hidden(ids=ids), L)
            ce = F.cross_entropy(logits.transpose(1, 2), self.target.expand(n, -1), reduction="none").mean(1)
            losses.append(ce)
            exact.append((logits.argmax(-1) == self.target).all(1))
        return torch.cat(losses), torch.cat(exact)

    def token_gradients(self, suffix):
        L = len(suffix)
        one_hot = torch.zeros(L, self.embed.shape[0], device=DEVICE, dtype=self.embed.dtype)
        one_hot.scatter_(1, suffix[:, None], 1.0)
        one_hot.requires_grad_(True)
        e = self.embed
        full = torch.cat([e[self.before], one_hot @ e, e[self.after], e[self.target]])[None]
        logits = self._target_logits(self._hidden(embeds=full), L)
        loss = F.cross_entropy(logits[0], self.target)
        loss.backward()
        return one_hot.grad.detach()

    def retokenizes(self, suffixes):
        """Keep candidates whose text re-encodes to the same tokens, so the result is a real string."""
        texts = self.tok.batch_decode(suffixes)
        enc = self.tok(texts, add_special_tokens=False)["input_ids"]
        return torch.tensor([e == s for e, s in zip(enc, suffixes.tolist())], device=suffixes.device)


def init_suffix(tok, L):
    ids = tok(" !" * L, add_special_tokens=False)["input_ids"]
    return torch.tensor(ids[:L], device=DEVICE)


def run(problem, method, L=20, steps=150, batch=128, topk=128, seed=0):
    """Returns the loss curve, the step of first exact success (or None) and the final suffix."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    suffix = init_suffix(problem.tok, L)
    allowed_ids = problem.allowed.nonzero().squeeze(1).to(DEVICE)
    best_loss, curve, first = float("inf"), [], None
    for step in range(1, steps + 1):
        pos = torch.randint(0, L, (batch,), generator=g).to(DEVICE)
        if method == "gcg":
            grad = problem.token_gradients(suffix)
            grad[:, ~problem.allowed.to(DEVICE)] = float("inf")
            top = (-grad).topk(topk, dim=1).indices  # most loss-reducing swaps per position
            pick = torch.randint(0, topk, (batch,), generator=g).to(DEVICE)
            new_tok = top[pos, pick]
        elif method == "random":
            new_tok = allowed_ids[torch.randint(0, len(allowed_ids), (batch,), generator=g).to(DEVICE)]
        else:
            raise ValueError(method)
        cands = suffix.repeat(batch, 1)
        cands[torch.arange(batch, device=DEVICE), pos] = new_tok
        cands = cands[problem.retokenizes(cands)]
        if len(cands) == 0:
            curve.append(best_loss)
            continue
        losses, exact = problem.evaluate(cands)
        i = losses.argmin()
        suffix = cands[i]  # gcg always moves to the batch's best candidate
        best_loss = min(best_loss, losses[i].item())
        curve.append(best_loss)
        if exact.any():
            suffix = cands[exact.nonzero()[0, 0]]
            first = step
            break
    return {"curve": curve, "success_step": first, "suffix_ids": suffix.tolist(), "suffix": problem.tok.decode(suffix)}


def legacy(problem, iterations=250, top_k=16, positions_per_iter=2, temperature=0.05, seed=0):
    """A faithful port of the original main.py, for comparison.

    Its objective scores every target token at the *last prompt position*
    (log_probs[:, -1, target_ids].mean()), i.e. it asks the next token to be all target tokens at
    once, with a 0.05 softmax temperature; it optimises the whole prompt rather than a suffix.
    Here it is adapted to the same suffix layout and evaluated with the same exact-match criterion.
    """
    torch.manual_seed(seed)
    L = 20
    suffix = init_suffix(problem.tok, L)
    best_obj, best = float("inf"), suffix.clone()
    curve, first = [], None
    e = problem.embed
    for step in range(1, iterations + 1):
        positions = torch.randperm(L)[:positions_per_iter].to(DEVICE)
        emb = e[suffix].detach().clone().requires_grad_(True)
        full = torch.cat([e[problem.before], emb, e[problem.after]])[None]
        h = problem._hidden(embeds=full)
        lp = F.log_softmax(problem.model.lm_head(h[:, -1]).float() / temperature, -1)
        (-lp[0, problem.target].mean()).backward()
        sims = emb.grad[positions] @ e.T
        sims[:, ~problem.allowed.to(DEVICE)] = float("inf")
        idx = sims.topk(top_k, dim=-1, largest=False).indices
        cands = []
        for j, p in enumerate(positions):
            c = suffix.repeat(top_k, 1)
            c[:, p] = idx[j]
            cands.append(c)
        cands = torch.cat(cands)
        with torch.no_grad():
            ids = torch.cat([problem.before.expand(len(cands), -1), cands, problem.after.expand(len(cands), -1)], 1)
            lp = F.log_softmax(problem.model.lm_head(problem._hidden(ids=ids)[:, -1]).float() / temperature, -1)
            obj = -lp[:, problem.target].mean(-1)
        j = obj.argmin()
        if obj[j] < best_obj:
            best_obj, best, suffix = obj[j].item(), cands[j].clone(), cands[j].clone()
        loss, exact = problem.evaluate(best[None])
        curve.append(loss.item())
        if exact.item():
            first = step
            break
    return {"curve": curve, "success_step": first, "suffix_ids": best.tolist(), "suffix": problem.tok.decode(best)}


def run_universal(problems, L=20, steps=150, batch=128, topk=128, seed=0):
    """One suffix for several targets at once (the multi-prompt variant in Zou et al.).

    Gradients are summed over targets and candidates are scored by their mean loss. Tracks how
    many targets the current suffix forces exactly at each step.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = problems[0]
    suffix = init_suffix(first.tok, L)
    best_loss, curve, forced = float("inf"), [], []
    for _ in range(steps):
        grad = sum(p.token_gradients(suffix) for p in problems)
        grad[:, ~first.allowed.to(DEVICE)] = float("inf")
        top = (-grad).topk(topk, dim=1).indices
        pos = torch.randint(0, L, (batch,), generator=g).to(DEVICE)
        pick = torch.randint(0, topk, (batch,), generator=g).to(DEVICE)
        cands = suffix.repeat(batch, 1)
        cands[torch.arange(batch, device=DEVICE), pos] = top[pos, pick]
        cands = cands[first.retokenizes(cands)]
        if len(cands) == 0:
            curve.append(best_loss)
            forced.append(forced[-1] if forced else 0)
            continue
        results = [p.evaluate(cands) for p in problems]
        losses = torch.stack([r[0] for r in results]).mean(0)
        hits = torch.stack([r[1] for r in results]).sum(0)
        i = losses.argmin()
        suffix = cands[i]
        best_loss = min(best_loss, losses[i].item())
        curve.append(best_loss)
        forced.append(int(hits[i].item()))
        if forced[-1] == len(problems):
            break
    return {"curve": curve, "forced": forced, "suffix_ids": suffix.tolist(), "suffix": first.tok.decode(suffix)}
