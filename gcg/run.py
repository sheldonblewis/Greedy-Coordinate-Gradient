"""Every experiment in the README. Each attack run is cached in results/runs/, so a stopped
script picks up where it left off.

    python -m gcg.run all          # everything below, ~1 hour on an M-series Mac
    python -m gcg.run main         # gcg vs random search vs the original implementation
    python -m gcg.run ablate       # top-k and suffix length
    python -m gcg.run universal    # one suffix for several targets
    python -m gcg.run transfer     # do gpt-2 suffixes carry over to qwen2.5-0.5b-instruct?
    python -m gcg.run defense      # does a perplexity filter catch the suffixes?
    python -m gcg.run report       # tables and figures
"""

import json
import sys
import time
from pathlib import Path

import torch

from gcg.attack import legacy, run, run_universal
from gcg.setup import TARGETS, load, problem

R = Path("results")
RUNS = R / "runs"
STEPS = 150
N_MAIN = 12
N_ABLATE = 6
UNIVERSAL_SETS = [TARGETS[0:3], TARGETS[3:6]]


def cached(name, fn, *args, **kwargs):
    path = RUNS / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    out = fn(*args, **kwargs)
    out["seconds"] = time.time() - t0
    path.write_text(json.dumps(out))
    done = out.get("success_step", out.get("forced", [0])[-1])
    print(
        f"{name}: {done} after {len(out['curve'])} steps, loss {out['curve'][-1]:.3f} ({out['seconds']:.0f}s)",
        flush=True,
    )
    return out


def main_exp():
    m, t, a = load("gpt2")
    for i, target in enumerate(TARGETS[:N_MAIN]):
        p = problem("gpt2", m, t, a, target)
        cached(f"gpt2-gcg-k128-L20-t{i}", run, p, "gcg", steps=STEPS)
        cached(f"gpt2-random-L20-t{i}", run, p, "random", steps=STEPS)
        cached(f"gpt2-legacy-t{i}", legacy, p, iterations=STEPS)


def ablate():
    m, t, a = load("gpt2")
    for i, target in enumerate(TARGETS[:N_ABLATE]):
        p = problem("gpt2", m, t, a, target)
        for k in (8, 32, 512):
            cached(f"gpt2-gcg-k{k}-L20-t{i}", run, p, "gcg", steps=STEPS, topk=k)
        for length in (5, 10):
            cached(f"gpt2-gcg-k128-L{length}-t{i}", run, p, "gcg", steps=STEPS, L=length)


def universal():
    m, t, a = load("gpt2")
    for j, targets in enumerate(UNIVERSAL_SETS):
        problems = [problem("gpt2", m, t, a, target) for target in targets]
        cached(f"gpt2-universal-set{j}", run_universal, problems, steps=STEPS)


def transfer():
    m, t, a = load("qwen")
    rows = []
    for i, target in enumerate(TARGETS[:N_MAIN]):
        src = json.loads((RUNS / f"gpt2-gcg-k128-L20-t{i}.json").read_text())
        p = problem("qwen", m, t, a, target)
        ids = torch.tensor(t(src["suffix"], add_special_tokens=False)["input_ids"], device=p.before.device)
        bang = torch.tensor(t(" !" * 20, add_special_tokens=False)["input_ids"][:20], device=p.before.device)
        loss, exact = p.evaluate(ids[None])
        base, _ = p.evaluate(bang[None])
        rows.append(
            {
                "target": target,
                "gpt2_success": src["success_step"] is not None,
                "qwen_loss_with_gpt2_suffix": loss.item(),
                "qwen_loss_with_init_suffix": base.item(),
                "qwen_exact": bool(exact.item()),
            }
        )
    (R / "transfer.json").write_text(json.dumps(rows, indent=2))


def defense():
    from gcg.defense import natural_sentences, scores

    m, t, _ = load("gpt2")
    prefix = "Here is a fact:"
    adversarial = [prefix + json.loads(p.read_text())["suffix"] for p in sorted(RUNS.glob("gpt2-gcg-*.json"))]
    natural = [f"{prefix} {s}" for s in natural_sentences()]
    out = {"adversarial": scores(m, t, adversarial), "natural": scores(m, t, natural)}
    (R / "defense.json").write_text(json.dumps(out))


STAGES = {"main": main_exp, "ablate": ablate, "universal": universal, "transfer": transfer, "defense": defense}

if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "report":
        from gcg.report import report

        report()
    elif cmd == "all":
        for stage in STAGES.values():
            stage()
        from gcg.report import report

        report()
    else:
        STAGES[cmd]()
