import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from gcg.run import N_ABLATE, N_MAIN, RUNS, STEPS, UNIVERSAL_SETS  # noqa: E402
from gcg.setup import TARGETS  # noqa: E402

R = Path("results")
F = Path("figures")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)
INK, GOLD, RUST, SAGE = "#1f3a5f", "#c9a46c", "#c0715f", "#8fb39a"


def load(prefix, n):
    return [json.loads(p.read_text()) for i in range(n) if (p := RUNS / f"{prefix}-t{i}.json").exists()]


def padded(runs):
    return np.array([r["curve"] + [r["curve"][-1]] * (STEPS - len(r["curve"])) for r in runs])


def success_curve(runs):
    done = np.array([r["success_step"] or np.inf for r in runs])
    steps = np.arange(1, STEPS + 1)
    return steps, np.array([(done <= s).mean() for s in steps])


def summary(runs):
    ok = [r["success_step"] for r in runs if r["success_step"]]
    return {
        "n": len(runs),
        "success_rate": len(ok) / len(runs) if runs else None,
        "median_steps_to_success": float(np.median(ok)) if ok else None,
        "median_final_loss": float(np.median([r["curve"][-1] for r in runs])) if runs else None,
        "mean_seconds": float(np.mean([r["seconds"] for r in runs])) if runs else None,
    }


def report():
    F.mkdir(exist_ok=True)
    main = {
        "gcg": load("gpt2-gcg-k128-L20", N_MAIN),
        "random search, same budget": load("gpt2-random-L20", N_MAIN),
        "original main.py": load("gpt2-legacy", N_MAIN),
    }
    ablation = {f"top-k {k}": load(f"gpt2-gcg-k{k}-L20", N_ABLATE) for k in (8, 32, 128, 512)}
    ablation |= {f"{n} suffix tokens": load(f"gpt2-gcg-k128-L{n}", N_ABLATE) for n in (5, 10, 20)}
    universal = [
        json.loads(p.read_text())
        for j in range(len(UNIVERSAL_SETS))
        if (p := RUNS / f"gpt2-universal-set{j}.json").exists()
    ]
    out = {"main": {k: summary(v) for k, v in main.items()}, "ablation": {k: summary(v) for k, v in ablation.items()}}
    lines = [
        f"| method (gpt-2, {N_MAIN} targets, {STEPS} steps x 128 candidates) | forced exactly | median steps "
        "| median final loss | s / target |",
        "|---|---|---|---|---|",
    ]
    for k, v in out["main"].items():
        lines.append(
            f"| {k} | {v['success_rate']:.0%} | {v['median_steps_to_success'] or '-'} | "
            f"{v['median_final_loss']:.2f} | {v['mean_seconds']:.0f} |"
        )
    lines += [
        "",
        f"| ablation ({N_ABLATE} targets) | forced exactly | median steps | median final loss |",
        "|---|---|---|---|",
    ]
    for k, v in out["ablation"].items():
        lines.append(
            f"| {k} | {v['success_rate']:.0%} | {v['median_steps_to_success'] or '-'} | {v['median_final_loss']:.2f} |"
        )

    if universal:
        lines += ["", "| universal suffix | most targets forced at once | steps |", "|---|---|---|"]
        for targets, u in zip(UNIVERSAL_SETS, universal):
            lines.append(f"| {' / '.join(targets)} | {max(u['forced'])}/{len(targets)} | {len(u['curve'])} |")
        out["universal"] = [
            {"targets": t, "forced": u["forced"][-1], "steps": len(u["curve"]), "suffix": u["suffix"]}
            for t, u in zip(UNIVERSAL_SETS, universal)
        ]

    if (R / "transfer.json").exists():
        tr = json.loads((R / "transfer.json").read_text())
        out["transfer"] = {
            "exact": sum(t["qwen_exact"] for t in tr),
            "n": len(tr),
            "median_loss_gpt2_suffix": float(np.median([t["qwen_loss_with_gpt2_suffix"] for t in tr])),
            "median_loss_init_suffix": float(np.median([t["qwen_loss_with_init_suffix"] for t in tr])),
        }
        t = out["transfer"]
        lines += [
            "",
            f"transfer gpt-2 -> qwen2.5-0.5b-instruct: {t['exact']}/{t['n']} forced; median target loss "
            f"{t['median_loss_gpt2_suffix']:.2f} with the gpt-2 suffix vs {t['median_loss_init_suffix']:.2f} "
            "with the unoptimised '! ! !' suffix",
        ]

    if (R / "defense.json").exists():
        d = json.loads((R / "defense.json").read_text())
        out["defense"] = {}
        lines += ["", "| perplexity filter | auroc | caught at 1% false alarms |", "|---|---|---|"]
        for key, name in (("log_ppl", "whole prompt"), ("max_window_log_ppl", "worst 10-token window")):
            adv = np.array([x[key] for x in d["adversarial"]])
            nat = np.array([x[key] for x in d["natural"]])
            y = np.r_[np.ones(len(adv)), np.zeros(len(nat))]
            s = np.r_[adv, nat]
            thr = np.quantile(nat, 0.99)
            out["defense"][name] = {
                "auroc": float(roc_auc_score(y, s)),
                "tpr_at_1pct_fpr": float((adv > thr).mean()),
                "n_adversarial": len(adv),
                "n_natural": len(nat),
            }
            v = out["defense"][name]
            lines.append(f"| {name} | {v['auroc']:.3f} | {v['tpr_at_1pct_fpr']:.0%} |")

    lines += ["", "| target | gcg suffix found on gpt-2 | steps |", "|---|---|---|"]
    for i, r in enumerate(main["gcg"][:8]):
        lines.append(f"| {TARGETS[i]} | `{r['suffix'].strip()}` | {r['success_step'] or 'not forced'} |")

    (R / "summary.json").write_text(json.dumps(out, indent=2))
    (R / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(main, ablation, universal)


def figures(main, ablation, universal):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for (k, runs), c in zip(main.items(), (INK, GOLD, RUST)):
        if not runs:
            continue
        x = np.arange(1, STEPS + 1)
        losses = padded(runs)
        axes[0].plot(x, np.median(losses, 0), color=c, lw=1.8, label=k)
        axes[0].fill_between(x, np.percentile(losses, 25, 0), np.percentile(losses, 75, 0), color=c, alpha=0.15)
        axes[1].plot(*success_curve(runs), color=c, lw=1.8, label=k)
    axes[0].set_yscale("log")
    axes[0].set_title("target loss, median and iqr")
    axes[1].set_title("share of targets forced exactly")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(F / "gcg_vs_baselines.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    cmap = plt.get_cmap("viridis")
    groups = [
        ([f"top-k {k}" for k in (8, 32, 128, 512)], "candidates per position (top-k)"),
        ([f"{n} suffix tokens" for n in (5, 10, 20)], "suffix length"),
    ]
    for ax, (keys, title) in zip(axes, groups):
        for j, k in enumerate(keys):
            if ablation[k]:
                ax.plot(*success_curve(ablation[k]), color=cmap(j / max(1, len(keys) - 1) * 0.85), lw=1.6, label=k)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("share forced exactly")
    fig.tight_layout()
    fig.savefig(F / "ablations.png")
    plt.close(fig)

    if universal:
        fig, ax = plt.subplots(figsize=(5.5, 3))
        for j, (u, c) in enumerate(zip(universal, (INK, SAGE))):
            ax.step(
                np.arange(1, len(u["forced"]) + 1),
                u["forced"],
                color=c,
                lw=1.6,
                where="post",
                label=f"set {j + 1}: " + ", ".join(t.split()[0].lower() + "..." for t in UNIVERSAL_SETS[j]),
            )
        ax.set_xlabel("step")
        ax.set_ylabel("targets forced by one suffix")
        ax.set_yticks(range(0, 4))
        ax.set_title("universal suffixes: one suffix, three targets")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(F / "universal.png")
        plt.close(fig)

    if (R / "defense.json").exists():
        d = json.loads((R / "defense.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        key = "max_window_log_ppl"
        adv = np.array([x[key] for x in d["adversarial"]])
        nat = np.array([x[key] for x in d["natural"]])
        bins = np.linspace(min(adv.min(), nat.min()), max(adv.max(), nat.max()), 40)
        axes[0].hist(nat, bins, color=SAGE, alpha=0.7, label="prompt + natural sentence", density=True)
        axes[0].hist(adv, bins, color=RUST, alpha=0.7, label="prompt + gcg suffix", density=True)
        axes[0].axvline(np.quantile(nat, 0.99), color="k", lw=0.8, ls="--")
        axes[0].set_xlabel("worst 10-token window, log perplexity under gpt-2")
        axes[0].set_title("gcg suffixes read as noise")
        axes[0].legend(fontsize=7)
        for k, c, name in (("log_ppl", GOLD, "whole prompt"), (key, INK, "worst 10-token window")):
            a = np.array([x[k] for x in d["adversarial"]])
            n = np.array([x[k] for x in d["natural"]])
            fpr, tpr, _ = roc_curve(np.r_[np.ones(len(a)), np.zeros(len(n))], np.r_[a, n])
            axes[1].plot(fpr, tpr, color=c, lw=1.6, label=name)
        axes[1].plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
        axes[1].set_xlabel("false positive rate")
        axes[1].set_ylabel("true positive rate")
        axes[1].set_title("perplexity filter roc")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(F / "perplexity_filter.png")
        plt.close(fig)
