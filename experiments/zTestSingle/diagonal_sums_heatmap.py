#!/usr/bin/env python3
import argparse
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from pyunigen import Sampler


# Problem:
#   x0 in {0,...,10}
#   x1 in {0,...,10}
#   x2 in {2,4,6,...,20}
#   x0 + x1 = x2
#
# Note: because x0 and x1 are both inclusive 0..10, the natural heatmap is 11x11.

X0_DOMAIN = list(range(11))
X1_DOMAIN = list(range(11))
X2_DOMAIN = list(range(2, 21, 2))
ALLOWED_SUMS = set(X2_DOMAIN)


def exactly_one(lits):
    clauses = [list(lits)]
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            clauses.append([-lits[i], -lits[j]])
    return clauses


def build_problem(extra_clauses=None):
    domains = {
        "x0": X0_DOMAIN,
        "x1": X1_DOMAIN,
        "x2": X2_DOMAIN,
    }

    next_lit = 1
    decode = {}
    family_lits = {}
    clauses = []

    for family, values in domains.items():
        lits = []
        for value in values:
            lit = next_lit
            next_lit += 1
            lits.append(lit)
            decode[lit] = (family, value)
        family_lits[family] = lits
        clauses.extend(exactly_one(lits))

    x0_lit = {decode[lit][1]: lit for lit in family_lits["x0"]}
    x1_lit = {decode[lit][1]: lit for lit in family_lits["x1"]}
    x2_lit = {decode[lit][1]: lit for lit in family_lits["x2"]}

    # Forbid assignments where x2 != x0 + x1.
    for a in X0_DOMAIN:
        for b in X1_DOMAIN:
            for c in X2_DOMAIN:
                if a + b != c:
                    clauses.append([-x0_lit[a], -x1_lit[b], -x2_lit[c]])

    # Also forbid x0,x1 pairs whose sum is not in the allowed diagonal set.
    for a in X0_DOMAIN:
        for b in X1_DOMAIN:
            if (a + b) not in ALLOWED_SUMS:
                clauses.append([-x0_lit[a], -x1_lit[b]])

    if extra_clauses:
        clauses.extend(extra_clauses)

    sampling_set = sorted(decode)
    return clauses, sampling_set, decode, family_lits


def permute_instance(clauses, sampling_set, decode, rng):
    old_vars = list(sampling_set)
    new_vars = old_vars[:]
    rng.shuffle(new_vars)
    var_map = dict(zip(old_vars, new_vars))

    def remap(lit):
        sign = 1 if lit > 0 else -1
        return sign * var_map[abs(lit)]

    new_clauses = [[remap(l) for l in clause] for clause in clauses]
    rng.shuffle(new_clauses)
    new_sampling_set = [var_map[v] for v in sampling_set]
    new_decode = {var_map[v]: decode[v] for v in sampling_set}
    return new_clauses, new_sampling_set, new_decode


def sample_once(clauses, sampling_set, decode, rng):
    p_clauses, p_sampling_set, p_decode = permute_instance(clauses, sampling_set, decode, rng)

    # New Sampler each time because this local pyunigen binding is effectively single-use.
    sampler = Sampler()
    for clause in p_clauses:
        sampler.add_clause(clause)

    # Compatibility with local bindings: avoid unsupported keyword args.
    result = None
    errors = []
    for mode in ("pos2", "pos1", "kw"):
        try:
            if mode == "pos2":
                result = sampler.sample(1, p_sampling_set)
            elif mode == "pos1":
                result = sampler.sample(1)
            else:
                result = sampler.sample(num=1, sampling_set=p_sampling_set)
            break
        except Exception as e:
            errors.append(f"{mode}: {e!r}")

    if result is None:
        raise RuntimeError("Could not call pyunigen sampler.sample successfully.\n" + "\n".join(errors))

    samples = result[2] if isinstance(result, tuple) and len(result) == 3 else result
    if not samples:
        raise RuntimeError("PyUniGen returned no samples.")

    decoded = {}
    for lit in samples[0]:
        if lit > 0:
            family, value = p_decode[lit]
            decoded[family] = value
    return decoded


def init_plot():
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    bars0 = axes[0].bar(X0_DOMAIN, np.zeros(len(X0_DOMAIN)))
    axes[0].set_title("x0 empirical distribution")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(0.0, 1.0)

    bars1 = axes[1].bar(X1_DOMAIN, np.zeros(len(X1_DOMAIN)))
    axes[1].set_title("x1 empirical distribution")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("probability")
    axes[1].set_ylim(0.0, 1.0)

    heat = np.zeros((len(X0_DOMAIN), len(X1_DOMAIN)), dtype=float)
    im = axes[2].imshow(heat, origin="lower", interpolation="nearest", aspect="equal")
    axes[2].set_title("(x0, x1) heatmap")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x0")
    axes[2].set_xticks(range(len(X1_DOMAIN)))
    axes[2].set_yticks(range(len(X0_DOMAIN)))
    axes[2].set_xticklabels(X1_DOMAIN)
    axes[2].set_yticklabels(X0_DOMAIN)
    cbar = fig.colorbar(im, ax=axes[2])
    cbar.set_label("count")

    fig.tight_layout()
    return fig, bars0, bars1, im


def update_plot(fig, bars0, bars1, im, c0, c1, pair_counts, n):
    for bar, value in zip(bars0, X0_DOMAIN):
        bar.set_height(c0[value] / n)
    for bar, value in zip(bars1, X1_DOMAIN):
        bar.set_height(c1[value] / n)

    heat = np.zeros((len(X0_DOMAIN), len(X1_DOMAIN)), dtype=float)
    for (a, b), count in pair_counts.items():
        heat[a, b] = count
    im.set_data(heat)
    im.set_clim(vmin=0, vmax=max(1, heat.max()))

    fig.suptitle(
        f"One-by-one PyUniGen sampling | n={n} | allowed sums={sorted(ALLOWED_SUMS)}",
        fontsize=12,
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def print_summary(samples):
    print(f"\nCollected {len(samples)} samples.\n")
    for name in ("x0", "x1", "x2"):
        counts = Counter(s[name] for s in samples)
        total = sum(counts.values())
        probs = {k: round(v / total, 4) for k, v in sorted(counts.items())}
        print(f"{name}: {probs}")


def run_demo(num_samples, refresh_every, delay, seed=None, change_after=None):
    rng = random.Random(seed)

    extra_clauses = []
    clauses, sampling_set, decode, family_lits = build_problem(extra_clauses)

    fig, bars0, bars1, im = init_plot()

    c0 = Counter()
    c1 = Counter()
    pair_counts = Counter()
    samples = []

    for i in range(1, num_samples + 1):
        # Optional on-the-fly change: after this many samples, force x0 >= 5.
        if change_after is not None and i == change_after + 1:
            x0_small_lits = family_lits["x0"][:5]  # x0 in {0,1,2,3,4}
            extra_clauses.extend([[-lit] for lit in x0_small_lits])
            clauses, sampling_set, decode, family_lits = build_problem(extra_clauses)

        sample = sample_once(clauses, sampling_set, decode, rng)
        samples.append(sample)
        c0[sample["x0"]] += 1
        c1[sample["x1"]] += 1
        pair_counts[(sample["x0"], sample["x1"])] += 1

        if i == 1 or i % refresh_every == 0 or i == num_samples:
            update_plot(fig, bars0, bars1, im, c0, c1, pair_counts, i)
            plt.pause(delay)

    plt.ioff()
    print_summary(samples)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-by-one PyUniGen sampling for diagonal sums with heatmap.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--refresh-every", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--change-after", type=int, default=None)
    args = parser.parse_args()

    run_demo(
        num_samples=args.num_samples,
        refresh_every=args.refresh_every,
        delay=args.delay,
        seed=args.seed,
        change_after=args.change_after,
    )
