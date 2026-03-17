import argparse
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from pyunigen import Sampler


DOMAIN = list(range(0,11))  # 0..10 inclusive


def exactly_one(lits):
    clauses = [list(lits)]
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            clauses.append([-lits[i], -lits[j]])
    return clauses


def build_problem_clauses():
    lit = 1
    decode = {}
    family_lits = {}

    for family in ("x0", "x1"):
        lits = []
        for value in DOMAIN:
            decode[lit] = (family, value)
            lits.append(lit)
            lit += 1
        family_lits[family] = lits

    clauses = []

    for family in ("x0", "x1"):
        clauses.extend(exactly_one(family_lits[family]))

    # Only allow pairs with x0 + x1 == 11.
    for a in DOMAIN:
        for b in DOMAIN:
            if a + b != 11:
                clauses.append([
                    -family_lits["x0"][a],
                    -family_lits["x1"][b],
                ])

    sampling_set = sorted(decode)
    return clauses, sampling_set, decode


def window_clauses(decode, current_state, D):
    """Restrict x0 and x1 to stay within D of the current state."""
    extra = []
    lo0 = max(0, current_state["x0"] - D)
    hi0 = min(10, current_state["x0"] + D)
    lo1 = max(0, current_state["x1"] - D)
    hi1 = min(10, current_state["x1"] + D)

    for lit, (family, value) in decode.items():
        if family == "x0" and not (lo0 <= value <= hi0):
            extra.append([-lit])
        elif family == "x1" and not (lo1 <= value <= hi1):
            extra.append([-lit])

    return extra


def permute_problem(clauses, sampling_set, decode, rng):
    old_vars = sampling_set[:]
    new_vars = sampling_set[:]
    rng.shuffle(new_vars)
    mp = dict(zip(old_vars, new_vars))

    def remap_lit(signed_lit):
        var = abs(signed_lit)
        mapped = mp[var]
        return mapped if signed_lit > 0 else -mapped

    permuted_clauses = [[remap_lit(l) for l in clause] for clause in clauses]
    rng.shuffle(permuted_clauses)
    permuted_sampling_set = [mp[v] for v in sampling_set]
    permuted_decode = {mp[v]: decode[v] for v in sampling_set}
    return permuted_clauses, permuted_sampling_set, permuted_decode


def call_sample_once(sampler, sampling_set):
    errors = []

    for mode in ("kw_num_sampling", "pos_num_sampling", "pos_num"):
        try:
            if mode == "kw_num_sampling":
                return sampler.sample(num=1, sampling_set=sampling_set)
            if mode == "pos_num_sampling":
                return sampler.sample(1, sampling_set)
            return sampler.sample(1)
        except Exception as e:
            errors.append(f"{mode}: {repr(e)}")

    raise RuntimeError("Could not call pyunigen successfully.\n" + "\n".join(errors))


def decode_sample(sample_lits, decode):
    out = {}
    for lit in sample_lits:
        if lit > 0:
            family, value = decode[lit]
            out[family] = value
    return out


def sample_one(base_clauses, sampling_set, decode, rng, current_state=None, D=0):
    clauses = list(base_clauses)
    if current_state is not None:
        clauses.extend(window_clauses(decode, current_state, D))

    pc, pss, pd = permute_problem(clauses, sampling_set, decode, rng)
    sampler = Sampler()
    for clause in pc:
        sampler.add_clause(clause)

    result = call_sample_once(sampler, pss)
    if isinstance(result, tuple) and len(result) == 3:
        _, _, samples = result
    else:
        samples = result

    if not samples:
        raise RuntimeError("PyUniGen returned no sample.")

    return decode_sample(samples[0], pd)


def init_plot():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    bars0 = axes[0].bar(DOMAIN, np.zeros(len(DOMAIN)))
    axes[0].set_title("x0 empirical distribution")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(0.0, 1.0)

    bars1 = axes[1].bar(DOMAIN, np.zeros(len(DOMAIN)))
    axes[1].set_title("x1 empirical distribution")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("probability")
    axes[1].set_ylim(0.0, 1.0)

    heat = np.zeros((len(DOMAIN), len(DOMAIN)), dtype=float)
    im = axes[2].imshow(heat, origin="lower", interpolation="nearest")
    axes[2].set_title("(x0, x1) heatmap")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x0")
    axes[2].set_xticks(range(len(DOMAIN)))
    axes[2].set_yticks(range(len(DOMAIN)))
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.ion()
    plt.show(block=False)

    return fig, bars0, bars1, im


def update_plot(fig, bars0, bars1, im, x0_counts, x1_counts, pair_counts, n, D, current_state):
    probs0 = [x0_counts[v] / n for v in DOMAIN]
    probs1 = [x1_counts[v] / n for v in DOMAIN]

    for bar, p in zip(bars0, probs0):
        bar.set_height(p)
    for bar, p in zip(bars1, probs1):
        bar.set_height(p)

    heat = np.zeros((len(DOMAIN), len(DOMAIN)), dtype=float)
    for (x0, x1), c in pair_counts.items():
        heat[x0, x1] = c / n
    im.set_data(heat)
    im.set_clim(vmin=0.0, vmax=max(heat.max(), 1e-12))

    title = f"Diagonal random-walk sampling — samples seen: {n}, D={D}"
    if current_state is not None:
        title += f", current=({current_state['x0']}, {current_state['x1']})"
    fig.suptitle(title, fontsize=12)
    fig.canvas.draw()
    fig.canvas.flush_events()


def run_demo(num_samples, refresh_every, delay, seed, D):
    if D < 0:
        raise ValueError("D must be nonnegative")

    rng = random.Random(seed)
    base_clauses, sampling_set, decode = build_problem_clauses()

    fig, bars0, bars1, im = init_plot()
    x0_counts = Counter()
    x1_counts = Counter()
    pair_counts = Counter()
    samples = []
    current_state = None

    for i in range(1, num_samples + 1):
        s = sample_one(base_clauses, sampling_set, decode, rng, current_state=current_state, D=D)
        current_state = s
        samples.append(s)
        x0_counts[s["x0"]] += 1
        x1_counts[s["x1"]] += 1
        pair_counts[(s["x0"], s["x1"])] += 1

        if i == 1 or i % refresh_every == 0 or i == num_samples:
            update_plot(fig, bars0, bars1, im, x0_counts, x1_counts, pair_counts, i, D, current_state)
            if delay > 0:
                plt.pause(delay)

    print(f"\nCollected {len(samples)} samples.\n")
    print("x0:", {k: round(v / len(samples), 4) for k, v in sorted(x0_counts.items())})
    print("x1:", {k: round(v / len(samples), 4) for k, v in sorted(x1_counts.items())})
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--refresh-every", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--D", type=int, default=1)
    args = parser.parse_args()

    run_demo(
        num_samples=args.num_samples,
        refresh_every=args.refresh_every,
        delay=args.delay,
        seed=args.seed,
        D=args.D,
    )
