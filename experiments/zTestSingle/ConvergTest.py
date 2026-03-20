import argparse
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from pyunigen import Sampler


DOMAIN = list(range(0, 11))  # 0..10 inclusive
VALID_PAIRS = [(a, b) for a in DOMAIN for b in DOMAIN if a + b == 11]
NUM_VALID = len(VALID_PAIRS)


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


def convergence_score(pair_counts, n):
    """
    Normalized convergence to the uniform distribution over VALID_PAIRS.

    score = 1 - TV(empirical, uniform_target) / (1 - 1/K)

    This maps:
      - a delta distribution on one valid state -> 0
      - exact uniform over all K valid states -> 1
    """
    if n <= 0:
        return 0.0

    empirical = np.array([pair_counts[p] / n for p in VALID_PAIRS], dtype=float)
    target = np.full(NUM_VALID, 1.0 / NUM_VALID, dtype=float)

    tv = 0.5 * np.abs(empirical - target).sum()
    worst_tv = 1.0 - (1.0 / NUM_VALID)

    # Guard for numerical noise
    score = 1.0 - (tv / worst_tv)
    return max(0.0, min(1.0, score))


def run_single_trajectory(base_clauses, sampling_set, decode, num_samples, D, seed):
    rng = random.Random(seed)
    current_state = None
    pair_counts = Counter()
    scores = []

    for i in range(1, num_samples + 1):
        s = sample_one(
            base_clauses,
            sampling_set,
            decode,
            rng,
            current_state=current_state,
            D=D,
        )
        current_state = s
        pair_counts[(s["x0"], s["x1"])] += 1
        scores.append(convergence_score(pair_counts, i))

    return np.array(scores, dtype=float)


def run_all_Ds(num_samples, num_trials, seed):
    base_clauses, sampling_set, decode = build_problem_clauses()
    D_values = list(range(0, 11))

    all_curves = {}

    for D in D_values:
        trial_curves = []
        for trial in range(num_trials):
            trial_seed = None if seed is None else seed + 1000 * D + trial
            curve = run_single_trajectory(
                base_clauses=base_clauses,
                sampling_set=sampling_set,
                decode=decode,
                num_samples=num_samples,
                D=D,
                seed=trial_seed,
            )
            trial_curves.append(curve)

        trial_curves = np.vstack(trial_curves)
        all_curves[D] = {
            "mean": trial_curves.mean(axis=0),
            "std": trial_curves.std(axis=0),
        }

    return all_curves


def plot_convergence(all_curves, num_samples, num_trials):
    x = np.arange(1, num_samples + 1)

    plt.figure(figsize=(12, 7))

    for D in sorted(all_curves):
        mean_curve = all_curves[D]["mean"]
        plt.plot(x, mean_curve, label=f"D={D}")

    plt.xlabel("Number of samples")
    plt.ylabel("Convergence score")
    plt.title(
        f"Convergence to uniform over valid states (x0 + x1 = 11)\n"
        f"Averaged over {num_trials} trial(s)"
    )
    plt.ylim(0.0, 1.02)
    plt.xlim(1, num_samples)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()


def print_summary(all_curves):
    print("\nFinal convergence scores:")
    for D in sorted(all_curves):
        final_mean = all_curves[D]["mean"][-1]
        final_std = all_curves[D]["std"][-1]
        print(f"D={D:2d} -> final mean score = {final_mean:.4f} ± {final_std:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Average convergence curves over this many repeated runs per D.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    all_curves = run_all_Ds(
        num_samples=args.num_samples,
        num_trials=args.num_trials,
        seed=args.seed,
    )

    print_summary(all_curves)
    plot_convergence(
        all_curves=all_curves,
        num_samples=args.num_samples,
        num_trials=args.num_trials,
    )


if __name__ == "__main__":
    main()