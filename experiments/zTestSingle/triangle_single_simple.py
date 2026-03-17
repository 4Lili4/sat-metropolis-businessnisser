import argparse
import random
import sys
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from pyunigen import Sampler


# Base encoding for:
#   x0 in {0,1,2,3}
#   x1 in {0,1,2,3}
#   x2 in {0,1,2,3,4,5,6}
#   x0 + x1 = x2
DOMAINS = {"x0": list(range(4)), "x1": list(range(4)), "x2": list(range(7))}
LIT_INFO = {}
FAMILY_LITS = {}
CLAUSES = []
next_lit = 1

for family, values in DOMAINS.items():
    lits = []
    for value in values:
        LIT_INFO[next_lit] = (family, value)
        lits.append(next_lit)
        next_lit += 1
    FAMILY_LITS[family] = lits


def exactly_one(lits):
    out = [list(lits)]
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            out.append([-lits[i], -lits[j]])
    return out


for family in FAMILY_LITS:
    CLAUSES.extend(exactly_one(FAMILY_LITS[family]))

for a in range(4):
    for b in range(4):
        for c in range(7):
            if a + b != c:
                CLAUSES.append([
                    -FAMILY_LITS["x0"][a],
                    -FAMILY_LITS["x1"][b],
                    -FAMILY_LITS["x2"][c],
                ])

SAMPLING_SET = sorted(LIT_INFO)


def permute_problem(clauses, sampling_set, lit_info, rng):
    vars_ = sorted(lit_info)
    permuted = vars_[:]
    rng.shuffle(permuted)
    var_map = dict(zip(vars_, permuted))

    def remap(lit):
        return var_map[lit] if lit > 0 else -var_map[-lit]

    new_clauses = [[remap(lit) for lit in clause] for clause in clauses]
    new_sampling_set = [var_map[lit] for lit in sampling_set]
    new_lit_info = {var_map[lit]: lit_info[lit] for lit in vars_}
    return new_clauses, new_sampling_set, new_lit_info


class TriangleEngine:
    def __init__(self, seed=None):
        self.extra_clauses = []
        self.rng = random.Random(seed)

    def add_clause(self, clause):
        self.extra_clauses.append(list(clause))

    def sample_one(self):
        local_rng = random.Random(self.rng.getrandbits(64))
        clauses, sampling_set, lit_info = permute_problem(
            CLAUSES + self.extra_clauses,
            SAMPLING_SET,
            LIT_INFO,
            local_rng,
        )
        local_rng.shuffle(clauses)

        sampler = Sampler()
        for clause in clauses:
            sampler.add_clause(clause)

        try:
            result = sampler.sample(1, sampling_set)
        except TypeError:
            result = sampler.sample(1)

        samples = result[2] if isinstance(result, tuple) and len(result) == 3 else result
        if not samples:
            raise RuntimeError("PyUniGen returned no sample.")

        decoded = {}
        for lit in samples[0]:
            if lit > 0:
                family, value = lit_info[lit]
                decoded[family] = value
        return decoded


def init_plot():
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    state = {}
    for ax, family in zip(axes, ("x0", "x1", "x2")):
        domain = DOMAINS[family]
        bars = ax.bar(domain, np.zeros(len(domain)))
        ax.set_title(f"{family} empirical distribution")
        ax.set_xlabel("value")
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.0)
        state[family] = {"domain": domain, "bars": bars}
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, state



def update_plot(fig, state, counters, num_seen):
    for family in ("x0", "x1", "x2"):
        probs = [counters[family][v] / num_seen for v in state[family]["domain"]]
        for bar, prob in zip(state[family]["bars"], probs):
            bar.set_height(prob)
    fig.suptitle(f"PyUniGen one-by-one — samples seen: {num_seen}", fontsize=13)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)



def print_summary(samples):
    print(f"\nCollected {len(samples)} samples.\n")
    for family in ("x0", "x1", "x2"):
        counts = Counter(sample[family] for sample in samples)
        total = sum(counts.values())
        probs = {k: round(v / total, 4) for k, v in sorted(counts.items())}
        print(f"{family}: {probs}")



def run_demo(num_samples, refresh_every, delay, seed, change_after):
    engine = TriangleEngine(seed=seed)
    fig, state = init_plot()
    counters = {"x0": Counter(), "x1": Counter(), "x2": Counter()}
    samples = []

    for i in range(1, num_samples + 1):
        if change_after is not None and i == change_after + 1:
            engine.add_clause([-FAMILY_LITS["x0"][0]])  # x0 != 0
            print(f"Added dynamic constraint at sample {i}: x0 != 0")

        sample = engine.sample_one()
        samples.append(sample)
        for family in counters:
            counters[family][sample[family]] += 1

        if i % refresh_every == 0 or i == 1 or i == num_samples:
            update_plot(fig, state, counters, i)
            if delay > 0:
                time.sleep(delay)

    print_summary(samples)
    plt.ioff()
    plt.show()



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Simplified true one-by-one PyUniGen sampling with live plotting.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--refresh-every", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible controller seed.")
    parser.add_argument("--change-after", type=int, default=None, help="If set, add x0 != 0 after this many samples.")
    return parser.parse_args(argv)



def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_demo(
        num_samples=args.num_samples,
        refresh_every=args.refresh_every,
        delay=args.delay,
        seed=args.seed,
        change_after=args.change_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
