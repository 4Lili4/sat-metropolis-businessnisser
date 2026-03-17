from __future__ import annotations

"""
True one-by-one PyUniGen sampling for the triangle SAT toy problem,
with per-draw variable-ID permutation to avoid systematic bias toward the
same variable family.

Why this version exists:
- Some pyunigen builds make Sampler single-use.
- Some pyunigen builds do NOT accept a `seed=` keyword.
- Rebuilding from the same CNF can make the solver keep choosing the same
  early variable family (often x0) on every draw.

This file keeps the semantics identical, but on each draw it randomly renames
all literals before building the sampler. That changes the solver's internal
variable ordering while preserving the constraint set.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import argparse
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from pyunigen import Sampler


@dataclass(frozen=True)
class IndicatorVar:
    family: str
    value: int
    lit: int


def exactly_one_clauses(lits: Sequence[int]) -> List[List[int]]:
    clauses = [list(lits)]
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            clauses.append([-lits[i], -lits[j]])
    return clauses


def build_triangle_cnf():
    domains = {
        "x0": list(range(4)),
        "x1": list(range(4)),
        "x2": list(range(7)),
    }

    next_lit = 1
    lit_to_var: Dict[int, IndicatorVar] = {}
    family_lits: Dict[str, List[int]] = {}
    clauses: List[List[int]] = []

    for family, values in domains.items():
        lits = []
        for value in values:
            lit = next_lit
            next_lit += 1
            lit_to_var[lit] = IndicatorVar(family=family, value=value, lit=lit)
            lits.append(lit)
        family_lits[family] = lits

    for family in ("x0", "x1", "x2"):
        clauses.extend(exactly_one_clauses(family_lits[family]))

    x0_lit_for = {lit_to_var[lit].value: lit for lit in family_lits["x0"]}
    x1_lit_for = {lit_to_var[lit].value: lit for lit in family_lits["x1"]}
    x2_lit_for = {lit_to_var[lit].value: lit for lit in family_lits["x2"]}

    for a in range(4):
        for b in range(4):
            for c in range(7):
                if a + b != c:
                    clauses.append([-x0_lit_for[a], -x1_lit_for[b], -x2_lit_for[c]])

    sampling_set = sorted(lit_to_var.keys())
    return clauses, sampling_set, lit_to_var


def decode_sample(sample_lits: Iterable[int], lit_to_var: Dict[int, IndicatorVar]) -> Dict[str, int]:
    decoded: Dict[str, int] = {}
    for signed_lit in sample_lits:
        if signed_lit > 0:
            info = lit_to_var[signed_lit]
            decoded[info.family] = info.value
    return decoded


def apply_var_permutation(
    clauses: Sequence[Sequence[int]],
    sampling_set: Sequence[int],
    lit_to_var: Dict[int, IndicatorVar],
    rng: random.Random,
):
    """Randomly rename variables 1..n to a new permutation 1..n.

    This preserves satisfiability and the represented model distribution,
    but changes the solver's internal branching order enough to avoid pinning
    the same variable family on every fresh sampler build.
    """
    all_vars = sorted(abs(l) for l in lit_to_var.keys())
    permuted = all_vars[:]
    rng.shuffle(permuted)
    var_map = {old: new for old, new in zip(all_vars, permuted)}

    def remap_lit(lit: int) -> int:
        sign = 1 if lit > 0 else -1
        return sign * var_map[abs(lit)]

    remapped_clauses = [[remap_lit(l) for l in clause] for clause in clauses]
    remapped_sampling_set = [var_map[l] for l in sampling_set]

    remapped_lit_to_var: Dict[int, IndicatorVar] = {}
    for old_lit, info in lit_to_var.items():
        new_lit = var_map[old_lit]
        remapped_lit_to_var[new_lit] = IndicatorVar(info.family, info.value, new_lit)

    return remapped_clauses, remapped_sampling_set, remapped_lit_to_var


def make_sampler(clauses: Sequence[Sequence[int]], rng: random.Random) -> Sampler:
    sampler = Sampler()
    shuffled = list(clauses)
    rng.shuffle(shuffled)
    for clause in shuffled:
        sampler.add_clause(list(clause))
    return sampler


def sample_one_raw(sampler: Sampler, sampling_set: Sequence[int]):
    """
    Try only call signatures that do not use unsupported keywords.
    Each attempt must be on a fresh sampler object because some builds are single-use.
    """
    try:
        return sampler.sample(1, list(sampling_set))
    except TypeError:
        return sampler.sample(1)


def init_live_plot():
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    plot_state = {}
    families = [("x0", list(range(4))), ("x1", list(range(4))), ("x2", list(range(7)))]

    for ax, (family, domain) in zip(axes, families):
        bars = ax.bar(domain, np.zeros(len(domain)))
        ax.set_title(f"{family} empirical distribution")
        ax.set_xlabel("value")
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.0)
        plot_state[family] = {"domain": domain, "bars": bars}

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, plot_state


def update_live_plot(fig, plot_state, counters, num_seen):
    for family in ("x0", "x1", "x2"):
        domain = plot_state[family]["domain"]
        bars = plot_state[family]["bars"]
        probs = [counters[family][v] / num_seen for v in domain]
        for bar, prob in zip(bars, probs):
            bar.set_height(prob)

    fig.suptitle(f"PyUniGen one-by-one with literal permutation — samples seen: {num_seen}", fontsize=13)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


class TriangleEngine:
    def __init__(self, base_seed: int | None = None):
        self.base_clauses, self.base_sampling_set, self.base_lit_to_var = build_triangle_cnf()
        self.extra_clauses: List[List[int]] = []
        self._seed_rng = random.Random(base_seed) if base_seed is not None else random.Random(os.urandom(16))

    @property
    def clauses(self) -> List[List[int]]:
        return self.base_clauses + self.extra_clauses

    @property
    def sampling_set(self) -> List[int]:
        return list(self.base_sampling_set)

    @property
    def lit_to_var(self) -> Dict[int, IndicatorVar]:
        return dict(self.base_lit_to_var)

    def add_clause(self, clause: Sequence[int]) -> None:
        self.extra_clauses.append(list(clause))

    def sample_one(self) -> Dict[str, int]:
        draw_seed = self._seed_rng.getrandbits(64)
        local_rng = random.Random(draw_seed)

        remapped_clauses, remapped_sampling_set, remapped_lit_to_var = apply_var_permutation(
            self.clauses,
            self.sampling_set,
            self.lit_to_var,
            local_rng,
        )

        sampler = make_sampler(remapped_clauses, local_rng)
        result = sample_one_raw(sampler, remapped_sampling_set)

        if isinstance(result, tuple) and len(result) == 3:
            _, _, samples = result
        else:
            samples = result

        if not samples:
            raise RuntimeError("PyUniGen returned no sample.")

        return decode_sample(samples[0], remapped_lit_to_var)


def print_summary(samples: List[Dict[str, int]]):
    print(f"\nCollected {len(samples)} samples.\n")
    for family in ("x0", "x1", "x2"):
        counts = Counter(s[family] for s in samples)
        total = sum(counts.values())
        probs = {k: round(v / total, 4) for k, v in sorted(counts.items())}
        print(f"{family}: {probs}")


def run_demo(num_samples: int, refresh_every: int, delay: float, seed: int | None, change_after: int | None):
    engine = TriangleEngine(base_seed=seed)
    fig, plot_state = init_live_plot()
    counters = {"x0": Counter(), "x1": Counter(), "x2": Counter()}
    samples: List[Dict[str, int]] = []

    # In the original naming, literal 1 means x0 == 0.
    x0_zero_lit = 1

    for i in range(1, num_samples + 1):
        if change_after is not None and i == change_after + 1:
            engine.add_clause([-x0_zero_lit])
            print(f"Added dynamic constraint at sample {i}: x0 != 0")

        sample = engine.sample_one()
        samples.append(sample)

        counters["x0"][sample["x0"]] += 1
        counters["x1"][sample["x1"]] += 1
        counters["x2"][sample["x2"]] += 1

        if i % refresh_every == 0 or i == 1 or i == num_samples:
            update_live_plot(fig, plot_state, counters, i)
            if delay > 0:
                time.sleep(delay)

    print_summary(samples)
    plt.ioff()
    plt.show()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="True one-by-one PyUniGen sampling with live plotting and per-draw literal permutation.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--refresh-every", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible controller seed.")
    parser.add_argument("--change-after", type=int, default=None, help="If set, add x0 != 0 after this many samples.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
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
