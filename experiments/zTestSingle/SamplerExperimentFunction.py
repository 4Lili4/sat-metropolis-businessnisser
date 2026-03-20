import random
import time
from collections import Counter

from pyunigen import Sampler


def run_diagonal_walk_experiment(num_samples, D_values=None, seed=None):
    """
    Run the constrained sampling experiment for multiple D values.

    Returns a dictionary keyed by D. For each D, the value is a dict with:
      - "trace_counts":
            list of length num_samples
            trace_counts[i] is a dict mapping (x0, x1) -> cumulative count
            after sample i+1
      - "samples":
            list of sampled states, each as {"x0": int, "x1": int}
      - "sample_times":
            list of per-sample elapsed times in seconds
      - "avg_sample_time":
            average time spent inside the sampling call for one sample
      - "total_time":
            total elapsed sampling time for this D
    """
    DOMAIN = list(range(0, 11))  # 0..10 inclusive

    if D_values is None:
        D_values = list(range(0, 11))

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

        t0 = time.perf_counter()
        result = call_sample_once(sampler, pss)
        elapsed = time.perf_counter() - t0

        if isinstance(result, tuple) and len(result) == 3:
            _, _, samples = result
        else:
            samples = result

        if not samples:
            raise RuntimeError("PyUniGen returned no sample.")

        decoded = decode_sample(samples[0], pd)
        return decoded, elapsed

    base_clauses, sampling_set, decode = build_problem_clauses()

    results = {}

    for D in D_values:
        rng = random.Random(None if seed is None else seed + 1000 * D)

        current_state = None
        pair_counts = Counter()
        trace_counts = []
        samples = []
        sample_times = []

        for _ in range(num_samples):
            s, elapsed = sample_one(
                base_clauses,
                sampling_set,
                decode,
                rng,
                current_state=current_state,
                D=D,
            )

            current_state = s
            samples.append(s)
            sample_times.append(elapsed)

            pair_counts[(s["x0"], s["x1"])] += 1
            trace_counts.append(dict(pair_counts))

        total_time = sum(sample_times)
        avg_sample_time = total_time / num_samples if num_samples > 0 else 0.0

        results[D] = {
            "trace_counts": trace_counts,
            "samples": samples,
            "sample_times": sample_times,
            "avg_sample_time": avg_sample_time,
            "total_time": total_time,
        }

    return results