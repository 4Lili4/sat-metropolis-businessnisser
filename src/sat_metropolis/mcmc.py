import random
import numpy as np
import arviz as az
import os
from sat_metropolis import smt, sat
from z3 import Solver, Int, Sum, Goal, BitVec, ULE
from typing import Callable
import time
import logging


def sample_mh_trace_from_z3_model(backend: str,
                                  z3_problem,
                                  num_vars: int = None,  # mandatory for spur/cmsgen
                                  num_bits: int = None,  # mandatory for spur/cmsgen
                                  num_samples: int = 10000,
                                  num_chains: int = 4,
                                  timeout_sampler: int = 1800,  # seconds
                                  algo: str = 'MeGA',  # only for MegaSampler
                                  f: Callable[[dict[str, int]], float] = lambda x: 1,  # by default all samples have the same probability
                                  reweight_samples: bool = False,  # only for samplers that produce sets of unique samples
                                  print_z3_model: bool = False, # for debugging purposes it is possible to print the z3 model
                                  time_execution: bool = False): # if true, returns the execution time of metropolis and the SAT/SMT backend separaterly
    """This function runs sat-metropolis using the Z3 problem specified
    in `z3_problem` with the backend specified in `backend`. The
    backend may be 'spur', 'cmsgen' or 'megasampler'. The z3_problem
    must be either a Z3 solver or goal, depending on whether the
    backend is megasampler or spur/cmsgen respectively.

    Function `f` is the unnormalized probability density of the
    corresponding probabilistic model.

    For megasampler, it is possible to ask the algorithm to reweight
    samples, as megasampler samples only unique solutions.
    """

    start_time_sample_gen = time.time()
    samples = []
    if backend == 'megasampler':
        samples = smt.get_samples_smt_problem(
            z3_problem=z3_problem,
            algo=algo,
            timeout=timeout_sampler,
            num_samples=num_samples
        )
    elif backend == 'spur':
        samples = sat.get_samples_sat_problem(
            z3_problem=z3_problem,
            num_vars=num_vars,
            num_bits=num_bits,
            num_samples=num_samples,
            timeout=timeout_sampler,
            print_z3_model=print_z3_model
        )
    elif backend == 'cmsgen':
        samples = sat.get_samples_sat_cmsgen_problem(
            z3_problem=z3_problem,
            num_vars=num_vars,
            num_bits=num_bits,
            timeout=timeout_sampler,
            num_samples=num_samples,
            print_z3_model=print_z3_model
        )
    elif backend == 'unigen':
        samples = sat.get_samples_sat_unigen_problem(
            z3_problem=z3_problem,
            num_vars=num_vars,
            num_bits=num_bits,
            timeout=timeout_sampler,
            num_samples=num_samples,
            print_z3_model=print_z3_model
    )
    time_sample_gen = time.time() - start_time_sample_gen


    # run MCMC using the samples from spur or megasampler
    start_mcmc_time = time.time()
    trace = sample_mh_trace(num_samples=num_samples,
                            num_chains=num_chains,
                            solver_samples=samples,
                            f=f,
                            reweight_samples=reweight_samples)
    time_mcmc = time.time() - start_mcmc_time

    if time_execution:
        return (time_sample_gen, time_mcmc, trace)
    else:
        return trace


# TODO: add function for computing metropolis ratio
def sample_mh_trace(num_samples: int,
                    num_chains: int,
                    solver_samples: [dict[str, int]],
                    f: Callable[[dict[str, int]], float] = lambda x: 1,  # by default all samples have the same probability
                                                                    # this should mean uniform prior (if samples uniformly distributed)
                    reweight_samples: bool = False,  # only toggle for a set of unique samples, e.g., the set from MegaSampler
                    var_names: [str] = []) -> az.InferenceData:
    # NOTE: For now, we use the same solver samples for all chains. We
    # could consider using a fresh set of samples for each chain. If
    # we do so, `solver_samples` could be a list of lists of samples
    # and the number of chains could be determine by the length of
    # this list.

    # Informing users if there is mismatch between the specified
    # number of samples to generate and the number of samples in
    # `solver_samples`
    num_solver_samples = len(solver_samples)
    if num_samples > num_solver_samples:
        if not reweight_samples:
            num_samples = num_solver_samples
            print(f'The parameter `solver_samples` only contains {num_solver_samples} samples. Thus, every chain will contain {num_solver_samples} instead of {num_samples}. Try running the SAT/SMT sampler longer to obtain more samples.\n')
        else:
            # compute sample weights (unnormalized posterior probability)
            samples_weights = np.array([f(s) for s in solver_samples])            
            total_weight = sum(samples_weights)
            normalized_weights = samples_weights/total_weight
            num_reps = [int(num_samples*nw) for nw in normalized_weights]
            unflatten_reweighted_samples = [np.repeat(s, r) for (s, r) in
                                            zip(solver_samples, num_reps)]
            reweighted_samples = [s for ls in unflatten_reweighted_samples
                                  for s in ls]
            # (mutable) update of samples and num_samples
            # a bit ugly but works for now
            solver_samples = reweighted_samples
            num_samples = len(reweighted_samples)
            print(f'Samples have been reweighted. The sampler generated {num_solver_samples} and the reweighting process has generated {num_samples} (according to the unnormalized posterior probability specified in the input parameter `f`).\n')
    elif num_samples < num_solver_samples:
        print(f'The parameter `solver_samples` contains {num_solver_samples}, which is larger that the number of sample per chain specified: {num_samples}. Every chain will contain {num_samples} samples as specified. Nevertheless, we inform you that you can produce chains of up to {num_solver_samples} if you specify so in {num_samples}.')

    # obtain variable names from samples if not specified
    if var_names == []:
        var_names = solver_samples[0].keys()
    trace = {var: np.ndarray(shape=(num_chains, num_samples), dtype=int)
             for var in var_names}

    should_I_stay_or_should_I_go = lambda s_i, s_prime, alpha, u: s_prime if u <= alpha else s_i

    get_var_trace = lambda var_name, s: [s_i[var_name] for s_i in s]

    # generate each chain separately
    for chain in range(num_chains):
        # init memory for trace
        s: [dict[str, int]] = []
        # random shuffle of samples (to produce different chains with
        # the same solver sample, TO-DISCUSS)
        np.random.shuffle(solver_samples)
        # take inital state from sample
        s.append(solver_samples[0])
        # s_i: dict[str, int] = s_0
        for i in range(num_samples-1):
            # 1. draw a sample for candidate next state
            s_prime: dict[str, int] = solver_samples[i+1]
            # 2. compute acceptance ratio (s[i] is the current state)
            alpha: float = f(s_prime)/f(s[i])
            # 3. draw a sample uniformly from the unit interval -> U(0,1)
            u: float = np.random.uniform(0, 1)
            # 4. decide whether you move to next state
            s_next: dict[str, int] = should_I_stay_or_should_I_go(s[i],
                                                                  s_prime,
                                                                  alpha, u)
            # 5. add it to trace
            s.append(s_next)
        # switch to input format for arviz
        for var in var_names:
            trace[var][chain] = get_var_trace(var, s)
    return az.convert_to_inference_data(trace)


def sample_mh_trace_from_conf_matrix_smt(A: np.ndarray,
                                         y: np.ndarray,
                                         num_samples: int = 10000,
                                         num_chains: int = 4,
                                         print_z3_model: bool = False):
    """Function to sample directly from a linear problem of the form y =
    Ax where A is a configuration matrix and y are observed
    values. The unknown variables are the elements of the vector
    x. This is a common format for linear problems, and this function
    is specially useful to compare our tool with the work by Hazelton
    et al.

    The matrix A must be a binary configureation matrix with len(y)
    rows and len(x) columns. The vector y are the observations. The
    configuration matrix defines the linear relations between
    variables.

    The function builds a Z3 model with the constraints defined by the
    configuration matrix and the vector y. Then, it calls the regular
    `sample_mh_trace_from_z3_model` and invokes the `megasampler`
    backend.

    The number of variables and is inferred directly from the
    configuration matrix.
    """
    num_vars = A.shape[1]
    num_ys   = y.shape[0]

    x = [Int(f'x{i}') for i in range(num_vars)]

    s = Solver()

    for i in range(num_vars):
        s.add(x[i] >= 0)

    for i in range(len(y)):
        vars_ = [x[j] for j in range(num_vars) if A[i][j] == 1]
        s.add(Sum(vars_) == y[i])

    trace = sample_mh_trace_from_z3_model(backend='megasampler',
                                          z3_problem=s,
                                          num_samples=num_samples,
                                          num_chains=num_chains,
                                          print_z3_model=print_z3_model)

    return trace

def sample_mh_trace_from_conf_matrix_sat(A: np.ndarray,
                                         y: np.ndarray,
                                         num_bits: int,
                                         max_int_bv: int,
                                         backend: str = 'spur',
                                         num_samples: int = 10000,
                                         num_chains: int = 4,
                                         print_z3_model: bool = False):
    """Function to sample directly from a linear problem of the form y
    = Ax where A is a configuration matrix and y are observed
    values. The unknown variables are the elements of the vector
    x. This is a common format for linear problems, and this function
    is specially useful to compare our tool with the work by Hazelton
    et al.

    This is the SAT version of the function
    `sample_mh_trace_from_conf_matrix_smt` (see the documentation for
    more details). The main difference is that this function requires
    the bit-vector size and a max bit-vector value to avoid overflows.
    """
    num_vars = A.shape[1]
    num_ys   = y.shape[0]

    x = [BitVec(f'x{i}', num_bits) for i in range(num_vars)]

    g = Goal()

    sat.add_bool_vars_to_goal(g, x)

    for i in range(len(y)):
        vars_ = [x[j] for j in range(num_vars) if A[i][j] == 1]
        g.add(sat.addition_does_not_overflow(vars_))
        g.add(Sum(vars_) == y[i])

    for i in range(num_vars):
        g.add(ULE(0, x[i]))
        g.add(ULE(x[i], max_int_bv))  # adding also max
                                      # value to avoid overflows

    trace = sample_mh_trace_from_z3_model(backend=backend,
                                          z3_problem=g,
                                          num_vars=num_vars,
                                          num_bits=num_bits,
                                          num_samples=num_samples,
                                          num_chains=num_chains,
                                          print_z3_model=print_z3_model)

    return trace


def __get_sample(var, samples):
    return lambda i: samples[i][var]


def sample_mh_trace_uniform_hardcoded(
        num_samples: int,
        num_chains: int,
        solver_samples: dict[str, int],
        var_names: [str] = []) -> dict[str, [[int]]]:
    if var_names == []:
        # if not specified, we get var names from the sample
        var_names = solver_samples[0].keys()
    trace = {var: [] for var in var_names}
    for var in var_names:
        map_for_samples = np.vectorize(__get_sample(var, solver_samples))
        samples_vector = np.random.randint(0, num_samples-1,
                                           size=(num_chains, num_samples))
        trace[var] = map_for_samples(samples_vector)
    return trace
