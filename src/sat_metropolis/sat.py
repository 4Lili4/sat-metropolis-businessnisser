from subprocess import call
from z3 import Goal, BitVecSort, Bool, And, BVAddNoOverflow, BVMulNoOverflow, BVSubNoUnderflow, Then, simplify, is_app_of, Z3_OP_NOT, Not, BoolRef, solve, unsat, sat, Solver, ULE, BitVec
from z3 import BitVec, BitVecVal, Goal, ULE, unsat
import math
import random
import numpy as np
import re
import os, sys
from warnings import warn
import time
import pyunigen as uni
import pickle
import random
import pycmsgen as cms
import os
import time
import random
import numpy as np
from z3 import Goal, unsat
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", ".."))
sys.path.append(project_root)

import src.sat_metropolis.utils as utils

# from utils import * 
# import utils

# NOTE: The function below creates a new variable for each bit in the
#       bit-vector.  Then, it maps the correspoding variable of the
#       bit vectors in `var_list` to the corresponding bit created
#       below.  The process is a bit convoluted. First, it creates the
#       Bool variable and adds it to a bitmap. Then, it creates a mask
#       that will be used to isolate the bit in the bit-vector
#       above. This last step is done by adding the final
#       constraint. The mask is of binary number of the kind 2^i with
#       i the bit position we are defining. Note that the binary
#       representation of 2^i numbers contains only a single bit equal
#       1. Then the operation and (&) is used with the bit-vector in
#       var_list to isolate the bit of interest. The result of this
#       operation is compared with the mask again, this is just to
#       create predicate that return 1 if the bit is 1 and 0 if it is
#       0. Finally, the result of this operation must be equal to the
#       bit representating that position of the bit-vector. This is
#       added as a constraint to the problem.


def get_bit_digits(num_bits: int) -> int:
    if num_bits <= 0:
        raise RuntimeError("num_bits must be positive")
    return len(str(num_bits - 1))


def blasted_bit_name(var_idx: int, bit_idx: int, num_bits: int) -> str:
    bit_digits = get_bit_digits(num_bits)
    return f'x{var_idx}{bit_idx:0{bit_digits}d}'


def add_bool_vars_to_goal(g: Goal, var_list: [BitVecSort]):
    bitmap = {}
    num_vars = len(var_list)

    if num_vars == 0:
        return

    num_bits = var_list[0].size()
    bit_digits = get_bit_digits(num_bits)

    for j in range(num_vars):
        if var_list[j].size() != num_bits:
            raise RuntimeError("All bit-vectors must have the same size")

        for i in range(num_bits):
            bitmap[var_list[j], i] = Bool(f'x{j}{i:0{bit_digits}d}')
            mask = BitVecVal(1 << i, num_bits)
            g.add(bitmap[(var_list[j], i)] == ((var_list[j] & mask) == mask))


# adds a constraint regarding the summation of all elements in xs
# (removes the overflow) and returns the variable
def addition_does_not_overflow(xs: [], signed=False):
    sofar      = 0
    noOverflow = True
    for x in xs:
        noOverflow = And(noOverflow, BVAddNoOverflow(x, sofar, signed))
        sofar += x
    return noOverflow


# same as above but multiplication
def multi_does_not_overflow(xs, signed=False):
    sofar      = 1
    noOverflow = True
    for x in xs:
        noOverflow = And(noOverflow, BVMulNoOverflow(x, sofar, signed))
        sofar *= x
    return noOverflow


# same as above but substraction
def sub_does_not_underflow(xs, signed=False):
    sofar      = 0
    noUnderflow = True
    for x in xs:
        noUnderflow = And(noUnderflow, BVSubNoUnderflow(x, sofar, signed))
        sofar = x-sofar
    return noUnderflow


def convert_to_cnf_and_dimacs_simp(g: Goal) -> (
        [[str]],
        int,
        dict[int, BoolRef]): # type: ignore
    # Z3 bit-blasting from De Moura's post -> https://stackoverflow.com/a/13059908
    t = Then('simplify', 'bit-blast', 'tseitin-cnf')
    subgoal = t(g)
    assert len(subgoal) == 1

    count_vars = 0
    map_vars_nums = {}
    map_nums_vars = {}
    dimacs_clauses = [[]]

    # subgoal[0] contains all clauses so we iterate over them
    for c in subgoal[0]:
        # temp var to store processed clause
        clause = []
        # iterate over literals of a clause
        # the +1 is to enter the loop when c has 0 (sub)arguments
        for i in range(c.num_args()+1):
            # exit the loop if we are in the last iteration of c with
            # more than 0 arguments
            if i == c.num_args() and i > 0:
                break
            # if the clause has only one literal then the clause is
            # the literal (it is of the form ¬x)
            # otherwise (it has from l_0 \/ l_1 \/ ...) and we use
            # arg(i) to select the literal
            lit = c.arg(i) if c.num_args() > 1 else c
            # check whether the literal is the negation, i.e., ¬x
            negation = is_app_of(lit, Z3_OP_NOT)
            # if negated the variable is arg(0) otherwise the literal
            # is the variable
            var = lit.arg(0) if negation else lit
            # if the variable is not a key in the map var -> num, then
            # we add it. Apparently, the condition below is computed
            # in O(1) time and space!
            if var not in map_vars_nums:
                # we select a new var number (for dimacs cnf format)
                count_vars = count_vars + 1
                # added to the two maps num -> var and var -> num
                map_vars_nums[var] = count_vars
                map_nums_vars[count_vars] = var
            # we append a string modeling the literal in dimacs cnf format
            clause.append(('-' if negation else '')+str(map_vars_nums[var]))
        # we add the end of line character for dimacs cnf format
        dimacs_clauses.append(clause+['0'])


    # we add the header of the dimacs cnf format
    n_varibles = len(map_vars_nums.keys())
    n_constraints = len(dimacs_clauses)-1
    s = "p cnf " + str(n_varibles) + " " + str(n_constraints)
    dimacs_clauses[0].append(s)
    return (dimacs_clauses, n_varibles, map_nums_vars)


def convert_to_cnf_and_dimacs(g: Goal):
    ## WARNING: Deprecated, we use convert_to_cnf_and_dimacs_simp (see above)

    # copied from De Moura's post -> https://stackoverflow.com/a/13059908
    t = Then('simplify', 'bit-blast', 'tseitin-cnf')
    subgoal = t(g)
    assert len(subgoal) == 1

    var_count = 1
    constraint_count = 0
    varibles_number = {}
    varibles_var = {}
    negation = False
    dimacs_format = [[]]

    for c in subgoal[0]:
        constraint_count += 1
        dimacs_format.append([])
        for i in range(c.num_args()):
            # Save varible names in dictonary
            # print(c.arg(i))
            if (c.num_args() == 1):
                if (is_app_of(c, Z3_OP_NOT)):
                    var = simplify(Not(c.arg(i)))
                    negation = True
            if (is_app_of(c.arg(i), Z3_OP_NOT)):
                var = simplify(Not(c.arg(i)))
                negation = True
            else:
                var = c.arg(i)
            if (var not in varibles_var):
                varibles_var[var] = var_count
                varibles_number[var_count] = var
                var_count += 1
            if (negation):
                dimacs_format[constraint_count].append(-varibles_var[var])
            else:
                dimacs_format[constraint_count].append(varibles_var[var])
            negation = False
        if (c.num_args() == 0):
            var = c
            if (var not in varibles_var):
                varibles_var[var] = var_count
                varibles_number[var_count] = var
                var_count += 1
            dimacs_format[constraint_count].append(varibles_var[var])
        dimacs_format[constraint_count].append(0)

    # appending heading line
    n_varibles = len(varibles_var)
    n_constraints = len(dimacs_format)-1
    s = "p cnf " + str(n_varibles) + " " + str(n_constraints)
    dimacs_format[0].append(s)
    return (dimacs_format, n_varibles, varibles_number)


def save_dimacs(g: Goal, output_filepath: str) -> (int, dict):
    # NOTE: We return n_variables because it is later used to parse
    #       the output of spur.
    #       Also, we return the map variables_number because we need
    #       to map back the results from spur to its Z3 variables.

    # NOTE: We use `convert_to_cnf_and_dimacs_simp`
    (dimacs_format, n_varibles, varibles_number) = convert_to_cnf_and_dimacs_simp(g)

    path = '/'.join(output_filepath.split('/')[:-1])
    if not os.path.exists(path) and len(path) > 0:
        raise RuntimeError(f'Directory {path} not found')

    with open(output_filepath, 'w') as file:
        for row in dimacs_format:
            print(row)
            file.write(' '.join([str(item) for item in row]))
            file.write('\n')
    return (n_varibles, varibles_number)


def execute_spur(input_filepath: str,
                 num_samples: int = 10000,
                 timeout: int = 1800  # seconds
                 ) -> None:

    """Executes spur on the specified input file `input_filepath`. By
    default, it generates 10000 samples. The samples are created on
    the same directory as the input file. The name of the output file
    is `samples_<name_of_input_file>.txt`.

    The function assumes that the spur executable is accessible
    by calling `spur`.
    """
    call(['spur',                  # - spur command (hardcoded, it
                                   #   assumes accessible for this user)
          '-s', str(num_samples),  # - number of samples
          '-t', str(timeout),      # - timeout in seconds (crashes
                                   #   if it cannot generate all samples
                                   #   before the specified timeout)
          '-cnf',                  # - Input format (DIMACS cnf)
          input_filepath])         # - input file path


def __repl_fun(match):
    # randint, discrete uniform distribution
    return str(random.randint(0, 1))


def parse_spur_samples(input_dir: str,
                       input_file: str,
                       num_samples: int,
                       num_variables: int) -> list[list[bool]]:

    spur_samples_filepath = f'{input_dir}/samples_{input_file[:-4]}.txt'
    n = num_samples
    m = num_variables
    # reserver space for the samples
    samples1 = np.zeros((n, m), dtype=np.int_)
    # open samples file
    with open(spur_samples_filepath, 'r') as f:
        # index for samples
        i = 0
        # iterave over lines of the samples file
        for line in f:
            # only consider lines starting with a digit
            # (these are the lines containing samples)
            if ((line[0]).isdigit()):
                # number of occurences of sample
                # (SPUR does not repeat same samples,
                #  instead it specifies the number of times it occurs)
                n_ = int(line.split(',')[0])
                # sample (as a string of 0s and 1s and *s ending in a \n)
                sampel2 = line.split(',')[1]
                # remove the \n
                sampel = sampel2.split('\n')[0]
                # create a different sample for each occurence
                for j in range(n_):
                    # replace each * with a 1 o 0 randomly
                    # (this is sound due to the meaning of *)
                    replaced = re.sub('\*', __repl_fun, sampel)
                    # add the sample to the result list of samples
                    samples1[i] = list(map(int, replaced))
                    # increase the sample index
                    i += 1
    return samples1


def execute_cmsgen(input_filepath: str,
                   output_filepath: str,
                   num_samples: int = 10000,
                   timeout: int = 1800
                   ) -> None:

    """Executes cmsgen on the specified input file
    `input_filepath`. By default, it generates 10000 samples. The
    samples are added to the file specified in `output_filepath`.

    The function assumes that the spur executable is accessible
    by calling `cmsgen`.

    """
    call(['cmsgen',                         # - cmsgen command
                                            #   (hardcoded, it assumes
                                            #   accessible for this
                                            #   user)
          '--samples', str(num_samples),    # - number of samples
          '--samplefile', output_filepath,  # - output file path
          input_filepath],                  # - input file path
         timeout=timeout)  # timeout in seconds


def parse_cmsgen_samples(input_dir: str,
                         input_file: str,
                         num_samples: int,
                         num_variables: int) -> list[list[bool]]:
    cmsgen_samples_filepath = f'{input_dir}/{input_file}'
    samples = []
    with open(cmsgen_samples_filepath, 'r') as file:
        for line in file:
            sample = [int(int(l) >= 0) for l in line.split(' ')][:-1]
            samples.append(sample)
    samples_numpy = np.array(samples, dtype=np.int_)
    if not ((num_samples, num_variables) == samples_numpy.shape):
                raise RuntimeError(f'The number of samples or number of variables do not match.\n \
                CMSGen generated {samples_numpy.shape[0]} samples on {samples_numpy.shape[1]} variables, but you specified {num_samples} samples and {num_variables} variables')
    return samples_numpy


def map_spur_samples_to_z3_vars(map_number_z3_var: dict[int, BoolRef],
                                num_variables: int,
                                spur_parsed_samples: list[list[bool]]
                                ) -> dict[str, list[bool]]:
    """This function takes as in put the mapping of integer to Z3 Bool
    variables returned by `convert_to_cnf_and_dimacs` in variable
    `variables_number`, and converts it into a dictonary with variable
    names as strings and the list of (spur) sampled values for each
    variable.

    For now, it is also required to specify, the number of variables
    in `map_number_z3_var`, althought this information could be
    obtained from `map_number_z3_var`.

    In this funciton, we still work with blasted variables.
    """
    ## NOTE: This function is a just an intermediate step to get the
    ## map from str of the variables (after bit-blasting) and an array
    ## of samples. The array of samples are the values of the variable
    ## specified in the key for each sample.

    # init the output map (str -> [bool])
    variable_values = {}
    # iterate over all variables (after bit-blasting) (this could be
    # replace by iterating over the keys of map_number_z3_var)
    for i in range(num_variables):
        # convert the z3 variable into str
        # (I assume this conversion works for all possible variables)
        z3_var_str = str(map_number_z3_var[i+1])

        # assign to the output map the samples for variable i+1.
        # We use `i` in spur_parsed_samples because the indexes start
        # from 0, but the int associated to variables in cnf format
        # starts in 1
        variable_values[z3_var_str] = spur_parsed_samples[:, i]
    return variable_values


def reverse_bit_blasting_simp(variable_values: dict[str, list[bool]],
                              num_samples: int,
                              num_vars: int,
                              num_bits: int) -> list[dict[str, int]]:
    def from_bin_to_dec(i, s, num_bits, map_variable_values):
        total = 0
        for j in range(num_bits):
            key = blasted_bit_name(i, j, num_bits)
            total += (2 ** j) * map_variable_values[key][s]
        return total

    solver_samples = [
        { f'x{i}': from_bin_to_dec(i, s, num_bits, variable_values)
            for i in range(num_vars)}
        for s in range(num_samples)
    ]

    return solver_samples



def __check_goal(z3_goal: Goal):
    """Helper funciton to easily check satisfiability of a Z3 Goal
    object.
    """
    sol = Solver()
    sol.add(z3_goal)
    return sol.check()


def get_samples_sat_problem(z3_problem: Goal,
                            num_vars: int,  # number of varibles unblasted
                            num_bits: int,  # number of bits of BitVectors
                                            # (assumption: all the same)
                            num_samples: int = 10000,
                            sanity_check_problem: bool = True,
                            sanity_check_samples: bool = False,
                            timeout: int = 1800,  # seconds
                            print_z3_model: bool = False):

    if sanity_check_problem and __check_goal(z3_problem) == unsat:
        raise RuntimeError('The problem you input is UNSAT')

    if print_z3_model:
        print(z3_problem)

    CWD = os.getcwd()

    SPUR_INPUT_DIR = 'spur_input'
    SPUR_INPUT_DIR_PATH = os.path.join(CWD, SPUR_INPUT_DIR)
    os.mkdir(SPUR_INPUT_DIR_PATH) if not os.path.exists(SPUR_INPUT_DIR_PATH) else None

    SPUR_INPUT_FILE = 'z3_problem.cnf'
    SPUR_INPUT_FILEPATH = f'{SPUR_INPUT_DIR}/{SPUR_INPUT_FILE}'
    (num_blasted_vars, variables_number) = save_dimacs(z3_problem,
                                                       SPUR_INPUT_FILEPATH)

    # spur sampling \o/
    execute_spur(SPUR_INPUT_FILEPATH,
                 num_samples=num_samples,
                 timeout=timeout)

    # parsing spur samples
    samples = parse_spur_samples(SPUR_INPUT_DIR, SPUR_INPUT_FILE,
                                 num_samples, num_blasted_vars)

    # map spur samples to the corresponding Z3 variable
    map_variable_values = map_spur_samples_to_z3_vars(variables_number,
                                                      num_blasted_vars,
                                                      samples)

    # reverse bit-blasting
    solver_samples = reverse_bit_blasting_simp(map_variable_values,
                                               num_samples,
                                               num_vars,
                                               num_bits)

    return solver_samples


def get_samples_sat_cmsgen_problem(z3_problem: Goal,
                                   num_vars: int, # number of varibles unblasted
                                   num_bits: int, # number of bits of BitVectors
                                                  # (assumption: all the same)
                                   num_samples: int = 10000,
                                   sanity_check_problem: bool = True,
                                   sanity_check_samples: bool = False,
                                   timeout: int = 1800,  # seconds
                                   print_z3_model: bool = False):

    if sanity_check_problem and __check_goal(z3_problem) == unsat:
        raise RuntimeError('The problem you input is UNSAT')

    if print_z3_model:
        print(z3_problem)

    CWD = os.getcwd()

    CMSGEN_INPUT_DIR = 'cmsgen_input'
    CMSGEN_INPUT_DIR_PATH = os.path.join(CWD, CMSGEN_INPUT_DIR)
    os.mkdir(CMSGEN_INPUT_DIR_PATH) if not os.path.exists(CMSGEN_INPUT_DIR_PATH) else None

    CMSGEN_INPUT_FILE = 'z3_problem.cnf'
    CMSGEN_INPUT_FILEPATH = f'{CMSGEN_INPUT_DIR}/{CMSGEN_INPUT_FILE}'

    CMSGEN_OUTPUT_FILE = 'cmsgen_samples.out'
    CMSGEN_OUTPUT_FILEPATH = f'{CMSGEN_INPUT_DIR}/{CMSGEN_OUTPUT_FILE}'

    (num_blasted_vars, variables_number) = save_dimacs(z3_problem,
                                                       CMSGEN_INPUT_FILEPATH)

    # cmsgen sampling \o/
    execute_cmsgen(CMSGEN_INPUT_FILEPATH,
                   CMSGEN_OUTPUT_FILEPATH,
                   num_samples=num_samples,
                   timeout=timeout)

    # parsing cmsgen samples
    samples = parse_cmsgen_samples(CMSGEN_INPUT_DIR, CMSGEN_OUTPUT_FILE,
                                   num_samples, num_blasted_vars)

    # map spur samples to the corresponding Z3 variable
    map_variable_values = map_spur_samples_to_z3_vars(variables_number,
                                                      num_blasted_vars,
                                                      samples)

    # reverse bit-blasting
    solver_samples = reverse_bit_blasting_simp(map_variable_values,
                                               num_samples,
                                               num_vars,
                                               num_bits)

    return solver_samples

# -------------------------------------------------------------------------------------#

def execute_unigen(input_filepath: str,
                   output_filepath: str,
                   num_samples: int = 10000,
                   timeout: int = 1800
                   ) -> None:

    """Executes unigen on the specified input file
    `input_filepath`. By default, it generates 10000 samples. The
    samples are added to the file specified in `output_filepath`.

    The function assumes that the spur executable is accessible
    by calling `cmsgen`.

    """
    call(['unigen',                         # - cmsgen command
                                            #   (hardcoded, it assumes
                                            #   accessible for this
                                            #   user)
          '--samples', str(num_samples),    # - number of samples
          '--sampleout', output_filepath,  # - output file path
          input_filepath],                  # - input file path
         timeout=timeout)  # timeout in seconds


def parse_unigen_samples(input_dir: str,
                         input_file: str,
                         num_samples: int,
                         num_variables: int) -> list[list[bool]]:
    unigen_samples_filepath = f'{input_dir}/{input_file}'
    samples = []
    with open(unigen_samples_filepath, 'r') as file:
        for line in file:
            sample = [int(int(l) >= 0) for l in line.split(' ')][:-1]
            samples.append(sample)
    samples_numpy = np.array(samples, dtype=np.int_)
    if not ((num_samples, num_variables) == samples_numpy.shape):
        if samples_numpy.shape[1] != num_variables:
            print(samples_numpy.shape[1])
            print(num_variables)
            time.sleep(2)
            raise RuntimeError("Number of variables mismatch")

        if samples_numpy.shape[0] < num_samples:
            raise RuntimeError("UniGen returned fewer samples than requested")

        # If more samples than requested then we truncate (Unigen is weird)
        samples_numpy = samples_numpy[:num_samples]
                ###### raise RuntimeError(f'The number of samples or number of variables do not match.\n \
                ###### unigen generated {samples_numpy.shape[0]} samples on {samples_numpy.shape[1]} variables, but you specified {num_samples} samples and {num_variables} variables')
    return samples_numpy


def get_samples_sat_unigen_problem(z3_problem: Goal,
                                   num_vars: int, # number of varibles unblasted
                                   num_bits: int, # number of bits of BitVectors
                                                  # (assumption: all the same)
                                   num_samples: int = 10000,
                                   sanity_check_problem: bool = True,
                                   sanity_check_samples: bool = False,
                                   timeout: int = 1800,  # seconds
                                   print_z3_model: bool = False):

    if sanity_check_problem and __check_goal(z3_problem) == unsat:
        raise RuntimeError('The problem you input is UNSAT')

    if print_z3_model:
        print(z3_problem)

    CWD = os.getcwd()

    UNIGEN_INPUT_DIR = 'unigen_input'
    UNIGEN_INPUT_DIR_PATH = os.path.join(CWD, UNIGEN_INPUT_DIR)
    os.mkdir(UNIGEN_INPUT_DIR_PATH) if not os.path.exists(UNIGEN_INPUT_DIR_PATH) else None

    UNIGEN_INPUT_FILE = 'z3_problem.cnf'
    UNIGEN_INPUT_FILEPATH = f'{UNIGEN_INPUT_DIR}/{UNIGEN_INPUT_FILE}'

    UNIGEN_OUTPUT_FILE = 'unigen_samples.out'
    UNIGEN_OUTPUT_FILEPATH = f'{UNIGEN_INPUT_DIR}/{UNIGEN_OUTPUT_FILE}'

    (num_blasted_vars, variables_number) = save_dimacs(z3_problem,
                                                       UNIGEN_INPUT_FILEPATH)

    # UNIGEN sampling \o/
    print("Executing Unigen sampler")
    execute_unigen(UNIGEN_INPUT_FILEPATH,
                   UNIGEN_OUTPUT_FILEPATH,
                   num_samples=num_samples,
                   timeout=timeout)

    # parsing UNIGEN samples
    print("Parsing unigen samples")
    samples = parse_unigen_samples(UNIGEN_INPUT_DIR, UNIGEN_OUTPUT_FILE,
                                   num_samples, num_blasted_vars)
    print(samples)
    # map spur samples to the corresponding Z3 variable
    map_variable_values = map_spur_samples_to_z3_vars(variables_number,
                                                      num_blasted_vars,
                                                      samples)

    # reverse bit-blasting
    solver_samples = reverse_bit_blasting_simp(map_variable_values,
                                               num_samples,
                                               num_vars,
                                               num_bits)
    print(solver_samples)
    # TEST_FILEPATH = F'{UNIGEN_INPUT_DIR}/{'unigen_test.pkl'}'
    # with open(TEST_FILEPATH, 'wb') as file:
    #     pickle.dump(solver_samples, file)

    # with open(TEST_FILEPATH, "wb") as f:
    #     for sample in solver_samples:
    #         f.write(" ".join(map(str, sample)) + "\n")
    return solver_samples
    

## *********************************************************************************************** ##
def get_samples_sat_pyunigen_problem(
    z3_problem: Goal,
    num_vars: int,
    num_bits: int,
    num_samples: int = 10000,
    sanity_check_problem: bool = False,
    timeout: int = 1800,
    print_z3_model: bool = False,
    time_tracking: bool = False,
):
    if print_z3_model:
        print(z3_problem)

    compiled = compile_base_problem(
        z3_problem=z3_problem,
        num_vars=num_vars,
        num_bits=num_bits,
        sanity_check_problem=sanity_check_problem,
    )

    solver_samples, elapsed = sample_cached_problem_pyunigen(
        compiled_problem=compiled,
        extra_clauses=[],
        num_samples=num_samples,
    )
    if time_tracking:
        return [solver_samples, [elapsed]]

    return solver_samples


##################################################################################################
#  ██╗ ███╗   ██╗ ██████╗██████╗ ███████╗███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ ██╗       #
#  ██║ ████╗  ██║██╔════╝██╔══██╗██╔════╝████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██║       #
#  ██║ ██╔██╗ ██║██║     ██████╔╝█████╗  ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║██║       #
#  ██║ ██║╚██╗██║██║     ██╔══██╗██╔══╝  ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║██║       #
#  ██║ ██║ ╚████║╚██████╗██║  ██║███████╗██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║███████╗  #
#  ╚═╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝  #
##################################################################################################

# =========================================================
# PyUniGen helpers
# =========================================================

def permute_problem(clauses, rng, sampling_set):
    old_vars = sampling_set[:]
    new_vars = sampling_set[:]
    rng.shuffle(new_vars)

    mp = dict(zip(old_vars, new_vars))

    permuted_clauses = [
        [mp[abs(lit)] if lit > 0 else -mp[abs(lit)] for lit in clause]
        for clause in clauses
    ]
    rng.shuffle(permuted_clauses)

    permuted_sampling_set = [mp[v] for v in sampling_set]
    permuted_decode = {mp[v]: v for v in sampling_set}

    return permuted_clauses, permuted_sampling_set, permuted_decode

def sample_raw_pyunigen(clauses, sampling_set, num_samples=1, rng=None):
    rng = rng or random.Random()
    permuted_clauses, permuted_sampling_set, permuted_decode = permute_problem(
        clauses, rng, sampling_set
    )

    sampler = uni.Sampler()
    for clause in permuted_clauses:
        sampler.add_clause(clause)

    errors = []
    for mode in ("kw_num_sampling", "pos_num_sampling", "pos_num"):
        try:
            if mode == "kw_num_sampling":
                result = sampler.sample(num=num_samples, sampling_set=permuted_sampling_set)
            elif mode == "pos_num_sampling":
                result = sampler.sample(num_samples, permuted_sampling_set)
            else:
                result = sampler.sample(num_samples)
            break
        except Exception as e:
            errors.append(f"{mode}: {repr(e)}")
    else:
        raise RuntimeError("Could not call pyunigen successfully.\n" + "\n".join(errors))

    samples = result[2] if isinstance(result, tuple) and len(result) == 3 else result
    if not samples:
        raise RuntimeError("pyunigen returned no sample.")
    if len(samples) < num_samples:
        raise RuntimeError(
            f"pyunigen returned fewer samples than requested: {len(samples)} < {num_samples}"
        )

    restored = []
    for sample in samples[:num_samples]:
        lits = [
            permuted_decode[abs(lit)] if lit > 0 else -permuted_decode[abs(lit)]
            for lit in sample
        ]
        lits.sort(key=lambda x: abs(x))
        restored.append(lits)

    return restored

# =========================================================
# PyCMSGen helpers
# =========================================================

def build_base_cmsgen_solver(base_clauses, rng=None):
    rng = rng or random.Random()
    solver = cms.Solver(seed=rng.randrange(1, 2**31))

    max_var = 0
    for clause in base_clauses:
        solver.add_clause(clause)
        for lit in clause:
            max_var = max(max_var, abs(lit))

    return solver, max_var

def add_guarded_window_clauses(solver, window_clauses, next_free_var):
    act = next_free_var + 1
    for clause in window_clauses:
        solver.add_clause([-act] + clause)
    return act, act

def call_sample_pycmsgen_assumptions(solver, sampling_set, num_samples=1, assumptions=None):
    assumptions = assumptions or [] #make current assumsions/window true, is autmatically not true at next call
    out = []

    for _ in range(num_samples):
        sat_result, model = solver.solve(assumptions=assumptions)
        if sat_result is not True:
            raise RuntimeError(f"CMSGen did not return SAT: {sat_result}")
        out.append([v if model[v] else -v for v in sampling_set])

    return out

# =========================================================
# General helpers
# =========================================================

def find_valid_start_fast_z3(z3_problem, num_vars, num_bits):
    s = Solver()
    s.add(z3_problem)

    if s.check() != sat:
        raise RuntimeError("Z3 could not find a satisfying start model.")

    m = s.model()
    return {
        f"x{i}": m.eval(BitVec(f"x{i}", num_bits), model_completion=True).as_long()
        for i in range(num_vars)
    }

def extract_current_values_from_solver_sample(sample0, num_vars: int):
    if all(i in sample0 for i in range(num_vars)):
        return [int(sample0[i]) for i in range(num_vars)]

    if all(f"x{i}" in sample0 for i in range(num_vars)):
        return [int(sample0[f"x{i}"]) for i in range(num_vars)]

    vals = list(sample0.values())
    if len(vals) < num_vars:
        raise RuntimeError(
            f"Could not extract {num_vars} variable values from sample: {sample0}"
        )
    return [int(v) for v in vals[:num_vars]]

# =========================================================
# CNF / bit-map / decode helpers
# =========================================================

def extract_bitvec_dimacs_map(variables_number, num_vars: int, num_bits: int):
    bit_digits = len(str(num_bits - 1))
    bit_map = {i: [None] * num_bits for i in range(num_vars)}

    for dimacs_id, z3var in variables_number.items():
        name = str(z3var)
        if not name.startswith("x"):
            continue

        digits = name[1:]
        if len(digits) <= bit_digits:
            continue

        var_part = digits[:-bit_digits]
        bit_part = digits[-bit_digits:]
        var_idx = int(var_part)
        bit_idx = int(bit_part)

        if 0 <= var_idx < num_vars and 0 <= bit_idx < num_bits:
            if bit_map[var_idx][bit_idx] is not None:
                raise RuntimeError(
                    f"Duplicate DIMACS mapping for x{var_idx} bit {bit_idx}: "
                    f"{bit_map[var_idx][bit_idx]} and {dimacs_id}"
                )
            bit_map[var_idx][bit_idx] = int(dimacs_id)

    missing = [
        (i, b)
        for i in range(num_vars)
        for b in range(num_bits)
        if bit_map[i][b] is None
    ]
    if missing:
        raise RuntimeError(f"Missing x-bit DIMACS ids in variables_number: {missing}")

    return bit_map

def reverse_bit_blasting_simp(variable_values, num_samples: int, num_vars: int, num_bits: int):
    bit_digits = len(str(num_bits - 1))

    def from_bin_to_dec(i, s):
        total = 0
        for j in range(num_bits):
            total += (2 ** j) * variable_values[f"x{i}{j:0{bit_digits}d}"][s]
        return total

    return [
        {f"x{i}": from_bin_to_dec(i, s) for i in range(num_vars)}
        for s in range(num_samples)
    ]

def decode_raw_samples_to_solver_samples(compiled_problem, raw_samples, num_samples):
    arr = np.array([[int(x >= 0) for x in sample] for sample in raw_samples], dtype=np.int_)

    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected sample shape: {arr.shape}")
    if arr.shape[1] != compiled_problem["num_blasted_vars"]:
        raise RuntimeError(
            f"Number of variables mismatch: got {arr.shape[1]}, "
            f"expected {compiled_problem['num_blasted_vars']}"
        )
    if arr.shape[0] < num_samples:
        raise RuntimeError(
            f"Returned fewer samples than requested: {arr.shape[0]} < {num_samples}"
        )

    arr = arr[:num_samples]
    variable_values = {
        name: arr[:, i]
        for i, name in enumerate(compiled_problem["precomputed_blasted_names"])
    }

    return reverse_bit_blasting_simp(
        variable_values,
        num_samples=num_samples,
        num_vars=compiled_problem["num_vars"],
        num_bits=compiled_problem["num_bits"],
    )

def compile_base_problem(z3_problem: Goal,
                         num_vars: int,
                         num_bits: int,
                         sanity_check_problem: bool = True):
    if num_bits <= 0:
        raise RuntimeError("num_bits must be positive")
    if sanity_check_problem and __check_goal(z3_problem) == unsat:
        raise RuntimeError("The problem you input is UNSAT")

    dimacs_format, num_blasted_vars, variables_number = convert_to_cnf_and_dimacs_simp(z3_problem)
    clauses = [[int(x) for x in row[:-1]] for row in dimacs_format[1:]]
    sampling_set = sorted({abs(lit) for clause in clauses for lit in clause})

    return {
        "num_blasted_vars": num_blasted_vars,
        "bit_map": extract_bitvec_dimacs_map(variables_number, num_vars, num_bits),
        "base_clauses": clauses,
        "num_vars": num_vars,
        "num_bits": num_bits,
        "precomputed_blasted_names": [str(variables_number[i + 1]) for i in range(num_blasted_vars)],
        "sampling_set": sampling_set,
    }

# =========================================================
# Window encoding helpers
# =========================================================


def cnf_encode_ule_constant(bit_vars, upper: int, width: int, bit_vars_are_lsb_first: bool = True):
    # ULE = unsigned less-than-or-equal
    if len(bit_vars) != width:
        raise RuntimeError("bit_vars width mismatch in cnf_encode_ule_constant")
    if upper < 0:
        return [[]]

    max_val = (1 << width) - 1
    if upper >= max_val:
        return []

    msb_vars = bit_vars[::-1] if bit_vars_are_lsb_first else bit_vars[:]
    upper_bits = [((upper >> i) & 1) for i in range(width)][::-1]

    clauses = []
    for i in range(width):
        if upper_bits[i] == 0:
            clause = []
            for j in range(i):
                clause.append(msb_vars[j] if upper_bits[j] == 0 else -msb_vars[j])
            clause.append(-msb_vars[i])
            clauses.append(clause)

    return clauses

def cnf_encode_uge_constant(bit_vars, lower: int, width: int, bit_vars_are_lsb_first: bool = True):
    # UGE = unsigned greater-than-or-equal
    if len(bit_vars) != width:
        raise RuntimeError("bit_vars width mismatch in cnf_encode_uge_constant")

    max_val = (1 << width) - 1
    if lower <= 0:
        return []
    if lower > max_val:
        return [[]]

    msb_vars = bit_vars[::-1] if bit_vars_are_lsb_first else bit_vars[:]
    lower_bits = [((lower >> i) & 1) for i in range(width)][::-1]

    clauses = []
    for i in range(width):
        if lower_bits[i] == 1:
            clause = []
            for j in range(i):
                clause.append(msb_vars[j] if lower_bits[j] == 0 else -msb_vars[j])
            clause.append(msb_vars[i])
            clauses.append(clause)

    return clauses

def encode_window_clauses_from_values(current_values,
                                      bit_map,
                                      num_vars: int,
                                      num_bits: int,
                                      D: int,
                                      bit_vars_are_lsb_first: bool = True):
    if len(current_values) != num_vars:
        raise RuntimeError(
            f"current_values length mismatch: got {len(current_values)}, expected {num_vars}"
        )

    clauses = []
    max_val = (1 << num_bits) - 1

    for i in range(num_vars):
        val = int(current_values[i])
        lower = max(0, val - D)
        upper = min(max_val, val + D)
        xi_bits = bit_map[i]

        clauses.extend(cnf_encode_uge_constant(
            bit_vars=xi_bits,
            lower=lower,
            width=num_bits,
            bit_vars_are_lsb_first=bit_vars_are_lsb_first,
        ))
        clauses.extend(cnf_encode_ule_constant(
            bit_vars=xi_bits,
            upper=upper,
            width=num_bits,
            bit_vars_are_lsb_first=bit_vars_are_lsb_first,
        ))

    return clauses

# =========================================================
# Backend-specific cached samplers
# =========================================================

def sample_cached_problem_pyunigen(compiled_problem, extra_clauses, num_samples, rng=None, time_tracking=False):
    combined_clauses = compiled_problem["base_clauses"] + (extra_clauses or [])
    if any(len(c) == 0 for c in combined_clauses):
        raise RuntimeError("Augmented CNF is immediately UNSAT (contains empty clause).")

    t0 = time.perf_counter()
    raw_samples = sample_raw_pyunigen(
        clauses=combined_clauses,
        sampling_set=compiled_problem["sampling_set"],
        num_samples=num_samples,
        rng=rng,
    )
    elapsed = time.perf_counter() - t0

    return decode_raw_samples_to_solver_samples(compiled_problem, raw_samples, num_samples), elapsed

def sample_cached_problem_pycmsgen(compiled_problem, solver, next_free_var, extra_clauses, num_samples):
    extra_clauses = extra_clauses or []
    if any(len(c) == 0 for c in extra_clauses):
        raise RuntimeError("Augmented CNF is immediately UNSAT (contains empty clause).")

    t0 = time.perf_counter()

    if extra_clauses:
        act_lit, next_free_var = add_guarded_window_clauses(solver, extra_clauses, next_free_var)
        assumptions = [act_lit]
    else:
        assumptions = []

    raw_samples = call_sample_pycmsgen_assumptions(
        solver,
        compiled_problem["sampling_set"],
        num_samples=num_samples,
        assumptions=assumptions,
    )
    elapsed = time.perf_counter() - t0

    return (
        decode_raw_samples_to_solver_samples(compiled_problem, raw_samples, num_samples),
        elapsed,
        next_free_var,
    )

def get_samples_sat_pycmsgen_problem(z3_problem: Goal,
                                     num_vars: int,
                                     num_bits: int,
                                     num_samples: int = 10000,
                                     sanity_check_problem: bool = False,
                                     timeout: int = 1800,
                                     print_z3_model: bool = False):
    if print_z3_model:
        print(z3_problem)

    compiled = compile_base_problem(z3_problem, num_vars, num_bits, sanity_check_problem)
    solver, next_free_var = build_base_cmsgen_solver(compiled["base_clauses"], rng=random.Random())
    samples, _, _ = sample_cached_problem_pycmsgen(
        compiled, solver, next_free_var, [], num_samples
    )
    return samples

# =========================================================
# Incremental pipelines
# =========================================================

def incremental_pyunigen_pipeline(
    compiled,
    chosen_sample,
    trace,
    elapsed_time_per_sample,
    start_idx,
    D,
    num_samples,
    parallel_samples,
    bit_vars_are_lsb_first,
    rng,
    restart_every=None,
    print_progress=True,
):
    for i in range(start_idx, num_samples):
        if i % 10 == 0 and print_progress:
            print(f"sample {i}")

        extra_clauses = [] if i == 0 else encode_window_clauses_from_values(
            current_values=extract_current_values_from_solver_sample(
                chosen_sample, compiled["num_vars"]
            ),
            bit_map=compiled["bit_map"],
            num_vars=compiled["num_vars"],
            num_bits=compiled["num_bits"],
            D=D,
            bit_vars_are_lsb_first=bit_vars_are_lsb_first,
        )

        solver_samples, elapsed = sample_cached_problem_pyunigen(
            compiled_problem=compiled,
            extra_clauses=extra_clauses,
            num_samples=parallel_samples,
            rng=rng,
        )
        chosen_sample = solver_samples[rng.randrange(len(solver_samples))]
        trace.append(chosen_sample)
        elapsed_time_per_sample.append(elapsed)

    return {"trace": trace, "elapsed_time_per_sample": elapsed_time_per_sample}

def incremental_cmsgen_pipeline(
    compiled,
    chosen_sample,
    trace,
    elapsed_time_per_sample,
    start_idx,
    D,
    num_samples,
    parallel_samples,
    bit_vars_are_lsb_first,
    rng,
    restart_every=100,
    print_progress=True,
):
    solver, next_free_var = build_base_cmsgen_solver(compiled["base_clauses"], rng=rng)

    for i in range(start_idx, num_samples):
        if print_progress:
            print(f"sample {i}")

        if i > start_idx and restart_every and i % restart_every == 0:
            if print_progress:
                print(f"Restarting CMSGen solver at sample {i}")
            solver, next_free_var = build_base_cmsgen_solver(compiled["base_clauses"], rng=rng)

        extra_clauses = [] if i == 0 else encode_window_clauses_from_values(
            current_values=extract_current_values_from_solver_sample(
                chosen_sample, compiled["num_vars"]
            ),
            bit_map=compiled["bit_map"],
            num_vars=compiled["num_vars"],
            num_bits=compiled["num_bits"],
            D=D,
            bit_vars_are_lsb_first=bit_vars_are_lsb_first,
        )

        solver_samples, elapsed, next_free_var = sample_cached_problem_pycmsgen(
            compiled_problem=compiled,
            solver=solver,
            next_free_var=next_free_var,
            extra_clauses=extra_clauses,
            num_samples=parallel_samples,
        )
        chosen_sample = solver_samples[rng.randrange(len(solver_samples))]
        trace.append(chosen_sample)
        elapsed_time_per_sample.append(elapsed)

    return {"trace": trace, "elapsed_time_per_sample": elapsed_time_per_sample}

# =========================================================
# Sleek dispatch
# =========================================================

def get_conditional_incremental_samples_sat_problem_cached_dispatch(
    backend: str,
    z3_problem: Goal,
    num_vars: int,
    num_bits: int,
    D: int = 1,
    num_samples: int = 10000,
    sanity_check_problem: bool = True,
    parallel_samples: int = 1,
    sanity_check_samples: bool = False,
    timeout: int = 1800,
    print_z3_model: bool = False,
    bit_vars_are_lsb_first: bool = True,
    time_tracking: bool = False,
    restart_every: int = 100,
    fast_start: bool = False,
    print_progress: bool = True,
):
    if print_z3_model:
        print(z3_problem)

    compiled = compile_base_problem(
        z3_problem=z3_problem,
        num_vars=num_vars,
        num_bits=num_bits,
        sanity_check_problem=sanity_check_problem,
    )

    rng = random.Random()
    trace = []
    elapsed_time_per_sample = []
    chosen_sample = None
    start_idx = 0

    if fast_start:
        print("Finding initial sample with Z3...")
        t0 = time.perf_counter()
        chosen_sample = find_valid_start_fast_z3(z3_problem, num_vars, num_bits)
        trace.append(chosen_sample)
        elapsed_time_per_sample.append(time.perf_counter() - t0)
        print(f"Z3 found initial sample in {time.perf_counter() - t0:.2f} seconds: {chosen_sample}, starting incremental sampling from there.")
        start_idx = 1

    common_kwargs = dict(
        compiled=compiled,
        chosen_sample=chosen_sample,
        trace=trace,
        elapsed_time_per_sample=elapsed_time_per_sample,
        start_idx=start_idx,
        D=D,
        num_samples=num_samples,
        parallel_samples=parallel_samples,
        bit_vars_are_lsb_first=bit_vars_are_lsb_first,
        rng=rng,
        restart_every=restart_every,
        print_progress=print_progress,
    )

    if backend == "pyunigen":
        result = incremental_pyunigen_pipeline(**common_kwargs)
    elif backend == "pycmsgen":
        result = incremental_cmsgen_pipeline(**common_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return [result["trace"], result["elapsed_time_per_sample"]] if time_tracking else result["trace"]