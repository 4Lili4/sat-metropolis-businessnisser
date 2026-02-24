from subprocess import call
from z3 import Goal, BitVecSort, Bool, And, BVAddNoOverflow, BVMulNoOverflow, BVSubNoUnderflow, Then, simplify, is_app_of, Z3_OP_NOT, Not, BoolRef, solve, unsat, sat, Solver
import math
import random
import numpy as np
import re
import os
from warnings import warn

from sat_metropolis import utils

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

def add_bool_vars_to_goal(g: Goal, var_list: [BitVecSort]):
    """This function creates a Bool variable for each bit in the
    BitVectors in `var_list`.

    Note that the name of the variables is fixed `x_ji` where j is the
    index of the BitVector in `var_list` and i is the index of bit in
    the corresponding BitVector.
    """
    # NOTE: We could try to obtain the variable name from `var_list`
    bitmap = {}
    num_vars = len(var_list)
    for j in range(num_vars):
        num_bits = var_list[j].size()  # returns #bits in `var_list[j]`
        for i in range(num_bits):
            bitmap[var_list[j], i] = Bool(f'x{j}{i}')
            mask = BitVecSort(num_bits).cast(math.pow(2, i))
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
        dict[int, BoolRef]):
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
        x = f'x{i}'
        total = 0
        for j in range(num_bits):
            total += 2**j * map_variable_values[f'{x}{j}'][s]
        return total

    solver_samples = [{f'x{i}': from_bin_to_dec(i, s, num_bits, variable_values)
                       for i in range(num_vars)} for s in range(num_samples)]

    return solver_samples


def reverse_bit_blasting(variable_values: dict[str, list[bool]],
                         num_samples: int,
                         num_vars: int,
                         num_bits: int) -> list[dict[str, int]]:

    # DEPRECATED function
    warn('This method is deprecated (and buggy). Please use the newer \
    `reverse_bit_blasting_simp`. It has the same signature',
         DeprecationWarning, stacklevel=2)

    map_var_name_samples = {}
    for i in range(num_vars):
        var_name = f'x{i}'
        map_var_name_samples[var_name] = np.zeros(num_samples, dtype=np.int_)
        for j in range(num_bits):
            bit_j_value = 2**j * variable_values[f'{var_name}{j}']
            map_var_name_samples[var_name] += bit_j_value

    # transform into format for the mcmc Metropolis-Hastings function
    # NOTE: this step could be avoided if we modify slightly the
    # mcmc.sample_mh_trace fucntion. In fact, mcmc.sample_mh_trace
    # maps back the dictionary into the shape in
    # `map_var_name_samples`
    var_names = list(map_var_name_samples.keys())
    # num_samples = len(map_var_name_samples[var_names[0]])
    solver_samples = []
    for i in range(num_samples):
        sample = {v: map_var_name_samples[v][i] for v in var_names}
        solver_samples.append(sample)
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

    """Executes cmsgen on the specified input file
    `input_filepath`. By default, it generates 10000 samples. The
    samples are added to the file specified in `output_filepath`.

    The function assumes that the spur executable is accessible
    by calling `unigen`.

    """
    call(['unigen',                         # - unigen command
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

