from subprocess import call
from z3 import Solver
import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", ".."))
sys.path.append(project_root)

def generate_smt2(solver: Solver) -> str:
    """
    The parameter `solver` contains the SMT problem composed of the variables
    and constraints of the problem to solve.
    """
    return solver.to_smt2()


# When used as a type, None is equivalent to type(None)
def save_smt2(solver: Solver, filepath: str) -> None:
    """Takes as input a Solver object from Z3, and stores it in a file
    `filepath`.

    The file produced by this function is intended to be used as the
    input file for `execute_megasampler`.
    """
    path = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(path) and len(path) > 0:
        raise RuntimeError(f'Directory {path} not found')

    with open(filepath, 'w') as file:
        file.write(generate_smt2(solver))


def execute_megasampler(input_filepath: str,
                        num_samples: int = 10000,
                        algo: str = 'MeGA',
                        time_limit_sec: int = 1800,
                        output_dir: str = 'Solutions') -> None:

    """Executes megasampler on the specified input file
    `input_filepath`. By default, it generates 10000 samples, uses the
    `MeGA` algorithm for sampling, has a time limit of 1800 seconds,
    and the output directory is `Solutions`.

    The function assumes that the megasampler executable is accessible
    by calling `megasampler`.
    """
    call(['megasampler',              # - megasampler command (hardcoded, it
                                      #   assumes accessible for this user)
          '-n', str(num_samples),     # - number of samples
          '-a', algo,                 # - Algorithm {MeGA, MeGAb, SMT, z3}
          '-t', str(time_limit_sec),  # - time limit (seconds)
          input_filepath,             # - input file
          '-o', output_dir])          # - output dir


def __get_var_sample(elem: str) -> (str, int):
    # NOTE: The casting to `int` below may need to be a function
    #       parameter, as it might depend on the problem
    """Parse the (string) samples produced by megasampler. It returns a
    pair (str,int) with the variable name as the first element of the
    pair and an integer with the sampled value.

    NOTE: We might want to add as a parameter the type of the
    sample. In some problems, we could have samples of type float or
    other types.

    """
    var_sample = elem.split(':')
    if len(var_sample) > 2:
        return (var_sample[1].strip(), int(var_sample[2].strip()))
    elif len(var_sample) > 1:
        return (var_sample[0].strip(), int(var_sample[1].strip()))
    else:
        return None


def parse_megasamples(filepath: str) -> [dict[str, int]]:
    """Takes as input the file with megasampler sapmles (I refer to them
    as megasamples). It returns the a array of dictionaries. Each
    element of the array corresponds to a sample. Each dictionary maps
    each variable to its value in the sample.
    """
    # NOTE: This is useful for sampling initial states for chains as well
    res_map = []
    with open(filepath, 'r') as f:
        for line in f:
            split_line = line.split(';')
            a = [i for i in [__get_var_sample(elem) for elem in split_line]
                 if i is not None]
            b = {i: j for (i, j) in a}
            res_map.append(b)
    return res_map


def get_samples_smt_problem(z3_problem: Solver,
                            num_samples: int = 10000,
                            algo: str = 'MeGA',
                            timeout: int = 1800  # seconds
                            ) -> [dict[str, int]]:
    # get current working directory for defining the necessary paths below
    CWD = os.getcwd()

    # define megasampler input directory and create it if it does not exist
    MEGASAMPLER_INPUT_DIR = 'megasampler_input'
    MEGASAMPLER_INPUT_DIR_PATH = os.path.join(CWD, MEGASAMPLER_INPUT_DIR)
    os.mkdir(MEGASAMPLER_INPUT_DIR_PATH) if not os.path.exists(MEGASAMPLER_INPUT_DIR_PATH) else None

    # define megasampler input file name (used as output of the smt2 format of the problem)
    megasampler_input_file = 'z3_problem.smt2'
    megasampler_input_filepath = f'{MEGASAMPLER_INPUT_DIR}/{megasampler_input_file}'
    save_smt2(solver=z3_problem, filepath=megasampler_input_filepath)

    MEGASAMPLER_OUTPUT_DIR = 'megasampler_output'
    MEGASAMPLER_OUTPUT_DIR_PATH = os.path.join(CWD, MEGASAMPLER_OUTPUT_DIR)

    # create directory if it does not exist
    os.mkdir(MEGASAMPLER_OUTPUT_DIR_PATH) if not os.path.exists(MEGASAMPLER_OUTPUT_DIR_PATH) else None

    # execute megasampler
    execute_megasampler(num_samples=num_samples,
                        input_filepath=megasampler_input_filepath,
                        algo=algo,
                        time_limit_sec=timeout,
                        output_dir=MEGASAMPLER_OUTPUT_DIR)

    # megasampler output file with samples (name automatically set by megasampler)
    file_samples = f'{MEGASAMPLER_OUTPUT_DIR}/{megasampler_input_file}.samples'

    # parsing the samples
    samples = parse_megasamples(file_samples)
    return samples
