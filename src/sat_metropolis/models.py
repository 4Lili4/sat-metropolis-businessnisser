"""
This module contains functions to quickly obtain the models we benchmark in experiments.ipynb
"""

from z3 import Goal, BitVec, ULE, Sum, Int, Solver
import numpy as np
import pandas as pd
import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", ".."))
sys.path.append(project_root)

# from sat_metropolis import sat
import src.sat_metropolis.sat as sat

# SAT Models

def get_triangle_sat(num_bits: int = 4,
                     num_vars: int = 3) -> tuple[Goal, int, int]:
    var_list = [BitVec(f'x{i}', num_bits) for i in range(num_vars)]
    x = var_list
    g = Goal()
    sat.add_bool_vars_to_goal(g, var_list)
    g.add(sat.addition_does_not_overflow([x[0], x[1], x[2]]))  # important
    g.add(ULE(0, x[0]))
    g.add(ULE(x[0], 3))
    g.add(ULE(0, x[1]))
    g.add(ULE(x[1], 3))
    g.add(ULE(0, x[2]))
    g.add(ULE(x[2], 6))
    g.add(x[0] + x[1] == x[2])

    return (g, num_bits, num_vars)


def get_db_cacm_sat(num_bits: int = 8,
                    num_vars: int = 5*2) -> tuple[Goal, int, int]:
    # num_bits = 8
    # num_vars = 5*2  # (age, sex)
    var_list = [BitVec(f'x{i}', num_bits) for i in range(num_vars)]  # 0-4 age vars, 5-9 sex vars
    x = var_list
    g = Goal()
    sat.add_bool_vars_to_goal(g, var_list)
    g.add(sat.addition_does_not_overflow([x[0], x[1], x[2], x[3], x[4]]))
    g.add(sat.addition_does_not_overflow([x[5], x[6], x[7], x[8], x[9]]))
    for i in range(5):
        g.add(sat.multi_does_not_overflow([x[i],x[i+5]])) # this is in case of multiplication

    for i in range(5, 10):
        g.add(ULE(0, x[i]))
        g.add(ULE(x[i], 1))


    #  establish an order to set the median
    g.add(ULE(0, x[0]))
    g.add(ULE(x[0], x[1]))
    g.add(ULE(x[1], x[2]))
    g.add(ULE(x[2], x[3]))
    g.add(ULE(x[3], x[4]))
    g.add(ULE(x[4], 125))

    g.add(x[2] == 30)  # median
    g.add(x[0] + x[1] + x[2] + x[3] + x[4] == 38*5)  # average age
    g.add(x[5] + x[6] + x[7] + x[8] + x[9] == 3)  # only 3 males
    g.add(x[0]*x[5] + x[1]*x[6] + x[2]*x[7] + x[3]*x[8] + x[4]*x[9] == 44*3)  # average age of males

    return (g, num_bits, num_vars)


def get_nz_stats_sat(path_to_dataset: str = "south_head.csv",
                     num_bits: int = 5,
                     num_vars: int = 19*2) -> tuple[Goal, int, int]:
    # Assumption, all bit-vectors are of the same size
    # num_bits = 5
    # num_vars = 19*2
    var_list = [BitVec(f'x{i}', num_bits) for i in range(num_vars)]
    x = var_list
    g = Goal()
    sat.add_bool_vars_to_goal(g, var_list)

    # load data
    south_head = pd.read_csv(path_to_dataset)

    # Add contraints
    numpy_data = south_head.to_numpy()[:, 1:]

    for i in range(38):
        g.add(ULE(0,x[i]))

    for i in range(19):
        g.add(ULE(x[i], numpy_data[0, i]+2))
        if (numpy_data[0, i] > 0):
            g.add(ULE(numpy_data[0, i]-2, x[i]))

        g.add(ULE(x[i+19], numpy_data[1, i]+2))
        if (numpy_data[1, i] > 0):
            g.add(ULE(numpy_data[1, i]-2, x[i+19]))

        g.add(ULE(x[i+19] + x[i], numpy_data[2, i]+2))
        if (numpy_data[2, i] > 0):
            g.add(ULE(numpy_data[2, i]-2, x[i+19]+x[i]))

    return (g, num_bits, num_vars)


def __gen_conf_matrix(y: np.ndarray):
    """Simple function to generate the configuration matrix for
    variable number of roads
    """
    num_vers = y.size+1
    num_pathways = num_vers-1

    pathways = []
    for i in range(1, num_vers+1):
        for j in range(1, num_vers+1):
            if i < j:
                pathways += [(i, j)]

    print(pathways)

    return np.array([
        [1 if (p1 <= i and i < p2) else 0 for (p1, p2) in pathways]
        for i in range(1, num_pathways+1)
    ])

def __get_z3_model_from_conf_matrix_sat(num_bits: int,
                                        max_int_bv: int,
                                        Aprime: np.ndarray,  # matrix
                                        yprime: np.ndarray,  # vector
                                        ) -> tuple[Goal, int]:
    num_vars = Aprime.shape[1]

    x = [BitVec(f'x{i}', num_bits) for i in range(num_vars)]

    g = Goal()

    sat.add_bool_vars_to_goal(g, x)

    for i in range(len(yprime)):
        vars_ = [x[j] for j in range(num_vars) if Aprime[i][j] == 1]
        g.add(sat.addition_does_not_overflow(vars_))
        g.add(Sum(vars_) == yprime[i])

    for i in range(num_vars):
        g.add(ULE(0, x[i]))
        g.add(ULE(x[i], max_int_bv))  # adding also max value to avoid
                                      # overflows

    return (g, num_vars)


def get_roads_sat(num_bits: int,
                  max_int_bv: int,
                  num_con: int = 4,
                  y_reduction_factor: int = 11) -> tuple[Goal, int, int]:
    # data from problem
    y = np.array([1087, 1008, 1068, 1204, 1158, 1151, 1143])

    # generate data and configuration matrix according to input parameters
    yprime = np.array([int(i/y_reduction_factor) for i in y[:num_con]])
    Aprime = __gen_conf_matrix(yprime)

    (g, num_vars) = __get_z3_model_from_conf_matrix_sat(num_bits, max_int_bv,
                                                        Aprime, yprime)

    return (g, num_bits, num_vars)


def get_books_sat(bound: int = 9,
                  y_reduction_factor: int = 1,
                  num_bits: int = 6,
                  max_int_bv: int = int(86/6)
                  ) -> tuple[Goal, int, int]:
    y = np.array([3, 12, 1, 21, 86, 16, 2, 24, 5, 184, 822, 163,
                  40, 102, 13, 58, 253, 38, 18, 104, 40, 19, 147,
                  25, 52, 220, 35, 1, 3, 6, 8, 36, 5, 1, 14, 2, 1,
                  10, 13, 3, 43, 4])

    Aprime = A_BOOKS[:bound,:]
    yprime = np.array([int(i/y_reduction_factor) for i in y[:bound]])

    (g, num_vars) = __get_z3_model_from_conf_matrix_sat(num_bits, max_int_bv,
                                                        Aprime, yprime)

    return (g, num_bits, num_vars)


def get_haplotypes_sat(genotypes: np.ndarray,  # size num_genotypes x
                                               # bits_per_haplotype
                       ) -> tuple[Goal, int, int]:

    (num_genotypes, bits_per_haplotype) = genotypes.shape

    # Assumption, all bit-vectors are of the same size. Each
    # bit-vector represents a haplotype bit, the length is two because
    # when summing 1 + 1 we must be able to store 10.
    num_bits = 2
    # the last 2 is due to the fact that a genotype is a combination
    # of 2 haplotypes
    num_vars = bits_per_haplotype*num_genotypes*2
    x = [BitVec(f'x{i}', num_bits) for i in range(num_vars)]
    # x = var_list
    g = Goal()
    sat.add_bool_vars_to_goal(g, x)

    # Add contraints
    for i in range(num_vars):
        g.add(ULE(x[i], 1))  # Each BitVec is in {0,1}

    # iterate over all genotypes
    for i in range(num_genotypes):
        # iterate over all bits of the BitVec
        for j in range(bits_per_haplotype):
            g.add(x[i*10+j] + x[i*10+j+5] == genotypes[i][j])

    return (g, num_bits, num_vars)


# SMT Models

def get_triangle_smt(num_vars: int = 3) -> tuple[Solver, int]:
    x = [Int(f'x{i}') for i in range(num_vars)]
    s = Solver()  # get an instance of a Z3 solver

    # model constraints
    s.add(x[0] >= 0)
    s.add(x[1] >= 0)
    s.add(x[2] >= 0)
    s.add(x[0] <= 3)
    s.add(x[1] <= 3)
    s.add(x[2] <= 6)
    s.add(x[0] + x[1] == x[2])

    return (s, num_vars)


def get_db_cacm_smt(num_vars: int = 5*2) -> tuple[Solver, int]:
    x = [Int(f'x{i}') for i in range(num_vars)]  # (age, sex)
    s = Solver()  # get an instance of a Z3 solver

    # model constraints
    s.add(0 <= x[0])
    s.add(x[0] <= x[1])
    s.add(x[1] <= x[2])
    s.add(x[2] == 30)  # median
    s.add(x[2] <= x[3])
    s.add(x[3] <= x[4])
    s.add(x[4] <= 125)

    # average
    s.add(x[0] + x[1] + x[2] + x[3] + x[4] == 38*5)

    for i in range(5, 10):
        s.add(0 <= x[i])
        s.add(x[i] <= 1)

    # only 3 males
    s.add(x[5] + x[6] + x[7] + x[8] + x[9] == 3)
    # average age of males
    s.add(x[0]*x[5] + x[1]*x[6] + x[2]*x[7] + x[3]*x[8] + x[4]*x[9] == 44*3)

    return (s, num_vars)


def get_nz_stats_smt(path_to_dataset: str = "south_head.csv",
                     num_vars: int = 19*2) -> tuple[Solver, int]:
    x = [Int(f'x{i}') for i in range(num_vars)]

    south_head = pd.read_csv(path_to_dataset)
    numpy_data = south_head.to_numpy()[:, 1:]
    s = Solver()

    for i in range(38):
        s.add(x[i] >= 0)

    for i in range(19):
        s.add(x[i] <= numpy_data[0, i]+2)
        if (numpy_data[0, i] > 0):
            s.add(numpy_data[0, i]-2 <= x[i])

        s.add(x[i+19] <= numpy_data[1, i]+2)
        if (numpy_data[1, i] > 0):
            s.add(numpy_data[1, i]-2 <= x[i+19])

        s.add(x[i+19] + x[i] <= numpy_data[2, i]+2)
        if (numpy_data[2, i] > 0):
            s.add(numpy_data[2, i]-2 <= x[i+19]+x[i])

    return (s, num_vars)


def __get_z3_model_from_conf_matrix_smt(max_int: int,
                                        A: np.ndarray,  # matrix
                                        y: np.ndarray,  # vector
                                        ) -> tuple[Solver, int]:

    num_vars = A.shape[1]

    x = [Int(f'x{i}') for i in range(num_vars)]

    s = Solver()

    for i in range(num_vars):
        s.add(0 <= x[i])
        s.add(x[i] <= max_int)

    for i in range(len(y)):
        vars_ = [x[j] for j in range(num_vars) if A[i][j] == 1]  # alternative
        s.add(Sum(vars_) == y[i])

    return (s, num_vars)


def get_roads_smt(max_int: int,
                  num_con: int = 4,
                  y_reduction_factor: int = 11) -> tuple[Solver, int]:
    # data from problem
    y = np.array([1087, 1008, 1068, 1204, 1158, 1151, 1143])

    # generate data and configuration matrix according to input parameters
    yprime = np.array([int(i/y_reduction_factor) for i in y[:num_con]])
    Aprime = __gen_conf_matrix(yprime)

    (s, num_vars) = __get_z3_model_from_conf_matrix_smt(max_int,
                                                        Aprime, yprime)

    return (s, num_vars)


def get_books_smt(bound: int = 9,
                  y_reduction_factor: int = 1,
                  max_int: int = int(86/6)
                  ) -> tuple[Solver, int]:
    y = np.array([3, 12, 1, 21, 86, 16, 2, 24, 5, 184, 822, 163,
                  40, 102, 13, 58, 253, 38, 18, 104, 40, 19, 147,
                  25, 52, 220, 35, 1, 3, 6, 8, 36, 5, 1, 14, 2, 1,
                  10, 13, 3, 43, 4])

    Aprime = A_BOOKS[:bound,:]
    yprime = np.array([int(i/y_reduction_factor) for i in y[:bound]])   

    (s, num_vars) = __get_z3_model_from_conf_matrix_smt(max_int,
                                                        Aprime, yprime)

    return (s, num_vars)


def get_haplotypes_smt(genotypes: np.ndarray,  # size num_genotypes x
                                               # bits_per_haplotype
                       ) -> tuple[Solver, int]:

    (num_genotypes, bits_per_haplotype) = genotypes.shape

    # Assumption, all bit-vectors are of the same size. Each
    # bit-vector represents a haplotype bit, the length is two because
    # when summing 1 + 1 we must be able to store 10.

    # the last 2 is due to the fact that a genotype is a combination
    # of 2 haplotypes
    num_vars = bits_per_haplotype*num_genotypes*2
    x = [Int(f'x{i}') for i in range(num_vars)]
    # x = var_list
    s = Solver()


    # Add contraints
    for i in range(num_vars):
        s.add(0 <= x[i])
        s.add(x[i] <= 1)

    # iterate over all genotypes
    for i in range(num_genotypes):
        # iterate over all bits of the BitVec
        for j in range(bits_per_haplotype):
            s.add(x[i*10+j] + x[i*10+j+5] == genotypes[i][j])

    return (s, num_vars)




###############
## Constants ##
###############


A_BOOKS = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
