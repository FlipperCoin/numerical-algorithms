import numpy as np
from scipy.sparse.linalg import splu
from scipy.linalg import lu
from scipy.sparse import csr_matrix

A = np.array([1e-20, 1, 1, 1]).reshape(2, 2)

def without_permutation():
    """
    find an LU decomposition of the matrix "A" without permutations.
    """

    # this function finds the LU decomposition without permutations
    slu = splu(A, permc_spec="NATURAL", diag_pivot_thresh=0, options={"SymmetricMode": True})

    # get the matrices LU from the result
    L = csr_matrix.todense(slu.L)
    U = csr_matrix.todense(slu.U)

    # matrix multiplication
    LU = L@U

    # print L
    for i in range(0, 2):
        for j in range(0, 2):
            print(f"L({i + 1},{j + 1}) = {L[i, j]}")

    print()

    # print U
    for i in range(0, 2):
        for j in range(0, 2):
            print(f"U({i + 1},{j + 1}) = {U[i, j]}")

    print()

    # print LU (after multiplication)
    for i in range(0, 2):
        for j in range(0, 2):
            print(f"LU({i + 1},{j + 1}) = {LU[i, j]}")

def with_permutation():
    """
    find an LU decomposition of the matrix "A" with permutations.
    """

    P, L, U = lu(A)

    LU = L@U
    PA = P@A

    for i in range(0, 2):
        for j in range(0, 2):
            print(f"P({i + 1},{j + 1}) = {P[i, j]}")

    print()

    for i in range(0, 2):
        for j in range(0, 2):
            print(f"L({i + 1},{j + 1}) = {L[i, j]}")

    print()

    for i in range(0, 2):
        for j in range(0, 2):
            print(f"U({i + 1},{j + 1}) = {U[i, j]}")

    print()

    for i in range(0, 2):
        for j in range(0, 2):
            print(f"LU({i + 1},{j + 1}) = {LU[i, j]}")

    print()

    for i in range(0, 2):
        for j in range(0, 2):
            print(f"PA({i + 1},{j + 1}) = {PA[i, j]}")

without_permutation()
with_permutation()