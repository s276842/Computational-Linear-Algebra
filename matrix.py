import numpy as np


def transpose(mat):
    (r, c) = mat.shape
    ret = np.zeros((c, r))

    for i in range(0, r):
        ret[:, i] = mat[i, :]

    return ret


def is_square(A):
    return A.shape[0] == A.shape[1]


def is_triangular(A):
    ''' A matrix is lower/upper triangular when all the entries below/above the main diagonal are zero'''
    if not is_square(A):
        exit()

    lower_flag = 1
    upper_flag = 2

    for i in range(1, A.shape[0]):
        for j in range(0, i):
            if A[i, j] != 0:
                lower_flag = 0
                break

    for i in range(1, A.shape[1]):
        for j in range(0, j):
            if A[j, i] != 0:
                upper_flag = 0
                break

    return lower_flag + upper_flag


def is_diagonal(A):
    ''' A matrix is diagonal if it is upper and lower triangular'''
    return is_triangular(A) == 3


def is_symmetric(A):
    ''' A matrix is symmetric if it's equal to its transpose'''
    return np.all(A == transpose(A))


def cofactor(A, i, j):
    ''' determinant of the initial matrix with the i row and j column removed'''
    return (-1) ** ((i + j)) * determinant(np.delete(np.delete(A, i, axis=0), j, axis=1))


def cof(A):
    ''' matrix of the cofactors '''

    C = np.zeros(A.shape)
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            C[i, j] = cofactor(A, i, j)

    return C


def determinant(A):
    ''' Determinant calculated with the second Laplace's theorem'''
    if A.size == 1:
        return A[0, 0]

    # if A.size >= 4 and is_triangular(A):
    #     s = 1
    #     for i in range(0,A.shape[0]):
    #         s *= A[i,i]
    #     return s

    # if has a row/column null -> 0
    # if is linear indipendent -> 0
    s = 0

    for i in range(0, A.shape[1]):
        s += A[0, i] * cofactor(A, 0, i)

    return s


def invert(A):
    ''' Inverse matrix determined with the corollary of the second Laplace's theorem'''

    return cof(A).T / determinant(A)


def rank(A):
    ''' Calculate the rank with the Kronecker's theorem '''
    if is_square(A) and determinant(A) != 0:
        return A.shape[0]

    if np.all(A == np.zeros(A.shape)):
        return 0

    r = 1

    for i in range(2, min(A.shape)):
        for j in range(0, A.shape[0] - i):
            if r == i:   continue
            for k in range(0, A.shape[1] - i):
                if not determinant(A[j:j + i, k:k + i]) == 0:
                    r = i
                    continue

    return r


def solve_system(A, B):
    ''' Solve a linear system with the Cramer's rules'''
    # Verify that the system has a solutions

    # if not rank(A) == rank(np.concatenate((A,B[np.newaxis,:]))):
    #     exit()
    # #
    # cof(A).T @ B / determinant(A)
    #
    # Si puo' ricavare la soluzione del sistema anche dal secondo teorema di Laplace

    return invert(A) @ B.T


def entry_wise_norm(A, p=2):
    B = list(np.array(A).reshape(A.size, 1))
    return float(sum(map(lambda x: abs(x) ** p, B)) ** (1 / p))


def trace(A):
    ''' The trace of a matrix is given by all the elements on the diagonal '''
    l = 0
    for i in range(min(A.shape)):
        l +=A[i, i]
    return l


#
# A = np.array([[1,ex2,3], [1,0,1], [1,1,1]])
# print(A)
# print(transpose(A))
# print(is_triangular(A))
# print(is_square(A))
# print(is_symmetric(A))

# B = np.array([[1,ex2,3],[0,0,0],[ex2,4,6]])
# print(determinant(A))
# print(rank(B))

A = np.array([[2, 1, 1], [0, 2, -1], [2, 0, 1]])
B = np.array([0, 1, 5])
print(f"norma di [1,2]: {entry_wise_norm(B)}")

print(solve_system(A, B))
