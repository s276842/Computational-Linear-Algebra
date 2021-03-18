import numpy as np
import scipy as sp


# np.linalg.norm

def my_norm(A, p=2):
    s = 0
    B = np.array(A, ndmin=2)
    (r, c) = B.shape
    for i in range(0, r):
        for j in range(0, c):
            s += abs(B[i, j]) ** p

    return s ** (1 / p)


def my_infnorm(A):
    s = 0
    (r, c) = A.shape
    for i in range(0, r):
        for j in range(0, c):
            if A[i, j] > s:
                s = A[i, j]

    return s


def my_determinant(A):
    s = 0
    if A.size == 1:
        return A[0, 0]

    for i in range(0, A.shape[1]):
        s += (-1) ** (i % 2) * A[0, i] * my_determinant(np.delete(np.delete(A, 0, axis=0), i, axis=1))

    return s


def my_cof(A):
    B = np.zeros(A.shape)

    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            B[i,j] = (-1)**((i+j)%2) * my_determinant(np.delete(np.delete(A,i,axis=0),j, axis=1))

    return B


def my_inverse(A):
    '''return the inverse of the matrix A computed with the second Laplace's theorem'''

    return my_cof(A).T / my_determinant(A)

# np.inner(vector 1, vector ex2) or np.dot(v1,v2)
# Frobenius prod -> np.trace(A.T@B)

def my_solvesys(A, B):
    ''' Solve a linear system with the Cramer's rule'''

    return my_inverse(A) @ B.T

A = np.array([[1, 0, 2], [2, 1, -1], [0, 3, 1]])
B = np.array([[1, 0, 2]])

V1 = np.array([1, 2, 3])
V2 = np.array([1, -2, 3])

# print(my_norm(A,ex2))
# print(my_infnorm(A))


### Considering a system: AX = B ###

AB = np.concatenate((A, B.T), axis=1)  # to concatenate two matrices (axis 1 vertically)
print(AB)

# Checking that the matrix has full rank
np.linalg.det(A)  # we should compute the determinant of the matrices

Ainv = np.linalg.inv(A)  # if this is non zero value => A as an invert
A @ Ainv == np.all(np.identity(3))  # testing that this is true (without np.all we have a matrix of bool)
# (see also np.any and np.array_equal)


# Checking that the linear systema has solution(s)
np.linalg.matrix_rank(A) == np.linalg.matrix_rank(AB)

print(np.linalg.solve(A, np.zeros(3)))
# np.linalg.lstsq(A,B)                        # returns the solutions that minimize the euclidean ex2 norm between B
# and AX


print(my_determinant(A))
print(my_inverse(A))

print(my_inverse(A) == np.linalg.inv(A))
print(my_solvesys(A,B))


A = np.array([[1,2,3], [1,0,1], [1,1,1]])
print(my_determinant(A))