import numpy as np
from enum import Enum



def compact(A, algorithm):
    return algorithm(A)

def DOK(A):
    ''' The matrix is represented as a dictionary (indices) : value '''
    dict = {}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] != 0:
                dict[(i,j)] = A[i,j]

    return dict

def r_DOK(dict, m, n):
    B = np.zeros((m,n))
    for k,v in dict.items():
        B[k] = v

    return B


def LIL(A):
    ''' The matrix is represented as a list of lists. Each list represent a row.
     Within each list are stored the pair of (column, value)'''

    l = list()
    for i in range(A.shape[0]):
        l.append(list())
        for j in range(A.shape[1]):
            if A[i,j] != 0:
                l[i].append((j,A[i,j]))

    return l

def r_LIL(A, m, n):
    B = np.zeros((m,n))

    for i, row in enumerate(A):
        for c,v in row:
            B[i,c] = v

    return B

def COO(A):
    ''' Coordinate Format. Three lists are returned:
            - The first one contain the non-zero entries
            - the second one and the last one are respectively the row and the column of the correspondent value'''

    AA = list()
    JR = list()
    JC = list()

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                AA.append(A[i,j])
                JR.append(i)
                JC.append(j)

    return AA, JR, JC

def r_COO(tup, m, n):
    AA, JR, JC = tup
    B = np.zeros((m,n))
    for i in range(len(AA)):
        B[JR[i], JC[i]] = AA[i]

    return B

def CSR(A):
    ''' Compressed Sparse Row format. Three lists are returned:
            - the first one contain the non-zero value
            - the second one contain the correspondent columns
            - the last one is calculated iteratively as>

                    I[0] = 1
                    I[i] = I[i-1] + non_zero_value_in_row_i-1       (for i >0)

    '''
    AA = list()
    JA = list()
    IA = list()

    n = 0
    for row in A:
        for i,x in enumerate(row):
            if x != 0:
                AA.append(x)
                JA.append(i)
                if n > A.shape[0]:
                    continue
                elif n == 0:
                    IA.append(1)
                else:
                    IA.append(IA[n-1] + (A[n-1,:] != 0).sum())
                n += 1
    return AA, JA, IA


def MSR(A):
    n = min(A.shape)
    AA = list()

    for i in range(n):
        AA.append(A[i,i])


def mul_CSR(AA, JA, IA, B, m):
    x = np.zeros(m)

    for i in range(m):
        for j in range(IA[i] -1, IA[i+1]-1):
            x[i] += AA[j] * B[JA[j]]

    return x

class alg(Enum):
    DOK = DOK
    LIL = LIL
    COO = COO
    CSR = CSR

if __name__ == '__main__':
    A = np.array([[3,0,-2,0], [0,0,0,0], [-1,0,2,0], [0,3,0,1]])

    # print((A[0,:] != 0).sum())
    print(compact(A, alg.CSR))
    B = [-1,2,-3,2]

    AA, JA, IA = compact(A, alg.CSR)
    print(mul_CSR(AA, JA, IA, np.array(B).T, 4))

    print(r_COO(compact(A,alg.COO), A.shape[0], A.shape[1]))