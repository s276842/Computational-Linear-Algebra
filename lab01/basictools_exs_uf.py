import numpy as np
import scipy as sp


def my_trasp(A):
    B=np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[j,i]=A[i,j]
    return B

def my_trace(A):
    t=0
    p=min(A.shape[0],A.shape[1])
    for i in range(p):
        t=t+A[i,i]
    return t

def my_operations(A,B):
    S=np.zeros(A.shape)
    M=np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            S[i,j]=A[i,j]+B[i,j]
            M[i,j]=A[i,j]*B[i,j]
    return S, M

def my_mult(A,B):
    C=np.zeros((A.shape[0],B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i,j]=C[i,j]+A[i,k]*B[k,j]
    return C

def my_norm(A,p):
    s=0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            s=s+abs(A[i,j])**p
    s=s**(1/p)
    return s

def my_infnorm(A):
    s=0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i,j])>s:
                s=abs(A[i,j])
    return s

def my_det(A):
    d=0
    if A.shape==(1,1):
        return A[0,0]
    for j in range(A.shape[1]):
        cof=np.delete(A,0,axis=0)
        cof=np.delete(cof,j,axis=1)
        d=d+A[0,j]*(-1)**(j)*my_det(cof)
    return d

def my_cof(A):
    C=np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            cof=np.delete(A,i, axis=0)
            cof=np.delete(cof, j, axis=1)
            C[i,j]=(-1)**(i+j)*my_det(cof)
    return C

def my_inv(A):
    return my_cof(A).T/my_det(A)

def my_gauss(M):
    # A=M.copy()
    A=M.astype(np.float)
    m = A.shape[0]
    n = A.shape[1]
    p =min(m,n)

    for j in range(p):
        # Find the maximum abs in column j
        maxEl = abs(A[j,j])
        maxRow = j
        for i in range(j+1, m):
            if abs(A[i,j]) > maxEl:
                maxEl = abs(A[i,j])
                maxRow = i
        if maxEl!=0:
            # Swap maximum row with row j
            for k in range(j, n):
                tmp = A[maxRow, k]
                A[maxRow,k] = A[j,k]
                A[j,k] = tmp

            # Perform row operations to put as 0 all the entries below entry of position (j,j)
            for k in range(j+1, m):
                c = -A[k,j]/A[j,j]
                for i in range(j, n):
                    A[k,i] = A[k,i] + c * A[j, i]

    return A

def my_gauss_solve(A,b):
    # Take as arguments: a 2D array A encoding a SQUARE INVERTIBLE matrix, and a 1D array b
    Ab=np.concatenate((A, np.array([b]).T ), axis=1)
    M=my_gauss(Ab)
    m = A.shape[0]
    x=np.zeros(m)
    for i in range(m-1, -1, -1):
        x[i] = M[i,m]/M[i,i]
        for k in range(i-1, -1, -1):
            M[k,m] = M[k,m] - M[k,i]*x[i]
    return x