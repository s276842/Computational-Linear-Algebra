import numpy as np


def my_gauss(M, tol = 1e-12):
    '''
    Given a matrix returns is row-echelon form

    :param M: the matrix
    :param tol: tolerance
    :return: the row echelon form of matrix M
    '''


    A = M.copy()        # save a copy of original matrix

    m, n = A.shape      # save dimension
    p = min(m,n)

    for j in range(p):          # for each column
        row_maxabsel = j + np.argmax(A[j:,j])       # finding the row max absolute value (pivot)
                                                    # j for count the offset
        maxabsel = A[row_maxabsel, j]

        if maxabsel != 0:
            jth_row = A[j, :].copy()                # permutate it with the one with the pivot
            A[j, :] = A[row_maxabsel, :].copy()
            A[row_maxabsel, :] = jth_row.copy()

            cvec = np.array(A[(j+1):, j] / maxabsel, ndmin=2).T     # finding the value to set to zero the other j-th's column
            pivot_row = np.array(A[j, :], ndmin=2)
            A[(j+1):,:] = A[(j+1):,:] -cvec*pivot_row

    A[abs(A) < tol] = 0

    return A


def my_rank(M):
    G = my_gauss(M)
    return sum(np.count_nonzero(G,axis=np.argmax(G.shape)) != 0)


def my_det(A):
    d=0
    if A.shape==(1,1):
        return A[0,0]
    for j in range(A.shape[1]):
        cof=np.delete(A,0,axis=0)
        cof=np.delete(cof,j,axis=1)
        d=d+A[0,j]*(-1)**(j)*my_det(cof)
    return d

def my_gauss_solver(A, b):
    """
    Function that solve the linear square system A @ x = b using the pivoting method
    :param A: 2D-array object (numpy ndarray), representing a square matrix
    :param b: 2D-array object (numpy ndarray), representing a column vector
    :return: 2D-array object (numpy ndarray), representing the column vector x s.t. A @ x = b
    """
    m, n = A.shape

    if m != n:
        x = None
        print('The matrix is rectangular!')
    elif my_det(A) == 0:
        x = None
        print('The matrix is singular!')
    else:
        x = np.zeros((m, 1))
        Ab = np.hstack([A, b])
        G_Ab = my_gauss(Ab)
        x[m - 1, 0] = G_Ab[m - 1, m] / G_Ab[m - 1, m - 1]
        for i in range(m - 2, -1, -1):
            x[i, 0] = (G_Ab[i, m] - (np.array(G_Ab[i, :-1], ndmin=2) @ x)) / G_Ab[i, i]

    return x




if __name__ == '__main__':
    # A = np.random.randint(-100,100,(5,6))
    A = np.array([[1,1,0],[0,2,1],[3,1,3]])
    B = np.array([[10],[4],[13]])
    print(my_gauss(A))
    print(my_rank(A))
    print(A)
    print(A@my_gauss_solver(A,B))
