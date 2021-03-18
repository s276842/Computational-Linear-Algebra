import numpy as np
import numpy.linalg as la


def choleskyBanachiewicz(A):
    """
        @param A: a nested list which represents a symmetric,
				  positiv definit n x n matrix.
        @return: False if it didn't work, otherwise the matrix G
    """
    n = len(A)

    # Initialisation of G with A
    G = []
    for i in range(n):
        line = []
        for j in range(n):
            line.append(float(A[i][j]))
        G.append(line)

    # Computation of G
    for j in range(n):
        # doesn't need the diagonals
        tmp = (A[j][j] - sum([G[j][k] ** 2 for k in range(0, j)]))
        if tmp < 0:
            return False
        G[j][j] = tmp ** 0.5
        for i in range(n):
            G[i][j] = 1 / G[j][j] * (A[i][j] - sum([G[i][k] * G[j][k] for k in range(0, j)]))
    return G


if __name__ == "__main__":
    print(choleskyBanachiewicz([[1, 2], [2, 1]]))  # Nicht positiv definit
    print(choleskyBanachiewicz([[5, 2], [1, 1]]))  # not Hermitian
    print(choleskyBanachiewicz([[5, 2], [2, 1]]))  # should be [[sqrt(5), 2/sqrt(5)],[0, 1/sqrt(5)]]
    print(choleskyBanachiewicz([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    A = np.random.randint(1,100, 9).reshape((3,3))

    print(choleskyBanachiewicz(A*A.T))