import numpy as np
import numpy.linalg as la


def my_pca(X):
    """
    description...
    :param X: Matrix N-by-n (np.array) where each row is a record of a single n-dimensional datum.
              We assume that N is always greater than (or, at least, equal to) n.
    :return:
    """

    N, n = X.shape
    # Computation of the mean row-vector of X using X.mean
    mu = np.mean(X, axis=0)
    # Computation of the re-centered matrix B
    B = X - mu

    # Computation of the sample covariance matrix S
    S = np.dot(B.T,B)/(N-1)

    # Computation of the eigenvalues and eigenvectors of S using (for simplicity) np.eig
    # S_eigs =
    lambdas, U = la.eig(S)

    # Sorting lambdas and U because np.eig does not guarantee that eigenvalues are sorted
    i_lambdas = np.argsort(lambdas)[::-1]

    lambdas = lambdas[i_lambdas]
    U = U[:, i_lambdas]

    # Computation of the coordinates of X with respect to the basis of principal components
    W = B @ U

    return S, W, U, lambdas, mu


def pc_approx(W, U, m, mu):
    """
    description ...
    :param W: Matrix N-by-n (np.array) where each row is a record of a single n-dimensional datum.
              We assume that N is always greater than (or, at leat, equal to) n.
    :param U: Matrix n-by-n (np.array) where each column is a principal component
    :param m:
    :param mu:
    :return:
    """

    # If (for error) m > n, we set it to n.

    if type(m) == int:
        # If (for error) m > n, we set it to n.
        m = min(U.shape[0], m)
    elif type(m) == float:
        B = np.dot(W, U.T)

        tot_variance = np.sum(la.eig(B)[0])
        var_explained = 0
        n = 0
        while var_explained < m:
            lam = B * U[:, n]
            var_explained += lam / tot_variance
            n += 1

        m = n

    # Truncation of U and W
    Um = U[:, :m]
    Wm = W[:, :m]
    # Computation of Xtilde
    Xtilde = Wm @ Um.T + mu     # Nota sono da invertire perche' si lavora per riga

    return Xtilde


def my_pca_varnorm(X):
    """
    description...
    :param X: Matrix N-by-n (np.array) where each row is a record of a single n-dimensional datum.
              We assume that N is always greater than (or, at leat, equal to) n.
    :return:
    """


    N, n = X.shape
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    B = (X - mu)/sigma

    # Computation of the sample covariance matrix S
    S = np.dot(B.T,B)/(N-1)

    # Computation of the eigenvalues and eigenvectors of S using (for simplicity) np.eig
    lambdas, U = la.eig(S)

    # Sorting lambdas and U because np.eig does not guarantee that eigenvalues are sorted
    i_lambdas = np.argsort(lambdas)[::-1]

    lambdas = lambdas[i_lambdas]
    U = U[:, i_lambdas]

    # Computation of the coordinates of X with respect to the basis of principal components
    W = B @ U

    return S, W, U, lambdas, mu, sigma


def pc_approx_varnorm(W, U, m, mu, sigma):
    """
    description ...
    :param W: Matrix N-by-n (np.array) where each row is a record of a single n-dimensional datum.
              We assume that N is always greater than (or, at leat, equal to) n.
    :param U: Matrix n-by-n (np.array) where each column is a principal component
    :param m:
    :param mu:
    :param sigma:
    :return:
    """

    if type(m) == int:
        # If (for error) m > n, we set it to n.
        m = min(U.shape[0], m)
    elif type(m) == float:
        B = np.dot(W, U.T)

        tot_variance = np.sum(la.eig(B)[0])
        var_explained = 0
        n = 0
        while var_explained < m:
            lam = B*U[:,n]
            var_explained += lam/tot_variance
            n+=1

        m = n

    # Truncation of U and W
    Um = U[:, :m]
    Wm = W[:, :m]
    # Computation of Xtilde
    Xtilde = np.multiply(Wm @ Um.T, sigma) + mu     # Nota sono da invertire perche' si lavora per riga

    return Xtilde



def my_pca_svd(X):
    """
    description...
    :param X: Matrix N-by-n (np.array) where each row is a record of a single n-dimensional datum.
              We assume that N is always greater than (or, at least, equal to) n.
    :return:
    """

    N, n = X.shape
    # Computation of the mean row-vector of X using X.mean
    mu = np.mean(X, axis=0)
    # Computation of the re-centered matrix B
    B = X - mu


    U, S, V = la.svd(B.T)

    lambdas = np.power(S,2)/(N-1)
    U = U/(N-1)

    # Sorting lambdas and U because np.eig does not guarantee that eigenvalues are sorted
    i_lambdas = np.argsort(lambdas)[::-1]

    lambdas = lambdas[i_lambdas]
    U = U[:, i_lambdas]

    # Computation of the coordinates of X with respect to the basis of principal components
    W = B @ U

    return S, W, U, lambdas, mu