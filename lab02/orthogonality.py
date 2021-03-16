import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def gramschmidt(X):
    '''
    Performs the gramshcmidt method to compute the QR decomposition of X

    :param X: A full rank square matrix
    :return: Two matrices Q (orthogonal) and R (uppertriangular) such that X = QR
    '''
    A = X.copy()
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)

    n = A.shape[0]
    for j in range(n):
        v = A[:,j]
        for i in range(j):
            R[i,j] = np.dot(Q[:,i],A[:,j])
            v = v - R[i,j]*Q[:,i]
        R[j,j] = la.norm(v)
        x = v/R[j,j]
        x = x[:,np.newaxis]

        Q[:,j] = x[:,0]

    return Q,R



def mod_gramschmidt(X):
    '''
    Performs the modified gramshcmidt method to compute the QR decomposition of X

    :param X: A full rank square matrix
    :return: Two matrices Q (orthogonal) and R (uppertriangular) such that X = QR
    '''

    V = X.copy()
    Q = np.zeros(V.shape)
    R = np.zeros(V.shape)

    n = V.shape[0]
    for i in range(n):
        R[i,i] = la.norm(V[:,i])
        x = V[:,i]/R[i,i]
        x = x[:,np.newaxis]
        Q[:,i] = x[:,0]

        for j in range(i+1,n):
            R[i,j] = np.dot(Q[:,i],V[:,j])
            V[:,j] = V[:,j] - R[i,j]*Q[:,i]

    return Q,R


def __givens_mat(h, k, X):
    '''

    :param h:
    :param k:
    :param X:
    :return:
    '''
    # den = np.sqrt(X[k,k]**2+X[h,k]**2)
    den = np.hypot(X[k,k],X[h,k])
    c = X[k,k]/den
    s = X[h,k]/den

    G = np.identity(X.shape[0])
    G[k,k] = c
    G[h,h] = c
    G[k,h] = s
    G[h,k] = -s

    return G

def givens(X):
    '''
    Compute the QR decomposition with the Givens triangularization
    :param X: n x n full rank square matrix
    :return: Q (orthogonal) and R (upper triangular) such that X = Q @ R
    '''

    R = X.copy()
    n = R.shape[0]
    Q = np.identity(n)

    for i in range(n):                      # For each column
        for j in range(i+1, n):             # Perform a series of rotations that nullify
            G = __givens_mat(j,i,R)         # the entries below the main diagonal
            Q = G @ Q
            R = G @ R
    return Q.T,R

def __houseolder_mat(x):


    if np.ndim(x) != 2 or x.shape[1] != 1:
        y = x[:, np.newaxis]
    else:
        y = x
    n = y.shape[0]
    e1 = np.zeros(n)
    e1[0] =1
    e1 = e1[:,np.newaxis]
    u = y + np.sign(y[0,0])*la.norm(y)*e1
    u_tilde = u/la.norm(u)

    Px = np.identity(n) - 2*u_tilde@u_tilde.T

    return Px


def houseolder(X):
    '''

    :param X:
    :return:
    '''

    R = X.copy()
    n = R.shape[0]
    Q = np.identity(n)

    for j in range(n):
        x = R[j:,j]

        Px =__houseolder_mat(x)
        Pj = np.identity(n)
        Pj[j:n,j:n] = Px
        R = Pj @ R
        Q = Pj @ Q

    return Q.T,R


def check(V,Q,R, method):
    method(V,Q,R)

def check_orthogonality(Q, verbose = True):

    mat = np.identity(Q.shape[0]) - Q.T@Q
    nor = la.norm(mat)
    if verbose:
        print(f"Norma quadratica: {nor}\n")
    return nor

def check_factorization(V,Q,R):
    print(la.norm(Q.T@V - R))
    return la.norm(Q.T@V - R)





results = np.zeros((12,4))

for n in range(3,15):
    V = np.vander(np.random.randint(0,100,n).tolist(),n)
    print(V)
    gs = gramschmidt(V)
    mgs = mod_gramschmidt(V)
    g = givens(V)
    h = houseolder(V)

    results[n-3, 0] =check_orthogonality(gs[0])
    results[n-3, 1]= check_orthogonality(mgs[0])
    results[n-3, 2]= check_orthogonality(g[0])
    results[n-3, 3]= check_orthogonality(h[0])

x = np.arange(3,15)
plt.plot(x,results[:,0])
plt.plot(x,results[:,1])
plt.plot(x,results[:,2])
plt.plot(x,results[:,3])

plt.show()
#
# A = np.random.randint(0,100,N)
# A = A.reshape(int(np.sqrt(N)),int(np.sqrt(N)))
#
#
# Q,R = houseolder(A)
#
# print(A)
# print("\n")
# print(Q)
# print("\n")
# print(R)
# print("\n")
# print(Q@R)
#
