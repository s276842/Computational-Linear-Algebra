import time

import numpy as np
import matplotlib.pyplot as plt
from graph import Graph
from scipy import sparse
from scipy.sparse import linalg as spla
import scipy.linalg as la
import scipy as sp

# Seaborn is only used to set the style. Disabling these two lines, the plots are displayed anyway but with a
# different style from the one in the report
import seaborn as sns
sns.set()


np.random.seed(276842)

from sparse import LIL, unweighted_LIL, CSR



def power_iteration(A, v = None, tol = 1e-15, maxIter = 100):

    n = A.shape[0]

    if v is None:
        v = np.random.rand(n)

    v = v/np.linalg.norm(v)
    v_next = np.array(A.dot(v)).squeeze()
    lam = np.dot(v,v_next)

    k = 0
    while k < maxIter:
        k +=1
        lam_old = lam
        v = v_next/np.linalg.norm(v_next)
        v_next = np.array(A.dot(v)).squeeze()
        lam = np.dot(v, v_next)     # Rayleigh quotient
        if np.abs(lam_old-lam) < tol:
            break
    return lam, v_next/np.linalg.norm(v_next)



def inverse(A, mu=0, v = None, tol = 1e-15, maxIter = 100):

    n = A.shape[0]

    if v is None:
        v = np.random.rand(n)

    LU = spla.spilu(A - mu*sp.sparse.eye(n))
    v = v/np.linalg.norm(v)
    v_next = LU.solve(v)
    lam = np.dot(v,v_next)
    k = 0

    while k < maxIter:
        k+=1
        lam_old = lam
        v = v_next/np.linalg.norm(v_next)
        v_next = LU.solve(v)
        lam = np.dot(v,v_next)

        if np.abs(lam- lam_old) < tol:
            break
    return (1/lam)+mu, v_next/np.linalg.norm(v_next)




if __name__ == '__main__':


    ### Part 1: adjacency matrix ###

    # 1.1 Construction of the graph and adjacency matrix
    G = Graph(path='edge_file_facebook_python.txt', adj_format=CSR)
    print(f"Graph loaded: {G.nodes} num of nodes and {G.num_edges} edges readed\n\n")

    # 1.2 Visualizing sparsity pattern
    G.spy()


    # 1.4 Eigvector centrality - My implementation
    start = time.time()
    first_eigvalue, first_eigvector = power_iteration(G.adj, maxIter=1000)
    time_myimp = time.time()-start

    # Showing results
    print('============ Part 1 - Adjacency matrix ============\n')
    print("(My implementation)")
    print(f"\tEigenvalue found with Rayleigh quotient iteration: {first_eigvalue}")
    print(f"\tThe Perron-Frobenius theorem (|avg_deg| < lambda_n < |max_deg|) is respected, since: {np.mean(G.degrees)} < {first_eigvalue} < {np.max(G.degrees)}")
    print(f"\tChebyshev distance max(|Av - lambda*v|): {np.max(G.adj.dot(first_eigvector) - first_eigvalue*first_eigvector)}\n")
  
    # Plotting components of first eigenvector
    plt.scatter(np.arange(len(first_eigvector)), first_eigvector, s=0.08)
    plt.xlabel('nodes')
    plt.ylabel('value')
    plt.title('First eigenvector')

    # Computing eigenvector centrality
    eigvector_centrality = np.abs(first_eigvector)/la.norm(np.abs(first_eigvector))
    print(f"\tEigv_centrality: {eigvector_centrality}")
    print(f"\tnorm={la.norm(eigvector_centrality):.2f}\n")

    # Showing results
    ind = np.argmax(eigvector_centrality)
    plt.scatter(ind, first_eigvector[ind], c='r', s=2, label='highest eigenvect. centrality')
    plt.legend(loc='upper left')
    plt.show()
    print(f"\tNode {ind} has maximum eigenvector centrality, with value {eigvector_centrality[ind]:.3f}")
    print(f"\tNode {ind} has degree {G.degrees[ind]}\n\n")

    # G.spy(dpi=900, highlight=1912, save=False)

    # Comparing with scipy implementation
    edges = np.loadtxt('edge_file_facebook_python.txt', dtype=int)
    edges = np.vstack( (edges , edges[:,::-1]) )
    A = sparse.coo_matrix( (np.ones(edges.shape[0]) , (edges[:,0], edges[:,1])), shape=(4039, 4039)).tocsr()

    start = time.time()
    sp_first_eigvalue, sp_first_eigvector = spla.eigs(A, k=1)
    time_spimp = time.time() - start
    sp_first_eigvalue = sp_first_eigvalue[0].real
    sp_first_eigvector = np.real(sp_first_eigvector.flatten())
    print("(Scipy implementation)")
    print(f"\tEigenvalue found with scipy.sparse.linalg.eigs: {sp_first_eigvalue}")
    print(f"\tChebyshev distance ||Av - lambda*v||: {np.max(A.dot(sp_first_eigvector) - sp_first_eigvalue*sp_first_eigvector)}\n")

    sp_eigvector_centrality = np.abs(sp_first_eigvector)/np.linalg.norm(np.abs(sp_first_eigvector))
    print(f"\tEigv_centrality: {sp_eigvector_centrality}")
    print(f"\tnorm={la.norm(sp_eigvector_centrality):.2f}\n")

    ind = np.argmax(sp_eigvector_centrality)
    print(f"\tNode {ind} has maximum eigenvector centrality, with value {sp_eigvector_centrality[ind]}")
    print(f"\tNode {ind} has degree {G.degrees[ind]}\n\n")


    # comparison
    print(" (Comparison)")
    print(f"\tEigenvalue approx error: {np.abs(first_eigvalue-sp_first_eigvalue)}")
    print(f"\tMax eigenvector approx error: {np.max(first_eigvector-sp_first_eigvector)}")
    print(f"\tDot product between the two eigenvectors:  {np.dot(first_eigvector, sp_first_eigvector)}")
    print(f"\tTime for the computation of the eigenvalues: {time_myimp:.3f}s (my implementation), {time_spimp:.3f}s (sp implementation)")




    ### Part 2: Laplacian matrix ###
    print('\n\n\n============ Part 2 - Laplacian matrix ============\n')
    laplacian = G.laplacian_matrix()
    print(f'Laplacian matrix constructed in LIL format:\n{laplacian}')
    sum_of_rows = [np.sum(x) if x is not None else 0 for x in laplacian.get_rows()]
    print(f"\nEach row of the Laplacian matrix sums to 0: {np.any(sum_of_rows != 0)}")


    ### Part 3: Normalized Laplacian matrix ###
    print('\n\n\n============ Part 3 - Normalized Laplacian matrix ============\n')

    norm_laplacian = G.norm_laplacian_matrix()
    sp_norm_laplacian = sparse.csgraph.laplacian(A, normed=True).tocsc()

    eigs = []
    times = []
    error_eigs = [0]
    error_eigv = [0]
    cheb = []
    names = ['scipy', 'method 1', 'method 2', 'method 3']

    start = time.time()
    res = spla.eigsh(sp_norm_laplacian, 2, which='SM')
    sp_eig2 = res[0][1]
    sp_eigv2 = res[1][:, 1]
    end = time.time()
    print(f"(Scipy)")
    print(f"\tsecond smallest eig of normalized Laplacian matrix: {sp_eig2}")
    print(f"\ttime equired: {end - start}s")
    print(f"\tmax error of approximation Av - lambda*v: {np.max(sp_norm_laplacian.dot(sp_eigv2) - sp_eig2 * sp_eigv2)}\n")
    eigs.append(sp_eig2)
    times.append(end - start)
    cheb.append(np.max(sp_norm_laplacian.dot(sp_eigv2) - sp_eig2 * sp_eigv2))


    start = time.time()
    M = 2*sp.sparse.eye(sp_norm_laplacian.shape[0]) - sp_norm_laplacian     # M = 2I - Ln
    eig, eigv = power_iteration(M, maxIter=1500)                            # computation of largest eig of M
    M = M - eig*eigv[:, sp.newaxis]*eigv[sp.newaxis,:]                      # deflation

    eig2, eigv2 = power_iteration(M, maxIter=1500)
    eig2 = 2 - eig2
    end = time.time()
    print(f"(Method 1 - Deflation method)")
    print(f"\tsecond smallest eig of normalized Laplacian matrix: {eig2}")
    print(f"\ttime required: {end - start}s")
    print(f"\tmax error of approximation Av - lambda*v: {np.max(sp_norm_laplacian.dot(eigv2) - eig2 * eigv2)}")
    print(f"\tdifference with eigenvalue computed with sp: {np.abs(eig2 - sp_eig2)}")
    print(f"\tl2 distance with eigenvector computed with sp: {np.linalg.norm(eigv2 - sp_eigv2, 2)}\n")
    eigs.append(eig2)
    times.append(end - start)
    error_eigs.append(np.abs(eig2 - sp_eig2))
    error_eigv.append(np.linalg.norm(eigv2 - sp_eigv2, 2))
    cheb.append(np.max(sp_norm_laplacian.dot(eigv2) - eig2 * eigv2))

    start = time.time()
    comp_eigv = sp.sparse.csr_matrix(np.diag(G.degrees)).power(1 / 2).dot(np.ones(4039))    # This is eigenvalue of zero
    comp_eigv = comp_eigv / np.linalg.norm(comp_eigv, 2)                                    # Normalization
    M = 2 * sp.sparse.eye(sp_norm_laplacian.shape[0]) - sp_norm_laplacian                   # M = 2I - Ln
    M = M - 2 * comp_eigv[:, sp.newaxis] * comp_eigv[sp.newaxis, :]                         # deflation
    eig2, eigv2 = power_iteration(M, maxIter=1500)
    eig2 = 2 - eig2
    end = time.time()
    print(f"(Method 2 - Deflation with computed eigenvalue)")
    print(f"\tsecond smallest eig of normalized Laplacian matrix: {eig2}")
    print(f"\ttime required: {end - start}s")
    print(f"\tmax error of approximation Av - lambda*v:  {np.max(sp_norm_laplacian.dot(eigv2) - eig2 * eigv2)}")
    print(f"\tdifference with eigenvalue computed with sp: {np.abs(eig2 - sp_eig2)}")
    print(f"\tl2 distance with eigenvector computed with sp: {np.linalg.norm(eigv2 - sp_eigv2, 2)}\n")
    eigs.append(eig2)
    times.append(end - start)
    error_eigs.append(np.abs(eig2 - sp_eig2))
    error_eigv.append(np.linalg.norm(eigv2 - sp_eigv2, 2))
    cheb.append(np.max(sp_norm_laplacian.dot(eigv2) - eig2 * eigv2))


    start = time.time()
    eig2, eigv2 = inverse(sp_norm_laplacian, mu=1e-7, maxIter=200)
    end = time.time()
    print(f"(Method 3 - shift-inverted power method)")
    print(f"\tsecond smallest eig of normalized Laplacian matrix: {eig2}")
    print(f"\ttime required: {end - start}s")
    print(f"\tmax error of approximation Av - lambda*v:  {np.max(sp_norm_laplacian.dot(eigv2) - eig2 * eigv2)}")
    print(f"\tdifference with eigenvalue computed with sp: {np.abs(eig2 - sp_eig2)}")
    print(f"\tl2 with eigenvector computed with sp: {np.linalg.norm(eigv2 - sp_eigv2, 2)}\n")
    eigs.append(eig2)
    times.append(end - start)
    error_eigs.append(np.abs(eig2 - sp_eig2))
    error_eigv.append(np.linalg.norm(eigv2 - sp_eigv2, 2))
    cheb.append(np.max(sp_norm_laplacian.dot(eigv2) - eig2 * eigv2))



    # Plot results
    ticks = np.arange(4)
    colors = ['orangered', 'gold', 'limegreen', 'dodgerblue']

    plt.bar(ticks, cheb, width=0.7, color=colors)
    plt.xticks(ticks, names)
    plt.title("Chebyshev distance Av - lambda*v")
    plt.show()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(12, 8))

    ax1.bar(ticks, eigs, width=0.7, color=colors)
    ax1.set_ylim(0, 0.0012)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(names)
    ax1.set_title("Second smallest eigenvalue")


    ax2.bar(ticks, times, width=0.7, color=colors)
    ax2.set_ylim(0, 10)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(names)
    ax2.set_title("Time required in seconds for the computation")

    ax3.bar(ticks, error_eigs, width=0.7, color=colors)
    ax3.set_ylim(0, 0.0012)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(names)
    ax3.set_title("Absolute difference of the eigenvalue respect to scipy")


    ax4.bar(ticks, error_eigv, width=0.7, color=colors)
    ax4.set_ylim(0, 0.55)
    ax4.set_xticks(ticks)
    ax4.set_xticklabels(names)
    ax4.set_title("l2 difference of the eigenvector respect to scipy")
    fig.suptitle("Results for the computation of the second smallest eigenvalue", fontsize=20)

    plt.legend(labels=names)
    #plt.savefig('result.png')
    plt.show()