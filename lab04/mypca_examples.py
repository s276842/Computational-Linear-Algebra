import pandas as pd
import numpy as np
import numpy.linalg as la
from mypca import my_pca, pc_approx, my_pca_varnorm, pc_approx_varnorm, my_pca_svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pd.options.display.max_rows = 15
pd.options.display.max_columns = 15

iris_path = 'Iris.csv'
wines_path = 'wine.csv'

def main():

    iris_df = pd.read_csv(iris_path, index_col='Id')
    wines_df = pd.read_csv(wines_path)

    iris_feats = iris_df.columns[:-1].to_list()
    wines_feats = wines_df.columns[1:].to_list()

    iris_X = iris_df.loc[:, iris_feats].values
    wines_X = wines_df.loc[:, wines_feats].values

    iris_N, iris_n = iris_X.shape
    wines_N, wines_n = wines_X.shape

    # ---- PCA for IRIS DATASET ---

    iris_S, iris_W, iris_U, iris_lambdas, iris_mu = my_pca_svd(iris_X)
    iris_lambdas_ratio = iris_lambdas / iris_lambdas.sum()
    iris_lambdas_ratio_incr = np.zeros(iris_n)
    for i in range(iris_n):
        iris_lambdas_ratio_incr[i] = iris_lambdas_ratio[:i + 1].sum()

    iris_pc_df = pd.DataFrame(iris_U.T, columns=iris_feats,
                              index=list(range(1, iris_n + 1)))
    iris_pc_df.index.name = 'P.C.'

    print('*** IRIS DATASET ***')
    print('Principal components:')
    print(iris_pc_df)
    print('-------------')
    print('Explained Variance (Ratio): ')
    for i in range(iris_n):
        print('lambda{}:'.format(i + 1), '~ {}'.format(np.round(iris_lambdas[i], 2)),
              '(~ {})'.format(np.round(iris_lambdas_ratio[i], 2)))
    print('-------------')
    print('Example: Appoximation of sample x1 with m P.C.')
    print('x1 =', iris_X[0, :])
    for i in range(iris_n):
        print('m={})'.format(i + 1))
        x1apprx = pc_approx(np.expand_dims(iris_W[0, :], axis=0), iris_U, i + 1, iris_mu)
        x1apprx = np.round(x1apprx, 3)
        print('x1 ~='.format(i + 1), x1apprx[0, :])


    plt.figure()
    plt.title('IRIS DATASET - Var. Explanation with m P.C.')
    plt.plot(np.arange(iris_n + 1), [0] + iris_lambdas_ratio_incr.tolist())
    plt.xlabel('m')
    plt.xticks(np.arange(0, iris_n + 1), [''] +
               ['u{}'.format(i) for i in range(1, iris_n + 1)])
    plt.ylabel('Total Var. Expl. Ratio')

    iris_colors_dict = {
        'Iris-setosa': 'red',
        'Iris-versicolor': 'green',
        'Iris-virginica': 'blue'}
    iris_colors_list = [iris_colors_dict[spec] for spec in iris_df['Species']]

    plt.figure()
    plt.title('IRIS DATASET - 2 P.C. plot')
    plt.scatter(iris_W[:, 0], iris_W[:, 1], c=iris_colors_list)
    plt.xlabel('w1')
    plt.ylabel('w2')


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False)
    fig.suptitle('IRIS DATASET - 2 Components plots')
    ax1.scatter(iris_X[:, 0], iris_X[:, 1], c=iris_colors_list)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax2.scatter(iris_X[:, 2], iris_X[:, 3], c=iris_colors_list)
    ax2.set_xlabel('x3')
    ax2.set_ylabel('x4')
    ax3.scatter(iris_X[:, 0], iris_X[:, 2], c=iris_colors_list)
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x3')
    ax4.scatter(iris_X[:, 1], iris_X[:, 3], c=iris_colors_list)
    ax4.set_xlabel('x2')
    ax4.set_ylabel('x4')

    # ---- PCA for WINES DATASET ---

    wines_S, wines_W, wines_U, wines_lambdas, \
    wines_mu, wines_sigma = my_pca_varnorm(wines_X)
    wines_lambdas_ratio = wines_lambdas / wines_lambdas.sum()
    wines_lambdas_ratio_incr = np.zeros(wines_n)
    for i in range(wines_n):
        wines_lambdas_ratio_incr[i] = wines_lambdas_ratio[:i + 1].sum()

    wines_pc_df = pd.DataFrame(wines_U.T, columns=wines_feats,
                              index=list(range(1, wines_n + 1)))
    wines_pc_df.index.name = 'P.C.'

    print('*** WINES DATASET ***')
    print('Principal components:')
    print(wines_pc_df)
    print('-------------')
    print('Explained Variance (Ratio): ')
    for i in range(wines_n):
        print('lambda{}:'.format(i + 1), '~ {}'.format(np.round(wines_lambdas[i], 2)),
              '(~ {})'.format(np.round(wines_lambdas_ratio[i], 2)))
    print('-------------')
    print('Example: Appoximation of sample x1 with m P.C.')
    print('x1 =', wines_X[0, :])
    for i in range(wines_n // 3):
        print('m={})'.format(i + 1))
        x1apprx = pc_approx_varnorm(np.expand_dims(wines_W[0, :], axis=0), wines_U, i + 1,
                                    wines_mu, wines_sigma)
        print('x1 ~='.format(i + 1), x1apprx[0, :])


    plt.figure()
    plt.title('WINES DATASET - Var. Explanation with m P.C.')
    plt.plot(np.arange(wines_n + 1), [0] + wines_lambdas_ratio_incr.tolist())
    plt.xlabel('m')
    plt.xticks(np.arange(0, wines_n + 1), [''] +
               ['u{}'.format(i) for i in range(1, wines_n + 1)])
    plt.ylabel('Total Var. Expl. Ratio')

    wines_colors_dict = {
        1: 'red',
        2: 'green',
        3: 'blue'}
    wines_colors_list = [wines_colors_dict[cl] for cl in wines_df['class']]

    plt.figure()
    plt.title('WINES DATASET - 2 P.C. plot')
    plt.scatter(wines_W[:, 0], wines_W[:, 1], c=wines_colors_list)
    plt.xlabel('w1')
    plt.ylabel('w2')

    fig = plt.figure()
    fig.suptitle('WINES DATASET - 3 P.C. plot')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wines_W[:, 0], wines_W[:, 1], wines_W[:, 2], c=wines_colors_list)
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('w3')

    plt.show()


if __name__ == '__main__':
    main()




