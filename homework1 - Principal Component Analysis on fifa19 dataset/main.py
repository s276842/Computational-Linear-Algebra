import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt

# Seaborn is required only to plot the graph as the one in the report.
# Disabling these lines the graph are plotted anyway but with a different style
import seaborn as sns
sns.set()


LABEL_SIZE = 12
TITLE_SIZE = 20
PAD = 20
FIG_SIZE =(10, 6)
FIG_SIZE_HBARGRAPH = (10, 10)

class PCA():
    n_components = None
    var_explained = None

    def __init__(self, n_components = 0.9, use_mean=True, use_std=True):
        self.use_mean = use_mean
        self.use_std = use_std

        self.n_components = n_components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        self.N, self.n = X.shape


        B = X.copy()
        if self.use_mean:
            self.mean = X.mean(axis=0)
            B -= self.mean
        if self.use_std:
            self.std = np.sqrt(X.var(axis=0))
            B /= self.std
        self.B = B
        # Computation of the sample covariance matrix S

        self.U, S, V = la.svd(B.T, full_matrices=False)

        self.lambdas = (S ** 2) / (self.N - 1)
        self.U = self.U / (self.N - 1)

        i_lambdas = np.argsort(self.lambdas)[::-1]

        self.lambdas = self.lambdas[i_lambdas]
        self.U = self.U[:, i_lambdas]
        self.U = self.U / la.norm(self.U, axis=0)


        self.__find_components(self.n_components)


    def __find_components(self, n_components):
        tot_variance = np.sum(self.lambdas)
        variance_explained = [x / tot_variance for x in self.lambdas]
        cum_variance = np.cumsum(variance_explained)

        if n_components is int:
            self.n_components = n_components
        else:
            self.n_components = np.argmax(cum_variance > n_components)+1

        self.var_explained = cum_variance[self.n_components-1]



    def plot_var_explained_ratio(self, save = False):
        tot_variance = np.sum(self.lambdas)
        variance_explained = [x / tot_variance for x in self.lambdas]
        cum_variance = np.cumsum(variance_explained)

        plt.figure(figsize=FIG_SIZE)
        plt.title("Percentage Variance Explained", fontdict={'fontsize': TITLE_SIZE, 'verticalalignment': 'baseline',
                        'horizontalalignment': 'center'}, pad=PAD)

        plt.xlim((0.4, len(self.lambdas)))
        plt.bar(np.arange((len(self.lambdas) ))+1, variance_explained, color=	'#1E90FF', label = '% Variance explained', clip_on=False)
        plt.plot(np.arange(len(self.lambdas))+1, cum_variance.tolist(), color='#DC143C', marker='o', markersize=2,
                 linewidth=0.5, linestyle='-', label='Cumulative variance', clip_on=False)

        if hasattr(self, 'var_explained'):
            plt.plot([self.n_components, self.n_components], [0, self.var_explained], color='green', linewidth=0.5, label = f'{self.var_explained:.2f} Variance')
            plt.plot([0.4, self.n_components], [self.var_explained, self.var_explained], color='green', linewidth=0.5, clip_on=False, zorder=100)

        plt.xlabel('Principal component', labelpad=PAD, size=LABEL_SIZE)
        plt.ylabel('% Variance Explained', labelpad=PAD, size=LABEL_SIZE)
        plt.margins(x=0.01)
        if save:
            plt.savefig('variance_explained.png')

        plt.legend(loc='lower right')
        plt.show()

    def transform(self, X):
        B = X.copy()
        if self.use_mean:
            B -= self.mean
        if self.use_std:
            B /= self.std

        Um = self.U[:, :self.n_components]
        return self.B @ Um

    def inv_transform(self, W):
        Um = self.U[:,:self.n_components]
        B = W @ Um.T
        if self.use_std:
            B *= self.std
        if self.use_mean:
            B += self.mean

        return B

    def plot_pc_horizontalbargraph(self, component='all', labels=None, title=None, save=False):
        if(type(component) is int):
            l = [component-1]
            title = [title]
        elif(component is list):
            l = np.array(component) - 1
        elif component=='all':
            l = [i for i in range(self.n_components)]

        if labels == None:
            labels = [f"attr{i}" for i in range(self.U.shape[0])]

        for k in l:
            if title == None:
                t = f"Principal Component {k+1}"
            else:
                t = title[k]
            self.__plot_pc(self.U[:, k], labels, t, save)

    def __plot_pc(self, u, labels, title, save):
        fig = plt.figure(figsize=FIG_SIZE_HBARGRAPH)

        ax = fig.add_subplot()
        ax.barh(np.arange(len(labels)), u, align="center", color='#F08080')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_title(title,  fontdict={'fontsize': TITLE_SIZE, 'horizontalalignment': 'center', 'verticalalignment': 'baseline'},
                     pad = PAD)

        # Switch off ticks
        ax.tick_params(axis="both", which="minor", bottom="off", top="off", labelbottom="on", left="off",
                       right="off",
                       labelleft="on")

        # Draw vertical axis lines
        vals = ax.get_xticks()
        for tick in vals:
            ax.axvline(x=tick, linestyle='dashed', alpha=0.3, color='#eeeeee', zorder=1)
        ax.yaxis.grid(True, linestyle='dashed', alpha=0.3, color='#86bf91')

        # Set labels
        ax.set_xlabel("Value (Contribution)", labelpad=PAD, size=LABEL_SIZE)
        ax.set_ylabel("Skill", labelpad=PAD, size=LABEL_SIZE)

        # Set y-axis ticks
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        plt.tight_layout()
        if save:
            plt.savefig(title+'.png')
        plt.show()




if __name__ == '__main__':

    ### Part 1 ####

    # 1.1 load data
    df = pd.read_csv('fifa19datastats.csv')


    # 1.2 set seed
    seed = 276842
    N = 10000
    ind = np.random.default_rng(seed=seed).choice(df.shape[0], size=N, replace=False)    # random integers without repetition
    ind.sort()

    df = df.reindex(ind)
    df.ID = df.ID.astype(int)

    #df.to_csv('fifa19datastats_'+ seed.__str__() +'.csv', index=False)
    print(f"Head of SAMPLED DATASET, extracted with random seed {seed}:")
    print(df.head())



    ### Part 2 ###

    # 2.1 Extraction of the skill columns
    skills = df.columns[5:].tolist()                # extract skill's names
    common_cols = df.columns[:5].tolist()
    X = df[skills].values                           # extract skills values dataset


    # Violin plots of skills
    plt.tight_layout(pad = 10)
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 6))
    for i in range(0, 33):
        ax.violinplot(dataset=X[:,i], positions=[i])
    ax.set_title(label='Violin Plot of Skills', fontsize=20, pad=PAD);
    ax.set_xlabel('Attribute', size=LABEL_SIZE)
    ax.set_ylabel('Value', size=LABEL_SIZE)

    #plt.savefig('violin.png')
    plt.show()

    # Looking at std
    skills_std = X.std(axis=0)
    i_max = np.argmax(skills_std)
    i_min = np.argmin(skills_std)
    mean = skills_std.mean()

    print(f"\n\nThe skill with maximum std is {skills[i_max]} with std={skills_std[i_max]:.2f}")
    print(f"The skill with minimum std is {skills[i_min]} with std={skills_std[i_min]:.2f}")
    print(f"Mean of stds: {mean:.2f}")
    print(f"Standard deviation of goalkeeper's attributes {skills_std[-4:].tolist()}")


    # 2.2 Choice of the algorithm and application
    pca = PCA(n_components=0.9, use_mean= True, use_std=False)

    W = pca.fit_transform(X)



    ### Part 3: PC Interpretation ###

    # 3.1 Computation of the num of principal components.
    #     The computation of the number of components needed to have at least 0.9 of variance explained is done
    #     directly in PCA class in the inner function __find_components
    pca.plot_var_explained_ratio(save=False)


    # 3.2 Plot horizontal bargraph of the principal components
    pca.plot_pc_horizontalbargraph(component='all', labels=skills, save=False)



    # 3.4 Scatter plots
    from matplotlib.colors import ListedColormap

    new_skills = ['Goalkepeer', 'Defensive/Offensive', 'Pace', 'Striker/Midfielders', 'Physical/Technical', 'Jumping',
                  'Tactical/Physical']
    player_color_dict = {'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3}
    classes = ['GK', 'DF', 'MF', 'FW']
    map = ListedColormap(['orangered', 'gold', 'limegreen', 'dodgerblue'])
    colors = [player_color_dict[spec] for spec in df['GeneralPosition']]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(12, 8))

    # Plot 1: Goalkepeer - Defensive/Offensive
    scatter = ax1.scatter(W[:, 0], W[:, 1], c=colors, cmap=map, alpha=0.3)
    ax1.set_xlabel(new_skills[0] + ' (pc1)')
    ax1.set_ylabel(new_skills[1] + ' (pc2)')

    # Plot 2: Striker/Midfielder - Physical/Technical
    ax2.scatter(W[:, 3], W[:, 4], c=colors, cmap=map, alpha=0.3)
    ax2.set_xlabel(new_skills[3] + ' (pc4)')
    ax2.set_ylabel(new_skills[4] + ' (pc5)')


    # Plot 3: Defensive/Offensive - Striker/midfielder
    ax3.scatter(W[:, 1], W[:, 3], c=colors, cmap=map, alpha=0.3)
    ax3.set_xlabel(new_skills[1] + ' (pc2)')
    ax3.set_ylabel(new_skills[3] + ' (pc4)')

    # Plot 4: Technical/Physical - Jumping
    ax4.scatter(W[:, 4], W[:, 5], c=colors, cmap=map,alpha=0.3)
    ax4.set_xlabel(new_skills[4] + ' (pc5)')
    ax4.set_ylabel(new_skills[5] + ' (pc6)')

    fig.suptitle("Scatter Plots", fontsize=20)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    #plt.savefig('scatterplots.png')
    plt.show()



    ### Part 4 ###

    # 4.1 Creating dataset with new attributes (components)
    wdataset = df[common_cols].copy()
    temp = pd.DataFrame(W, index=wdataset.index, columns=new_skills)
    wdataset = wdataset.join(temp)

    print("\nHead of NEW DATASET with attributes computed with PCA:")
    print(wdataset.head())

    # 4.2 Saving new dataset
    #wdataset.to_csv('fifa19datastats_w_'+seed.__str__()+'.csv')



    # 4.3 Restoring original dataset and computing mean relative error
    X_tilde = pca.inv_transform(W)

    MRE = lambda Xreal, Xpred: np.sum([la.norm(Xreal[:,j] - Xpred[:,j], 2)/la.norm(Xreal[:,j], 2) for j in range(Xreal.shape[1])])/Xreal.shape[0]
    mean_relative_error = MRE(X, X_tilde)
    print(f"\n   Dataset approximated with the first {pca.n_components} principal components. Mean Relative Error = {mean_relative_error}")


    # 4.3.1 Comparison with sklearn PCA implementation
    from sklearn.decomposition import PCA as skPCA
    skpca = skPCA(n_components=0.9)
    skW = skpca.fit_transform(X)
    skX_tilde = skpca.inverse_transform(skW)

    # error = np.sum([la.norm(X[i] - skX_tilde[i])/la.norm(X[i]) for i in range(X.shape[0])])/X.shape[0]
    error = MRE(X, skX_tilde)
    print(f"   Dataset approximated with sklearn implementation ({skpca.n_components} var explained/{len(skpca.singular_values_)} components). Mean Relative Error = {error}")

    pca_std = PCA(n_components=0.9, use_std=True)
    W_std = pca_std.fit_transform(X)
    X_tilde_std = pca_std.inv_transform(W_std)
    error = MRE(X, X_tilde_std)
    print(f"   Dataset approximated with use of std ({pca_std.var_explained} var explained/{pca_std.n_components} components). Mean Relative Error = {error}")

    # 4.4 Saving approx dataset
    approxdataset = df[common_cols].copy()
    temp = pd.DataFrame(X_tilde, index=wdataset.index, columns=skills)  #
    approxdataset = approxdataset.join(temp)

    print("\nHead of APPROXIMATED DATASET from PCA:")
    print(approxdataset.head())

    #approxdataset.to_csv('fifa19datastats_approx_'+seed.__str__()+'.csv')

