import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class unweighted_LIL:

    def __init__(self, arg1, shape= None, symmetric=False):
        self.data = {}
        self.shape = shape
        self.symmetric = symmetric

        if type(arg1) is list or type(arg1) is np.ndarray:
            self.from_list(arg1, symmetric=symmetric)

            if self.shape is None:
                # self.shape = (max(self.data.keys())+1,max(self.data.keys())+1)
                self.shape = (np.max(arg1)+1, np.max(arg1)+1)

    def __str__(self):
        str1 = ('\n').join(
            [f"Node {k} : {self.data[k]}" if self.data.get(k) is not None else f"Node {k} : []" for k in range(3)])
        str2 = '...'
        str3 = ('\n').join([f"Node {k} : {self.data[k]}" if self.data.get(k) is not None else f"Node {k} : []" for k in
                            range(self.shape[0] - 3, self.shape[0])])

        return ('\n').join([str1,str2,str3])

    # ('\n').join([f"Node {k} : {self.data[k]}" for k in self.data.keys()])

    def get_rows(self):
        return list(map(lambda k: self.data.get(k), np.arange(self.shape[0])))

    def from_list(self, values, symmetric=False):
        '''
        Add values to the matrix from a list of tuples.

        Parameters
        ----------
            values : list of tuples (u,v). Each tuple must contain 2 values: the starting node and the ending node

            symmetric : bool used to define symmetric matrices. Given the entry (i, j), if True add automatically
                        the symmetric entry (j, i)

        '''
        for u, v, in values:
            self.add_value(u, v)

            if symmetric:
                self.add_value(v, u)

    def add_value(self, row, col):
        val = self.data.get(row)

        if val is None:
            self.data[row] = [col]
        else:
            val.append(col)

    def dot(self, v):
        res = np.zeros(self.shape[0])
        for k in self.data.keys():
            res[k] = v[self.data[k]].sum()
        return res

        # return np.array([np.sum(v[self.data[k]]) if self.data.get(k) is not None else 0 for k in range(self.shape[0])])

    def visualize(self, dpi=600, save=False, highlight=None):
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)
        ax.plot(*zip(*[(x,y) for x in self.data.keys() for y in self.data[x]]), markersize=0.005, marker='o', color='white', lw=0, linestyle='')
        ax.set_xlim(-0.5, self.shape[0])
        ax.set_ylim(-0.5, self.shape[0])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title("Sparsity Pattern")
        if highlight:
            ax.scatter(highlight, highlight, c='r', s=5, marker='*', label='node with highest eigenvect. centrality')
            plt.legend(loc='upper left')
        if save:
            plt.savefig('spy.png', dpi = dpi,format='png')

        plt.show()

    # def to_CSR(self):
    #     for k in range(self.shape[0]):
    #         cols = self.data.get(k)
    #
    #         if cols is not None:
    #             cols = sorted(cols)

class LIL():
    def __init__(self, arg1, shape= None):
        self.data = {}
        self.shape = shape

        if type(arg1) is list or type(arg1) is np.ndarray:
            self.from_list(arg1)

        if self.shape is None:
            self.shape = (max(self.data.keys())+1,max(self.data.keys())+1)



    def __str__(self):
        str1 = ('\n').join([f"Node {k} : {self.data[k][:3]} ..." if self.data.get(k) is not None else f"Node {k} : []" for k in range(3)])
        str2 = '...'
        str3 = ('\n').join([f"Node {k} : {self.data[k][:3]}..." if self.data.get(k) is not None else f"Node {k} : []" for k in range(self.shape[0]-3,self.shape[0])])
        return  ('\n').join([str1,str2,str3])

    # def __str__(self):
    #     str = ''
    #     for i in range(np.min(3, len(self.shape[0]))):
    #         str += f"Node {i}: "
    #         row = self.data.get(i)
    #         if row is not None:
    #             str += row[:np.min(len(row), 3)] + '...\n'
    #     str += '...\n'
    #     for i in range(self.shape[0] - 3, self.shape[0])):
    #         str += f"Node {i}: "
    #         row = self.data.get(i)
    #         if row is not None:
    #             str += row[:np.min(len(row), 3)] + '...\n'
    #
    #     return str

    def get_rows(self):
        '''
        Return a list of lists of tuples. Each list correspond to a row, while each tuple in a row correspond
        to a pair (column, value).
        '''
        return list(map(lambda k: self.data.get(k), np.arange(self.shape[0])))



    def from_list(self, values, symmetric=False):
        '''
        Add values to the matrix from a list of tuples.

        Parameters
        ----------
            values : list of tuples (u,v,w). Each tuple must contain 3 values: the starting node,
                     the ending node and the weight

            symmetric : bool used to define symmetric matrices. Given the entry (i, j, w), if True add automatically
                        the symmetric entry (j, i, w)

        '''
        for u, v, w in values:
            self.add_value(u, v, w)
            if symmetric:
                self.add_value(v, u, w)


    def add_value(self, row,col, val):
        '''
        Add a single entry to the matrix. The entry is stored as a tuple (col, value) in the corresponding row
        '''

        v = self.data.get(row)
        if v is None:
            self.data[row] = [(col,val)]
        else:
            v.append((col,val))

    # def dot(self, v):
    #     return [np.sum(np.multiply(v[self.data[k[]]])) if self.data.get(k) is not None else 0 for k in range(self.shape[0])]

    # def visualize(self, dpi=600):
    #     fig = plt.figure(dpi=dpi)
    #     ax = fig.add_subplot(111, facecolor='black')
    #     ax.plot(*zip(*[(x,y) for x in self.data.keys() for y in self.data[x]]), markersize=0.005, marker='o', color='white', lw=0, linestyle='')
    #     ax.set_xlim(-0.5, self.shape[0])
    #     ax.set_ylim(-0.5, self.shape[0])
    #     ax.set_aspect('equal')
    #     ax.invert_yaxis()
    #     ax.set_aspect('equal')
    #     ax.xaxis.set_ticks_position('top')
    #     ax.xaxis.set_label_position('top')
    #     # plt.savefig('spy.png', dpi = 1200,format='png')
    #     plt.show()

    # def to_CSR(self):
    #     for k in range(self.shape[0]):
    #         cols = self.data.get(k)
    #
    #         if cols is not None:
    #             cols = sorted(cols)



class CSR:
    def __init__(self, arg1, shape=None):
        self.val = []
        self.col_ind = []
        self.row_ptr = []
        self.shape = shape

        if type(arg1) is list:
            self.from_list(np.array(arg1))
        if type(arg1) is np.ndarray:
            self.from_list(arg1)

        if self.shape is None:
            self.shape = (np.max(arg1) + 1, np.max(arg1) + 1)

    def __str__(self):
        return ('\n').join(['val: ' + self.val.__str__(), 'cols: ' + self.col_ind.__str__(),'row_ptr: ' +self.row_ptr.__str__()])

    def from_list(self, values):
        df = pd.DataFrame(values)
        df.sort_values(by=[0,1], inplace=True)

        self.val = df[2].to_numpy()
        self.col_ind = df[1].to_numpy(dtype = int)
        self.row_ptr = np.append(0, np.cumsum(np.unique(df[0].to_numpy(dtype = int), return_counts=True)[1]))

    def visualize(self, dpi=600, save=False, highlight=None):

        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)

        pairs = []
        j = 0
        for i in range(self.row_ptr.size - 1):
            for k in range(self.row_ptr[i+1] - self.row_ptr[i]):
                pairs.append((i, self.col_ind[j]))
                j+=1
        pairs = np.array(pairs)
        ax.plot(pairs[:,0], pairs[:,1], markersize=0.05, marker='.', lw=0, linestyle='')
        ax.set_xlim(-0.5, self.shape[0])
        ax.set_ylim(-0.5, self.shape[0])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        if highlight:
            ax.scatter(highlight, highlight, c='r', s=30, marker='.', label='highest centrality score', zorder=10)
            plt.legend(loc='upper left')
        if save:
            plt.savefig('spy.png', dpi=dpi, format='png')

        plt.show()

    def dot(self, v):
        if type(v) is not np.ndarray:
            v = np.array(v)

        res = []
        for ind_i, ind_f in zip(self.row_ptr, self.row_ptr[1:]):
            a = self.val[ind_i:ind_f]
            b = v[self.col_ind[ind_i:ind_f]]
            res.append(np.dot(a,b))
        return np.array(res)




if __name__ == '__main__':
    path = 'edge_file_facebook_python.txt'
    x = [[0,0,3],[2,0,-4],[2,1,1],[1,2,2],[0,3,1],[3,3,1]]
    a = CSR(1, shape=(4,4))
    a.from_list(x)
    print(f"values = {a.val}")
    print(f"colums = {a.col_ind}")
    print(f"rows = {a.row_ptr}")

    print(f"{a.dot([1,2,3,4])}")