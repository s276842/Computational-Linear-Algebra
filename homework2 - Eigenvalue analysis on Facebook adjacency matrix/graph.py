import re
import numpy as np
from sparse import LIL, unweighted_LIL, CSR

import warnings
class Graph:
    def __init__(self, path=None, undirected=True, adj_format=CSR):

        if path != None:
            self.from_file(path, undirected, adj_format=adj_format)
        else:
            self.nodes = 0
            self.num_edges = 0
            self.edges = []


    def add_edge(self, u, v):
        self.adj.add_value(row=u, col=v)
        if hasattr(self, 'edges'):
            pass
        else:
            pass

    def spy(self, save=False, dpi=600, highlight=None):
        self.adj.visualize(dpi=dpi, save=save, highlight=highlight)

    def from_file(self, path, comments='#', delimiter='\t', undirected=True, adj_format=CSR):

        # Extract name of the graph and num of nodes and edges
        with open(path, 'r') as f:
            self.Data = re.sub('[#\n\t]', '', f.readline()).split(':')[1].strip()
            self.nodes, self.num_edges = [int(x.split(':')[1].strip()) for x in re.sub('[#\n\t]', '', f.readline()).split(',')]
            f.close()

        # Extract edges
        self.edges = np.loadtxt(path, dtype=int)
        self.num_edges = self.edges.shape[0]

        if undirected:
            self.edges = np.vstack((self.edges, np.flip(self.edges, axis=1)))


        if adj_format==LIL or adj_format==unweighted_LIL:
            self.adj = unweighted_LIL(self.edges, shape=(self.nodes, self.nodes), symmetric=False)
        elif adj_format==CSR:
            self.edges = np.hstack((self.edges, np.ones(self.edges.shape[0])[:, np.newaxis]))
            self.edges = np.array(self.edges, dtype=int)
            self.adj = CSR(self.edges, shape=(self.nodes, self.nodes))

        # Control if num of edges and edges loaded are coherent
        if self.num_edges > self.edges.shape[0]:
            warnings.warn(f"Read {self.edges.shape[0]} of {self.num_edges} edges")
            self.num_edges = self.edges.shape[0]

        # self.adj = unweighted_LIL(self.edges, shape=(self.nodes, self.nodes), symmetric=True)

        # self.degrees = np.array([len(x) if x is not None else 0 for x in self.adj.get_rows()], dtype=int)
        self.degrees = np.array([np.sum(self.edges[:,0] == k) for k in range(self.nodes)], dtype=int)


    def laplacian_matrix(self, format = LIL):

        if hasattr(self, 'lap'):
            if type(self.lap) is not format:
                return format(self.lap)
            else:
                return self.lap

        edges = self.edges.copy()

        if type(self.adj) is CSR:
            edges[:,2] = -1
        else:
            edges = np.hstack((edges, np.full(edges.shape[0], -1)[:, np.newaxis]))

        diagonal = np.array([(i,i, self.degrees[i]) for i in range(self.nodes)])
        self.lap = format(np.vstack((edges,diagonal)), shape=(self.nodes, self.nodes))

        return self.lap



    def norm_laplacian_matrix(self, format = CSR):

        if hasattr(self, 'norm_lap'):
            if type(self.norm_lap) is not format:
                return format(self.norm_lap)
            else:
                return self.norm_lap

        c = lambda i,j: -np.divide(1,np.sqrt(self.degrees[i]*self.degrees[j])) if (self.degrees[i]*self.degrees[j] != 0) else 0

        # edges = np.hstack((self.edges), np.array(list(map(c, G.edges))))
        if type(self.adj) is CSR:
            edges = [(i,j,c(i,j)) for i,j,_ in self.edges]
        else:
            edges = [(i, j, c(i, j)) for i, j in self.edges]
        diagonal = [(i, i, 1) for i in range(self.nodes)]
        self.norm_lap = CSR(np.vstack((edges, diagonal)), shape=(self.nodes, self.nodes))
        return self.norm_lap


if __name__ == '__main__':
    stp = \
        '''
from graph import Graph
import numpy as np
G = Graph(path='edge_file_facebook_python.txt')
lap = G.norm_laplacian_matrix()'''

    G = Graph(path='edge_file_facebook_python.txt')
    lap = G.laplacian_matrix()
    G.norm_laplacian_matrix()
    # G.spy(highlight=1912)

    import timeit
    N = 100
    print(timeit.timeit('G.adj.dot(np.random.rand(G.nodes))', setup=stp, number=N))
    print(timeit.timeit('lap.dot(np.random.rand(G.nodes))', setup=stp, number=N))