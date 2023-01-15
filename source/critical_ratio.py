import sympy as sy
import numpy as np
import igraph as ig
import random
import math

# ejemplo de grafo
random.seed(0)
N = 100
K = 50 # desired average degree
P = 0.5 # proportion of directed edges
G = ig.Graph.Erdos_Renyi(n=N, p=K/N, directed=False, loops=False)
# print del average degree (comparar con K)
#print(np.mean(np.array(G.get_adjacency().data, np.float32).sum(axis=0)))
sample = random.sample(list(range(G.ecount())),math.floor(G.ecount()*P))
sample = G.es[sample]
#print(G.ecount())
G = ig.Graph.as_directed(G,"mutual")
for e in sample:
    v1_v2 = [e.source, e.target]
    # one of both directions is droped
    random.shuffle(v1_v2)
    G.delete_edges(tuple(v1_v2))
#print(G.ecount())
A = np.array(G.get_adjacency().data, np.float32)

# matriz de p_ij = w_ji / sum_l(w_li)
tmp = A.sum(axis=0)
np.reciprocal(tmp, where=tmp > 0.0, out=tmp)
p = (A.dot(np.diagflat(tmp))).T

# sistema de equaciones para encontrar pi_i
coeff = p.T-np.eye(N)
coeff = coeff[1:,:]
coeff = np.vstack((coeff, np.ones(N)))
_, inds = sy.Matrix(coeff).T.rref()
coeff = np.array(coeff[list(inds)])
print(np.linalg.matrix_rank(coeff)==N)
const = np.zeros(N)
const[-1] = 1
pi = np.linalg.solve(coeff, const)

print(np.linalg.norm(coeff.dot(pi)-const))
print(pi)