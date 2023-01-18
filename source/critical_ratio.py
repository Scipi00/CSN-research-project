import random
import math
import numpy as np
import igraph as ig

# ejemplo de grafo
random.seed(0)
N = 100
K = 50  # desired average degree
P = 0.5  # proportion of directed edges
reps = 1

for rep in range(reps):

    G = ig.Graph.Erdos_Renyi(n=N, p=K/N, directed=False, loops=False)
    # print del average degree (comparar con K)
    #print(np.mean(np.array(G.get_adjacency().data, np.float32).sum(axis=0)))
    sample = random.sample(list(range(G.ecount())), math.floor(G.ecount()*P))
    sample = G.es[sample]
    # print(G.ecount())
    G = ig.Graph.as_directed(G, "mutual")
    for e in sample:
        v1_v2 = [e.source, e.target]
        # one of both directions is droped
        random.shuffle(v1_v2)
        G.delete_edges(tuple(v1_v2))
    # print(G.ecount())
    A = np.array(G.get_adjacency().data, np.float64)

    # matriz de p_ij = w_ji / sum_l(w_li)
    tmp = A.sum(axis=0)
    np.reciprocal(tmp, where=tmp > 0.0, out=tmp)
    p = (A.dot(np.diagflat(tmp))).T

    # sistema de equaciones para encontrar pi_i
    coeff = p.T-np.eye(N)
    coeff = coeff[1:, :]  # one of the equations is always linearly dependent
    coeff = np.vstack((coeff, np.ones(N)))
    const = np.zeros(N)
    const[-1] = 1
    pi = np.linalg.solve(coeff, const)
    
    # sistema de equaciones para encontrar tau_ij
    coeff = np.zeros((N*N,N*N))
    const = np.ones(N*N)
    for i in range(0,N):
        const[i*N+i] = 0
        for j in range(0,N):
            m = np.array([])
            if i == j:
                m = p*0.5 - np.eye(N)
                vec = np.zeros((1,N))
                #print(vec)
                vec[:,i] = 1
                m[i,:] = vec
            else:
                m = np.eye(N)*p[i,j]
                m[i,i] = 0
            coeff[i*N:i*N+N, j*N:j*N+N] = m
    #print(coeff)
    
    #print(np.linalg.matrix_rank(coeff))

    tau = np.linalg.solve(coeff, const)
    tau = np.reshape(tau,(N,N))
    print(tau)

