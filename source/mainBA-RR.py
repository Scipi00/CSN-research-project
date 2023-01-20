import math
import random
import time
import igraph as ig
import numpy as np
from joblib import Parallel, delayed


def critical_benefit_cost(G, H):
    N = G.vcount()
    A_g = np.array(G.get_adjacency().data, np.float64)
    # matriz de p_ij = w_ji / sum_l(w_li)
    tmp = A_g.sum(axis=0)
    np.reciprocal(tmp, where=tmp > 0.0, out=tmp)
    p = (A_g.dot(np.diagflat(tmp))).T

    # sistema de equaciones para encontrar pi_i
    coeff = p.T-np.eye(N)
    coeff = coeff[1:, :]  # one of the equations is always linearly dependent
    coeff = np.vstack((coeff, np.ones(N)))
    const = np.zeros(N)
    const[-1] = 1
    pi = np.linalg.solve(coeff, const)

    # sistema de equaciones para encontrar tau_ij
    coeff = np.zeros((N*N, N*N))
    const = -1*np.ones(N*N)
    for i in range(N):
        const[i*N+i] = 0
        for j in range(N):
            m = np.array([])
            if i == j:
                m = p*0.5 - np.eye(N)
                vec = np.zeros((1, N))
                # print(vec)
                vec[:, i] = 1
                m[i, :] = vec
            else:
                m = np.eye(N)*p[i, j]*0.5
                m[i, i] = 0
            coeff[i*N:i*N+N, j*N:j*N+N] = m

    tau = np.linalg.solve(coeff, const)
    tau = np.reshape(tau, (N, N))

    # computing the critical cost
    A_h = np.array(H.get_adjacency().data, np.float64)
    u0 = 0
    u2 = 0
    v2 = 0
    print("start")
    ti = time.perf_counter()
    for i in range(N):
        for j in range(N):
            if i != j:
                for k in range(N):
                    if j != k:
                        u0 += pi[i] * p[i, j] * tau[j, k] * A_h[k, j]
                    for l in range(N):
                        if i != k and k != l:
                            if j != k:
                                v2 += pi[i] * p[i, j] * \
                                    p[i, k] * tau[j, k] * A_h[k, l]
                            if j != l:
                                u2 += pi[i] * p[i, j] * \
                                    p[i, k] * tau[j, l] * A_h[l, k]
    ti = time.perf_counter() - ti
    print(ti)

    return v2/(u2 - u0)


def mix_directed(P, G):  # Make G directed and make a proportion P of the graph not mutially directed
    sample = random.sample(list(range(G.ecount())), math.floor(G.ecount()*P))
    sample = G.es[sample]
    # print(G.ecount())
    G = ig.Graph.as_directed(G, "mutual")
    for e in sample:
        v1_v2 = [e.source, e.target]
        # one of both directions is droped
        random.shuffle(v1_v2)
        G.delete_edges(tuple(v1_v2))
    return G


def check_avg_degree(G):
    k = np.mean(G.as_undirected().degree(mode='all', loops=False))
    print(k)
    return k


def generate_ER(N, P, avgK):
    G = ig.Graph.Erdos_Renyi(n=N, p=avgK/N, directed=False, loops=False)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P, G)
    return G


def generate_RR(N, P, avgK):
    G = ig.Graph.K_Regular(n=N, k=avgK, directed=False, multiple=False)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P, G)
    return G


def generate_BA(N, P, avgK):
    #m1 = (-2*(N+1) + math.sqrt(4*(N+1)**2 -4*2*avgK*(N+1)))/(-4)
    M = math.floor(23/36*avgK - 531/300)  # Suitable approximation for our input
    G = ig.Graph.Barabasi(N, m=M, outpref=False, directed=False, power=1,
                          zero_appeal=1, implementation='psumtree', start_from=None)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P, G)
    return G


def generate_WS(N, P, avgK):
    G = ig.Graph.Watts_Strogatz(1, size=N, nei=round(avgK/2), p=0.1, loops=False, multiple=False)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P, G)
    return G


def main():
    N = 100
    P = [1.,.9,.67,.5,.1,.0]
    K = [20,40,48,50,52,54,56]
    #mBA = [11,23,28,30,31,32,34]

    def experiment(P_i, K_i):
        G = generate_RR(N, P_i, K_i)
        H = generate_BA(N, P_i, K_i)
        check_avg_degree(G)
        check_avg_degree(H)
        return critical_benefit_cost(G, H)

    ti = time.perf_counter()
    print(ti)

    results = Parallel(n_jobs=1, prefer="threads")(delayed(experiment)
                                                    (P_i, K_i) for P_i in P for K_i in K)
    results = np.array(results).reshape(len(P), len(K))
    print(results)
    ti =  time.perf_counter() - ti
    print("Total table time"+ str(ti))


if __name__ == "__main__":
    main()
