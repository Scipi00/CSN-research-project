import igraph as ig
import numpy as np
from math import floor
from random import shuffle, sample, seed
from time import perf_counter
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
    ti = perf_counter()
    for i in range(N):
        pi_i = pi[i]
        for j in range(N):
            if i != j:
                p_ij = p[i, j]
                for k in range(N):
                    tau_jk = tau[j, k]
                    if j != k:
                        u0 += pi_i*p_ij*tau_jk*A_h[k, j]
                    p_ik = p[i, k]
                    for l in range(N):
                        if i != k and k != l:
                            if j != k:
                                v2 += pi_i*p_ij*p_ik*tau_jk*A_h[k, l]
                            if j != l:
                                u2 += pi_i*p_ij*p_ik*tau[j, l]*A_h[l, k]
    ti = perf_counter() - ti
    print(ti)

    return v2/(u2 - u0)


def mix_directed(P, G):  # Make G directed and make a proportion P of the graph not mutially directed
    S = sample(list(range(G.ecount())), floor(G.ecount()*P))
    S = G.es[S]
    G = ig.Graph.as_directed(G, "mutual")
    for e in S:
        v1_v2 = [e.source, e.target]
        # one of both directions is droped
        shuffle(v1_v2)
        G.delete_edges(tuple(v1_v2))
    return G


def avg_degree(G):
    return np.mean(G.as_undirected().degree(mode='all', loops=False))


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
    # m1 = (-2*(N+1) + math.sqrt(4*(N+1)**2 -4*2*avgK*(N+1)))/(-4)
    M = floor(23/36*avgK - 531/300)  # Suitable approximation for our input
    G = ig.Graph.Barabasi(N, m=M, outpref=False, directed=False, power=1,
                          zero_appeal=1, implementation='psumtree', start_from=None)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P, G)
    return G


def generate_WS(N, P, avgK):
    G = ig.Graph.Watts_Strogatz(1, size=N, nei=round(
        avgK/2), p=0.1, loops=False, multiple=False)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P, G)
    return G


def rewiring(G, r):
    # retrieve degree from r
    # H deepcopy G
    # for all edged (i,j) in G and while H connected:
    # i is origin j is target
    # if toss coin with prob r rewire
    # generate aproppiate neighbourhood of i in H
    # retrieve score degree of neighbours in G
    # normalize degrees
    # select new random origin
    return G


def main_experiment_replication():
    # global experiment parameters
    N = 100
    P = [1., .9, .67, .5, .1, .0]
    K = [20, 40, 48, 50, 52, 54, 56]
    # mBA = [11,23,28,30,31,32,34]
    downstream = False  # downstream or upstream

    def experiment_ER(P_i, K_i):
        G = generate_ER(N, P_i, K_i)
        print(avg_degree(G), K_i)
        H = G.copy()
        if not downstream:
            H.reverse_edges()
        return critical_benefit_cost(G, H)

    ti = perf_counter()
    results_ER = Parallel(n_jobs=-1, prefer="threads")(delayed(experiment_ER)
                                                       (P_i, K_i) for P_i in P for K_i in K)
    results_ER = np.array(results_ER).reshape(len(P), len(K))
    print(results_ER)
    np.save('data_ER.npy', results_ER)  # save
    np.savetxt("data_ER.csv", results_ER, fmt='%10.1f', delimiter=",")
    # new_num_arr = np.load('data.npy') # load
    ti = perf_counter() - ti
    print("Total table time ER" + str(ti))

    def experiment_RR(P_i, K_i):
        G = generate_RR(N, P_i, K_i)
        print(avg_degree(G), K_i)
        H = G.copy()
        if not downstream:
            H.reverse_edges()
        return critical_benefit_cost(G, H)
    ti = perf_counter()
    results_RR = Parallel(n_jobs=-1, prefer="threads")(delayed(experiment_RR)
                                                       (P_i, K_i) for P_i in P for K_i in K)
    results_RR = np.array(results_RR).reshape(len(P), len(K))
    print(results_RR)
    np.save('data_RR.npy', results_RR)  # save
    np.savetxt("data_RR.csv", results_RR, fmt='%10.1f', delimiter=",")
    # new_num_arr = np.load('data.npy') # load
    ti = perf_counter() - ti
    print("Total table time RR" + str(ti))

    def experiment_BA(P_i, K_i):
        G = generate_BA(N, P_i, K_i)
        print(avg_degree(G), K_i)
        H = G.copy()
        if not downstream:
            H.reverse_edges()
        return critical_benefit_cost(G, H)

    ti = perf_counter()
    results_BA = Parallel(n_jobs=-1, prefer="threads")(delayed(experiment_BA)
                                                       (P_i, K_i) for P_i in P for K_i in K)
    results_BA = np.array(results_BA).reshape(len(P), len(K))
    print(results_BA)
    np.save('data_BA.npy', results_BA)  # save
    np.savetxt("data_BA.csv", results_BA, fmt='%10.1f', delimiter=",")
    # new_num_arr = np.load('data.npy') # load
    ti = perf_counter() - ti
    print("Total table time BA" + str(ti))

    def experiment_WS(P_i, K_i):
        G = generate_WS(N, P_i, K_i)
        print(avg_degree(G), K_i)
        H = G.copy()
        if not downstream:
            H.reverse_edges()
        return critical_benefit_cost(G, H)

    ti = perf_counter()
    results_WS = Parallel(n_jobs=-1, prefer="threads")(delayed(experiment_WS)
                                                       (P_i, K_i) for P_i in P for K_i in K)
    results_WS = np.array(results_WS).reshape(len(P), len(K))
    print(results_WS)
    np.save('data_WS.npy', results_WS)  # save
    np.savetxt("data_WS.csv", results_WS, fmt='%10.1f', delimiter=",")
    # new_num_arr = np.load('data.npy') # load
    ti = perf_counter() - ti
    print("Total table time WS" + str(ti))


def main_experiment_celebrities():
    N = 100
    P = [1., .9, .67, .5, .1, .0]
    K = [20, 40, 48, 50, 52, 54, 56]
    # mBA = [11,23,28,30,31,32,34]

    def experiment_celeb(P_i, K_i):
        G = generate_RR(N, P_i, K_i)
        H = generate_BA(N, P_i, K_i)
        print(avg_degree(G), avg_degree(H), K_i)
        return critical_benefit_cost(G, H)

    ti = perf_counter()
    results_celeb = Parallel(n_jobs=-1, prefer="threads")(delayed(experiment_celeb)
                                                          (P_i, K_i) for P_i in P for K_i in K)
    results_celeb = np.array(results_celeb).reshape(len(P), len(K))
    print(results_celeb)
    np.save('data_celeb.npy', results_celeb)  # save
    np.savetxt("data_celeb.csv", results_celeb, fmt='%10.1f', delimiter=",")
    # new_num_arr = np.load('data.npy') # load
    ti = perf_counter() - ti
    print("Total table time celeb" + str(ti))


def main_experiment_communities():
    N = 100
    P = [1., .9, .67, .5, .1, .0]
    K = [6, 8, 10, 15]

    def experiment_comm(P_i, K_i):
        G = generate_RR(100, P_i, K_i)
        while not G.is_connected():
            G = generate_RR(100, P_i, K_i)
        H = G.as_undirected()
        communities = H.community_fastgreedy().as_clustering()
        H = community_to_clique(H,communities)
        H = mix_directed(P_i,H)
        print(avg_degree(G), avg_degree(H), K_i)
        return critical_benefit_cost(G, H)

    ti = perf_counter()
    results_comm = Parallel(n_jobs=4, prefer="threads")(delayed(experiment_comm)
                                                          (P_i, K_i) for P_i in P for K_i in K)
    results_comm = np.array(results_comm).reshape(len(P), len(K))
    print(results_comm)
    np.save('data_comm.npy', results_comm)  # save
    np.savetxt("data_comm.csv", results_comm, fmt='%10.1f', delimiter=",")
    # new_num_arr = np.load('data.npy') # load
    ti = perf_counter() - ti
    print("Total table time comm" + str(ti))

def community_to_clique(H,communities):
    for community in communities:
        edges_clique = []
        for i in community:
            for j in community:
                if i < j:
                    edges_clique.append((i,j))
        H.add_edges(edges_clique)
    return H.simplify()

if __name__ == "__main__":
    # main_experiment_replication()
    # main_experiment_celebrities()
    main_experiment_communities()