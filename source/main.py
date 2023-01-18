import sympy as sy
import numpy as np
import igraph as ig
import random
import math

def mix_directed(P, G): #Make G directed and make a proportion P of the graph not mutially directed
    sample = random.sample(list(range(G.ecount())),math.floor(G.ecount()*P))
    sample = G.es[sample]
    #print(G.ecount())
    G = ig.Graph.as_directed(G,"mutual")
    for e in sample:
        v1_v2 = [e.source, e.target]
        # one of both directions is droped
        random.shuffle(v1_v2)
        G.delete_edges(tuple(v1_v2))
    return G

def check_avg_degree(G):
    k = np.mean(G.as_undirected().degree( mode='all', loops=False))
    print(k)
    return k


def generate_ER(N, P, avgK):
    G = ig.Graph.Erdos_Renyi(n=N, p = avgK/N, directed=False, loops=False)
    if (not G.is_connected()):
        print("Graph not connected")
    check_avg_degree(G)
    G = mix_directed(P,G)
    return G

def generate_RR(N ,P, avgK):
    G = ig.Graph.K_Regular(n=N, k=avgK, directed=False, multiple=False)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P,G)
    return G

def generate_BA(N ,P, avgK):
    #m1 = (-2*(N+1) + math.sqrt(4*(N+1)**2 -4*2*avgK*(N+1)))/(-4)
    M = math.floor(23/36*avgK - 53/30) #Suitable approximation for our input

    G = ig.Graph.Barabasi(N, m = M, outpref=False, directed=False, power=1, zero_appeal=1, implementation='psumtree', start_from=None)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P,G)
    return G

def generate_WS(N ,P, avgK):
    G = ig.Graph.Watts_Strogatz(1, size = N, nei = round(avgK/2), p = 0.1, loops=False, multiple=False)
    if (not G.is_connected()):
        print("Graph not connected")
    G = mix_directed(P,G)
    return G


N = 100
P = 0.5
K =   [20,40,48,50,52,54,56]
mBA = [11,23,28,30,31,32,34]
mBA_dic = dict(zip(K,mBA))

for m1 in mBA:

    G = ig.Graph.Barabasi(N, m = m1, outpref=False, directed=False, power=1, zero_appeal=1, implementation='psumtree', start_from=None)
    check_avg_degree(G)

for i in K:
    G = generate_BA(N,P,i)
    check_avg_degree(G)
