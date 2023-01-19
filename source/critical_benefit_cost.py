import numpy as np


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
    for i in range(N):
        for j in range(N):
            if i != j:
                for l in range(N):
                    if j != l:
                        u0 += pi[i] * p[i, j] * tau[j, l] * A_h[l, j]
    u2 = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                for k in range(N):
                    if i != k:
                        for l in range(N):
                            if j != l and l != k:
                                u2 += pi[i] * p[i, j] * p[i, k] * tau[j, l] * A_h[l, k]
    v2 = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                for k in range(N):
                    if i != k and j != k:
                        for l in range(N):
                            if k != l:
                                v2 += pi[i] * p[i, j] * p[i, k] * tau[j, k] * A_h[k, l]

    return v2/(u2 - u0)
