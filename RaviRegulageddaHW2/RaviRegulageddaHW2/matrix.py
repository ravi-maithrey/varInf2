import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse


# defining the matrix decomposition
# this is a generator i.e, it 'yields' a value after 100 iterations
# that makes it easy for us to store and plot the performance metrics
# i used black to format my code. if it looks strange blame that :)
def matrix_factorization(m, rank):
    """Performs matrix factorization according to Lim & Teh, 2007. Takes in m: a sparse matrix and rank"""
    # initalize the variables
    # shape of the matrix
    # this assumes we have a 2D matrix (which we do)

    i_m = m.shape[0]
    j_m = m.shape[1]
    d = rank

    # variables and constants
    sigma_2 = np.ones(d)
    rho_2 = np.ones(d) / d
    tau_2 = 1
    u = []
    v = []
    # notation from paper
    S, phi, psi, t = [], [], [], []

    # intializing our variables with ones and random values according to the question
    for i in range(i_m):
        phi.append(np.eye(d))
        u.append(np.random.normal(0, 1, d))
    for j in range(j_m):
        psi.append(np.eye(d))
        S.append(np.diag(1 / rho_2))
        t.append(np.zeros(d))
        v.append(np.random.normal(0, 1, d))

    # make sure everything is an numpy array so there are no runtime/indexing errors
    phi = np.array(phi)
    psi = np.array(psi)
    S = np.array(S)
    t = np.array(t)
    u = np.array(u)
    v = np.array(v)

    norm_u = 0
    norm_v = 0

    N = []
    for i in range(i_m):
        # from the paper N[i] is 1 if we observe m_ij. Since we observe a 0 or 1, we just append it
        N.append(scipy.sparse.find(m[i])[1])
    # all the places in the sparse matrix where there are elements
    ob = scipy.sparse.find(m)

    # we perform our EM now
    # iterating 100 times
    for _ in range(100):
        # E step
        # Q(U)
        for i in range(0, i_m):
            # this stores the matrix product (outer product in numpy)
            outer = np.zeros((d, d))
            N_i = N[i]
            for j in N_i:
                outer += np.outer(v[j], v[j])
            phi[i] = np.linalg.inv(
                np.diag(1 / sigma_2) + (psi[N_i].sum(0) + outer) / tau_2
            )
            mtplr = ((m[i, N_i] * (v[N_i])) / tau_2).sum(0)
            u[i] = phi[i].dot(mtplr)
            S[N_i] += (phi[i] + np.outer(u[i], u[i])) / tau_2
            t[N_i] += np.outer(csr_matrix.toarray(m[i, N_i]), (u[i])) / tau_2

        # Q(v)
        psi = np.linalg.inv(S)
        for j in range(j_m):
            v[j] = psi[j].dot(t[j])

        # M step
        for l in range(d):
            sigma_2[l] = ((phi[:, l, l] + u[:, l] ** 2).sum()) / (i_m - 1)

        K = len(ob[1])
        Tr = 0  # trace
        for i, j in np.array([ob[0], ob[1]]).T:
            A = phi[i] + np.outer(u[i], u[i])
            B = psi[j] + np.outer(v[j], v[j])
            Tr += np.trace(A.dot(B))

        tau_2 = (
            (
                (ob[2] ** 2) - (2 * ob[2] * np.einsum("ij,ij->i", u[ob[0]], v[ob[1]]))
            ).sum()
            + Tr
        ) / (K - 1)

        new_norm_u = np.linalg.norm(u)
        new_norm_v = np.linalg.norm(v)

        if abs(new_norm_u - norm_u) < 0.01 or abs(new_norm_v - norm_v) < 0.01:
            # early stopping
            break
        else:
            norm_u, norm_v = new_norm_u, new_norm_v
        yield np.array(u), np.array(v)
