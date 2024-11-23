import numpy as np
from sklearn.utils.extmath import randomized_svd

error_eps = 1e-8

def generate_random_matrix(m, n, a, b):
    return np.array([[np.random.uniform(a,b) for _ in range (0,m)] for _ in range (0,n)])


def perform_trunc_svd(A, k=None):
    if k is None:
        k = min(A.shape[0], A.shape[1])
    elif k < 0:
        k = min(A.shape[0], A.shape[1])+k    

    U, D, V_transposed = randomized_svd(A, n_components=k)
    D = D.reshape((D.shape[0], -1))*np.eye(D.shape[0])
    return U, D, V_transposed 


def test_svd():
    m, n = 400, 100
    a, b = 0, 10
    A = generate_random_matrix(m, n, a, b)
    U, D, V_transposed = perform_trunc_svd(A,-1)
    print(U.shape, D.shape, V_transposed.shape)
    built_A = U@D@V_transposed
    print(A.shape, built_A.shape)
    print(A[:4, :4])
    print(built_A[:4, :4])
    print(np.sum(np.abs(A-built_A) < error_eps) == A.shape[0]*A.shape[1])


if __name__ == "__main__":
    test_svd()         