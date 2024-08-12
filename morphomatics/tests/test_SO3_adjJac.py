import timeit
from jax import random
from jax.scipy.linalg import expm

from morphomatics.manifold import SO3
from morphomatics.manifold.util import multiskew

if __name__ == '__main__':

    k = 10000
    # make 2 pseudo random number generator keys
    key, subkey = random.split(random.PRNGKey(42))
    R = expm(multiskew(random.normal(subkey, (k, 3, 3))))
    key, subkey = random.split(key)
    Q = expm(multiskew(random.normal(subkey, (k, 3, 3))))
    key, subkey = random.split(key)
    X = multiskew(random.normal(subkey, (k, 3, 3)))

    M = SO3(k)

    # jit compile
    Y = M.metric.adjJacobi(R, Q, 0.1, X)
    print(Y[0])
    print(M.metric.eval_adjJacobi(R, Q, 0.1, X)[0])
    # benchmark
    print(timeit.timeit(lambda: M.metric.adjJacobi(R, Q, .1, X).block_until_ready(), number=100))

# if __name__ == '__main__':
#     import timeit
#     from jax import random
#
#     cpulogm = lambda O: np.einsum('ijk,...k', versor2skew, Rotation.from_matrix(O).as_rotvec())
#
#     key, subkey = random.split(random.PRNGKey(0))
#     S = random.normal(subkey, (10000, 3, 3))
#
#     S = multiskew(S)
#     R = expm(S)
#     print(R[0])
#
#     X = logm(R)
#     print(X[0])
#     print(cpulogm(R[0]))
#
#     print(timeit.timeit(lambda: logm(R).block_until_ready(), number=100))
#     R_ = np.array(R)
#     print(timeit.timeit(lambda: cpulogm(R_), number=100))
#
#     # print(timeit.timeit(lambda: expm(S).block_until_ready(), number=100))
#     #
#     # skew2versor = .5 * versor2skew
#     # cpuexpm = lambda X: Rotation.from_rotvec(np.tensordot(X, skew2versor, axes=([1, 2], [1, 2]))).as_matrix()
#     # R_ = cpuexpm(S)
#     # print(R_[0])
#     #
#     # S_ = np.array(S)
#     # print(timeit.timeit(lambda: cpuexpm(S_), number=100))