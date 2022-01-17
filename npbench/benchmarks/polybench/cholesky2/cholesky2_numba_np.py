import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
