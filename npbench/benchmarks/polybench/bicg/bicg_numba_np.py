import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(A, p, r):

    return r @ A, A @ p
