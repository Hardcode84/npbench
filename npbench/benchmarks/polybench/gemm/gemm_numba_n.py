import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C
