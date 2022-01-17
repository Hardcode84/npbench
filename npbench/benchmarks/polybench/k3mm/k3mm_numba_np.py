import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(A, B, C, D):

    return A @ B @ C @ D
