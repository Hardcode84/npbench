import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def kernel(alpha, beta, A, B, C, D):

    D[:] = alpha * A @ B @ C + beta * D
