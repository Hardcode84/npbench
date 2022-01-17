import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x
