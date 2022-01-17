import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def kernel(A, x):

    return (A @ x) @ A
