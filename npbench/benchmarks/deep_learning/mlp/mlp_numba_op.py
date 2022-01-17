import numpy as np
import npbench.infrastructure.numba_decorator as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def relu(x):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x
