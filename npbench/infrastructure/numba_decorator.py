import numba as nb
import os

_decorator = os.environ["NPBENCH_NUMBA_DECORATOR"]

if _decorator == "numba":
    jit = nb.jit
    njit = nb.njit
    vectorize = nb.vectorize
elif _decorator == "numba-mlir":
    import numba_mlir
    jit = numba_mlir.jit
    njit = numba_mlir.njit
    vectorize = numba_mlir.vectorize
elif _decorator == "replace-parfor":
    import numba_mlir
    def jit(*args, **kwargs):
        return numba_mlir.jit(*args, *kwargs, replace_parfors=True)

    def njit(*args, **kwargs):
        return numba_mlir.njit(*args, *kwargs, replace_parfors=True)

    vectorize = nb.vectorize
else:
    assert False, f"Invalid decorator {_decorator}"

prange = nb.prange
