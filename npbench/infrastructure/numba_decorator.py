import numba as nb

if False:
    jit = nb.jit
    njit = np.njit
    vectorize = nb.vectorize
else:
    import numba_mlir
    jit = numba_mlir.jit
    njit = numba_mlir.njit
    vectorize = numba_mlir.vectorize

prange = nb.prange
