import numba as nb

if False:
    jit = nb.jit
    njit = np.njit
    vectorize = nb.vectorize
else:
    import numba_dpcomp
    jit = numba_dpcomp.jit
    njit = numba_dpcomp.njit
    vectorize = numba_dpcomp.vectorize

prange = nb.prange
