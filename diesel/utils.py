""" Helper functions for the DIESEL package.

"""
import dask.array as da


CHUNK_REDUCTION_FACTOR = 4

def find_closest_multiple(x, base):
    closest_lower = int(base * round(float(x)/base))
    if closest_lower == x: return x
    else: return closest_lower + base

def cholesky_invert(A):
    """ Computes the (lower) Cholesky factor and the inverse 
    of a symmetric positive definite matrix using Cholesky decomposition 
    and backward substitution.

    Parameters
    ----------
    A: dask.array

    Returns
    -------
    L, A_inv: dask.array (lazy)
        Lower Cholesky factor and inverse of the input matrix.

    """
    # Note that the daks cholesky implementation requires square chunks.
    # Hence, to keep chunks of a manageable size, one possible trick is to make 
    # R into a matrix wiht shape divisible by CHUNK_REDUCTION_FACTOR, to have CHUNK_REDUCTION_FACTOR chunks along each 
    # dimension.
    # Appending with identity matrix (in block diag fashion) allows us 
    # to recover the original Cholesky decomposition from the one of the 
    # augmented matrix.

    # If small enough, then use only one chunk.
    if A.shape[0] < CHUNK_REDUCTION_FACTOR - 1:
        chunk_size = A.shape[0]
        A_rechunked = A.rechunk(chunk_size)
        shape_diff = 0

    # Else append identity matrix to get a shape that is 
    # divisible by CHUNK_REDUCTION_FACTOR.
    else:
        new_shape = find_closest_multiple(A.shape[0], CHUNK_REDUCTION_FACTOR)
        if new_shape > 0:
            shape_diff = new_shape - A.shape[0]
            A_rechunked = da.vstack(
                    [
                        da.hstack([A, da.zeros((A.shape[0], shape_diff))]),
                        da.hstack([da.zeros((shape_diff, A.shape[0])), da.eye(shape_diff)])
                    ]
            )
        chunk_size = int(new_shape / CHUNK_REDUCTION_FACTOR)
        A_rechunked = A_rechunked.rechunk(chunk_size)

    R = da.linalg.cholesky(A_rechunked, lower=False)
    R_inv = da.linalg.solve_triangular(R, da.linalg.eye(R.shape[0], chunks=chunk_size), lower=False)

    # Extract the part of interest for us.
    if shape_diff > 0:
        R = R[:-shape_diff, :-shape_diff]
        R_inv = R_inv[:-shape_diff, :-shape_diff]
    return da.transpose(R), da.matmul(R_inv, da.transpose(R_inv))
