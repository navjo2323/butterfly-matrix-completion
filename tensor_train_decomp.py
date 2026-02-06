import numpy as np
import tensorly as tl
#from tensorly.decomposition import matrix_product_state
from tensorly.decomposition import tensor_train as matrix_product_state
import numpy.linalg as la
import time
from dependencies.butterfly_tensor_train import reconstruct_sparse_butterfly, tensor_train_ALS_solve, tensor_train_gradient, compute_error_sparse
from dependencies.butterfly_tensor_train import (
    FastTTComputer, _make_mid_list,
    tensor_train_ALS_solve_fast, tensor_train_gradient_fast,
    compute_error_sparse_fast, get_available_memory,
    sort_inds_and_T_short
)
import scipy.linalg as sla

import numpy as np



def sort_inds_and_T(tuples, T, k = None):
    """
    Sorts a numpy array of tuples according to kth index as above
    if k is not given, do the sort lexicographically
    """
    if k is None:
        sorted_indices = np.lexsort(np.fliplr(tuples).T)
    else:
        sorted_indices = np.argsort(tuples[:, k])

    sorted_array = tuples[sorted_indices]
    reordered_T = T[sorted_indices]

    return sorted_array,  reordered_T

def convert_matrix_to_QTT_indices(L, c, indices):
    indices = np.array(indices)
    
    I = indices[:, 0]
    J = indices[:, 1]
    
    # Calculate ind_is and ind_js by integer division
    ind_is = I // c
    ind_js = J // c
    
    # Convert ind_is and ind_js to binary representations using bit-shifting
    ind_is_binary = ((ind_is[:, None] >> np.arange(L - 1, -1, -1)) & 1).astype(np.int32)
    ind_js_binary = ((ind_js[:, None] >> np.arange(L - 1, -1, -1)) & 1).astype(np.int32)
    
    # Generate block indices using I and J
    block_indices_i = I % c
    block_indices_j = J % c

    # Calculate the first index from reshaped (block_indices_i, block_indices_j) matrix
    first_index = block_indices_i * c + block_indices_j
    
    # Create a NumPy array of zeros with shape (number_of_tuples, L + 1)
    number_of_tuples = len(indices)
    result = np.zeros((number_of_tuples, L + 1), dtype=int)
    
    # Fill the first column with first_index
    result[:, 0] = first_index
    
    # Generate combined binary indices for each level and fill in result[:, i]
    for n in range(L):
        # Concatenate the binary bits
        combined_bits = np.hstack((ind_is_binary[:, n:n+1], ind_js_binary[:, n:n+1]))
        
        # Convert combined bits to integer
        part = np.dot(combined_bits, 1 << np.arange(combined_bits.shape[1] - 1, -1, -1))
        
        # Fill the corresponding column in result
        result[:, n + 1] = part
    
    # Convert the result array to a list of tuples
    s = time.time()
    encoded_list = np.array([tuple(row) for row in result])
    e= time.time()

    print('time array loop',e-s)
    
    return encoded_list


def reshape_matrix_to_tensor_QTT(M, L, c):
    # Generate all possible indices for the matrix
    N = c * (2 ** L)
    row_col_indices = np.indices((N, N)).reshape(2, -1).T
    
    # Get tensor indices using the provided function
    s = time.time()
    tensor_indices = convert_matrix_to_QTT_indices(L, c, row_col_indices)
    e = time.time()

    print('total time converting indices',e-s)
    
    # Determine the shape of the tensor: (c^2, 2^2, ..., 2^2) with L twos
    tensor_shape = [c**2] + [4] * L
    
    # Initialize an empty tensor with the determined shape

    print('data type of M',M.dtype)
    tensor = np.zeros(tensor_shape, dtype=M.dtype)
    

    tensor[tuple(tensor_indices.T)] = M[row_col_indices[:, 0], row_col_indices[:, 1]]
    return tensor

# def reshape_matrix_to_tensor(M,L,c):



# # Example usage:
# c = 2  # Example parameter
# L = 2  # Example level
# M = np.random.randn(c * (2 ** L), c * (2 ** L))  # Example matrix

# # Reshape matrix to tensor using the function
# tensor_result = reshape_matrix_to_tensor(M, L, c)


# Output the combined indices result
#print(combined_indices_result)

def tensor_train_decomposition_low(left_mat, right_mat, L, c, ranks):
    mat = left_mat@right_mat.T

    T = reshape_matrix_to_tensor_QTT(mat, L, c)

    factors = matrix_product_state(tl.tensor(T), rank=ranks)
    # Calculate reconstruction error

    reconstructed_tensor = tl.tt_to_tensor(factors)

    # Calculate the reconstruction error
    error = tl.norm(tl.tensor(T) - reconstructed_tensor) / tl.norm(tl.tensor(T))

    print("Reconstruction Error:", error)

    numpy_factors = [np.squeeze(tl.to_numpy(factor)) for factor in factors]

    numpy_factors[1:-1] = [factor.transpose(1, 0, 2) for factor in numpy_factors[1:-1]]

    return numpy_factors


def tensor_train_decomposition(mat, L, c, ranks):

    T = reshape_matrix_to_tensor_QTT(mat, L, c)

    factors = matrix_product_state(tl.tensor(T), rank=ranks)
    # Calculate reconstruction error

    reconstructed_tensor = tl.tt_to_tensor(factors)

    # Calculate the reconstruction error
    error = tl.norm(tl.tensor(T) - reconstructed_tensor) / tl.norm(tl.tensor(T))

    print("Reconstruction Error:", error)

    numpy_factors = [ np.squeeze(tl.to_numpy(factor)) for factor in factors]


    return numpy_factors



def tensor_train_truerank(mat, L, c, tol):
    
    max_r = c*2**L
    T = reshape_matrix_to_tensor_QTT(mat, L, c)
    cores, ranks = tt_svd(T,tol)   
    
    # T_rec = tt_reconstruct(cores)
    # rel_error = np.linalg.norm(T - T_rec) / np.linalg.norm(T)
    # print("Relative error:", rel_error)    
    
    return ranks


def tt_reconstruct(cores):
    T = cores[0]
    for G in cores[1:]:
        T = np.tensordot(T, G, axes=([-1], [0]))
    return T.squeeze()


def tt_svd(T, tol):
    from sklearn.utils.extmath import randomized_svd

    dims = T.shape
    d = len(dims)
    cores = []
    ranks = [1]

    # total Frobenius norm
    norm_T = np.linalg.norm(T)
    eps = tol * norm_T / np.sqrt(d - 1)

    tensor = T.copy()
    r_prev = 1

    for k in range(d - 1):
        n_k = dims[k]

        # reshape to matrix
        tensor = tensor.reshape(r_prev * n_k, -1)

        # SVD
        U, S, Vh = np.linalg.svd(tensor, full_matrices=False)
        
        # # randomized SVD
        # r_try = min(tensor.shape[0], tensor.shape[1])
        # U, S, Vh = randomized_svd(tensor,n_components=r_try)


        # determine rank via tolerance
        r_k = len(S)
        for i in range(len(S)):
            if S[i] <= S[0]*tol/np.sqrt(d - 1):
                r_k = i
                break
        # truncate
        U = U[:, :r_k]
        S = S[:r_k]
        Vh = Vh[:r_k, :]

        # form TT core
        core = U.reshape(r_prev, n_k, r_k)
        cores.append(core)
        ranks.append(r_k)

        # prepare next tensor
        tensor = np.diag(S) @ Vh
        r_prev = r_k

    # last core
    cores.append(tensor.reshape(r_prev, dims[-1], 1))
    ranks.append(1)

    return cores, ranks


def compute_slice_norms_sq(T, inds, mode):
    """
    Compute the norm of each slice of a sparse tensor along a given mode.
    
    Parameters:
    -----------
    T : ndarray
        Sparse tensor values, shape (nnz,)
    inds : ndarray
        Index tuples, shape (nnz, num_modes)
    mode : int
        Mode along which to compute slice norms
    
    Returns:
    --------
    norms : ndarray
        norms[i] = ||T[mode == i]||^2_2
        Array of length max(inds[:, mode]) + 1
    """
    mode_indices = inds[:, mode]
    
    # Sum of squares for each slice using bincount
    if np.issubdtype(T.dtype, np.complexfloating):
        sum_sq = np.bincount(mode_indices, weights=np.abs(T)**2)
    else:
        sum_sq = np.bincount(mode_indices, weights=T**2)
    
    #norms = np.sqrt(sum_sq)
    return sum_sq


def Update_fac_and_grad(N, Z, inds, level, L):
    """
    Compute scaled updates for factor N and sparse tensor Z.
    
    Scaling is done slice-wise:
        alpha[i] = ||N[i]||^2_F / ||Z[slice i]||^2
        
    For 2D factor (level 0 or L+1):
        N has shape (num_slices, R)
        delta_N[i,:] = alpha[i] * N[i,:]
        
    For 3D factor (level 1 to L):
        N has shape (K, R1, R2)
        delta_N[i,:,:] = alpha[i] * N[i,:,:]
    
    Parameters:
    -----------
    N : ndarray
        Dense factor matrix, shape (num_slices, R) for 2D or (K, R1, R2) for 3D
    Z : ndarray
        Sparse tensor values, shape (nnz,)
    inds : ndarray
        Index tuples for Z, shape (nnz, num_modes)
    level : int
        Mode along which slices are defined
    L : int
        Number of inner levels
    
    Returns:
    --------
    delta_N : ndarray
        Scaled factor, same shape as N
    delta_Z : ndarray
        Scaled sparse tensor values, shape (nnz,)
    """
    num_slices = N.shape[0]
    
    # Compute ||N[i]||^2_F for each slice
    if level == 0 or level == L + 1:
        # 2D case: N has shape (num_slices, R)
        if np.issubdtype(N.dtype, np.complexfloating):
            N_norms_sq = np.sum(np.abs(N)**2, axis=1)
        else:
            N_norms_sq = np.sum(N**2, axis=1)
    else:
        # 3D case: N has shape (K, R1, R2)
        if np.issubdtype(N.dtype, np.complexfloating):
            N_norms_sq = np.sum(np.abs(N)**2, axis=(1, 2))
        else:
            N_norms_sq = np.sum(N**2, axis=(1, 2))
    
    # Compute ||Z[slice i]||^2 for each slice
    Z_norms_sq = compute_slice_norms_sq(Z, inds, level)
    
    # Pad if some slices have no entries in Z
    if len(Z_norms_sq) < num_slices:
        Z_norms_sq = np.pad(Z_norms_sq, (0, num_slices - len(Z_norms_sq)))
    
    # alpha[i] = ||N[i]||^2_F / ||Z[slice i]||^2
    alpha = np.zeros(num_slices, dtype=np.float64)
    nonzero_mask = Z_norms_sq > 0
    alpha[nonzero_mask] = N_norms_sq[nonzero_mask] / Z_norms_sq[nonzero_mask]
    
    # Scale factor slices
    if level == 0 or level == L + 1:
        # 2D: alpha[:, None] broadcasts (num_slices,) to (num_slices, 1)
        delta_N = alpha[:, None] * N
    else:
        # 3D: alpha[:, None, None] broadcasts (K,) to (K, 1, 1)
        delta_N = alpha[:, None, None] * N
    
    # Scale sparse entries
    mode_indices = inds[:, level]
    delta_Z = alpha[mode_indices] * Z
    
    return delta_N, delta_Z

def qr_factor_tensor_train(factor, outer, side):
    shape = list(factor.shape)
    
    if outer:
        # Assuming factor is 2D at the last level
        Q, R_fac = la.qr(factor,  mode='reduced')
        output = Q  # Already 2D

    else:
        if side == 0:
            mat = factor.reshape((shape[0]*shape[1], shape[2]))
        else:
            mat = factor.transpose(0, 2, 1).reshape((shape[0]*shape[2], shape[1]))
        
        Q, R_fac = la.qr(mat,  mode='reduced')

        if side == 0:
            output = Q.reshape((shape[0], shape[1], shape[2]))
        else:
            output = Q.reshape((shape[0], shape[2], shape[1])).transpose(0, 2, 1)

    return output, R_fac


def absorb_factor(R_fac,next_factor, side):
    shape = list(next_factor.shape)
    if len(shape) == 3:
        if side == 0:
            output = np.einsum('ij,ajk->aik', R_fac, next_factor, optimize=True) #coming from the left
        else:
            output = np.einsum('ij,akj->aki', R_fac, next_factor, optimize=True) # coming from the right
    else:
        output = np.einsum('ij,aj->ai', R_fac, next_factor, optimize=True) # direction does not matter
    
    return output


def orthogonalize_all(tensor_lst, wrt):
    length = len(tensor_lst) - 1
    if  wrt == 0:
        for level in range(length,0,-1):
            outer = (level == length)
            tensor_lst[level], R_fac = qr_factor_tensor_train(tensor_lst[level], outer=outer, side=1)
            tensor_lst[level-1] = absorb_factor(R_fac, tensor_lst[level-1], side=1)  
    else:
        for level in range(length):
            outer = (level == 0)
            tensor_lst[level], R_fac = qr_factor_tensor_train(tensor_lst[level], outer=outer, side=0)
            tensor_lst[level+1] = absorb_factor(R_fac, tensor_lst[level+1], side=0)
    
    return tensor_lst



def ADAM_tensor_train(T_sparse, inds, T_test, inds_test, L, tensor_lst, 
    regu=1e-9, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-6):
    """
    Memory-optimized ADAM optimizer.
    """
    m = [np.zeros_like(x) for x in tensor_lst]
    v = [np.zeros_like(x) for x in tensor_lst]
    errors = []

    inds, T_sparse = sort_inds_and_T(inds, T_sparse, 0)
    unqs, starts, counts = np.unique(inds[:, 0], return_index=True, return_counts=True)
    inds_tups = [inds[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    nnz = len(T_sparse)
    
    # Precompute bias correction denominators
    bias1 = 1.0
    bias2 = 1.0

    for t in range(1, max_iter + 1):
        recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups, tensor_lst, 0, L-1)
        residual = T_sparse - recon
        del recon  # Free immediately

        s = time.time()
        
        # Update bias correction terms
        bias1 *= beta1
        bias2 *= beta2
        lr_t = lr * np.sqrt(1 - bias2) / (1 - bias1)  # Bias-corrected learning rate
        
        max_grad_norm = 0.0
        
        # Process ONE level at a time to reduce peak memory
        for level in range(len(tensor_lst)):
            # Compute gradient for this level only
            g = tensor_train_gradient(residual, inds, tensor_lst, level, L-1, regu)
            
            # Track gradient norm for convergence
            g_norm = np.linalg.norm(g)
            max_grad_norm = max(max_grad_norm, g_norm)
            
            # Update moments IN-PLACE
            m[level] *= beta1
            m[level] += (1 - beta1) * g
            
            v[level] *= beta2
            np.add(v[level], (1 - beta2) * (g * np.conj(g)).real, out=v[level])  # For complex
            # Or for real: v[level] += (1 - beta2) * (g ** 2)
            
            # Update parameters IN-PLACE (no m_hat, v_hat arrays)
            tensor_lst[level] += lr_t * m[level] / (np.sqrt(v[level]) + epsilon)
            
            del g  # Free gradient immediately
        
        del residual  # Free residual
        
        e = time.time()
        grad_time = e - s
        print('Time in gradient computation', grad_time)

        # Check convergence
        if max_grad_norm < tol:
            print(f"Converged in {t} iterations.")
            return tensor_lst, errors

        s1 = time.time()
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L-1)
        errors.append(error)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L-1)
        e1 = time.time()
        
        print('Time in error computation', e1 - s1)
        print('Total time in iteration', t, ':', e - s + grad_time)
        print('Relative error in observed entries:', error)
        print('Relative test error after', t, 'iterations:', test_error)
    
    print("Maximum iterations reached without convergence.")
    return tensor_lst, errors


def tensor_train_completion(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu):
    # We should have L +1 length list for QTT
    if(L==0):
        print('------------------matrix completion----------------------------')
    else:
        print('------------------tensor train completion----------------------------')
    nnz = len(inds)
    print("Number of observed entries:",nnz)
    
    errors = []
    for iters in range(num_iters):
        s = time.time()
        print("Iteration", iters+1,"/",num_iters)

        for level in range(L+1):
            print('At level: ',level)
            # Important to note that I have given the L argument as L - 1
            # As the last index in butterfly solve is L+1, i.e., (L+2)th element
            # But we have L +1 total elements in QTT

            tensor_lst = tensor_train_ALS_solve(T_sparse, inds, tensor_lst, level, L - 1, regu=regu)
        
        e = time.time()
        print('Time in iteration', iters+1 ,':', e-s)
        
        s= time.time()
        # Same arguments here as above
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L-1)
        errors.append(error)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L-1)
        e = time.time()
        print('Time in error computation',e-s)
        print('Relative error in observed entries: ',error)
        print('Relative test error after', iters + 1,' iterations: ',test_error)
        print('-----------------')
        if iters + 1 >= 5 and error >= 3:
            print('Overfitting or error not reducing, stopping iterations')
            break
        if error < tol:
            print('converged')
            break
    
    return tensor_lst


def tensor_train_ADF(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu=0):
    print('------------------tensor train ADF completion----------------------------')
    errors = []

    inds,  T_sparse = sort_inds_and_T(inds,  T_sparse, 0)
    unqs, starts, counts = np.unique(inds[:, 0], return_index = True, return_counts = True)
    inds_tups = [inds[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    nnz = len(T_sparse)
    print("Number of observed entries:",nnz)
    recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,0, L-1)
    grad_tensor = T_sparse - recon

    
    for iters in range(num_iters):
        print("Iteration", iters+1,"/",num_iters)
        #Since we are going to start from 0 to L+1, we will orthogonalize all factors wrt first
        tensor_lst = orthogonalize_all(tensor_lst, wrt=0)
        grad_time = 0
        recon_time = 0
        factor_time = 0
        s = time.time()
        for level in range(L+1):
            print('At level: ',level)
            s_grad = time.time()
            N = tensor_train_gradient(grad_tensor, inds, tensor_lst, level, L-1, regu) # N as used in paper algo 5
            e_grad = time.time()
            
            grad_time += e_grad - s_grad

            new_lst = [t.copy() for t in tensor_lst] 
            new_lst[level] = N

            s_recon_time = time.time()
            Z = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,new_lst,0, L-1) # paper algo 5
            e_recon_time = time.time()

            recon_time += e_recon_time - s_recon_time

            delta_fac, delta_grad = Update_fac_and_grad(N, Z, inds, level, L-1)
            # alpha = la.norm(N)**2 / (la.norm(Z))**2 # We may need to change alpha for each slice in the level
            # delta_fac = alpha*N
            # delta_grad = alpha*Z    

            tensor_lst[level] += delta_fac
            grad_tensor -= delta_grad

            # Now we have to orthogonalize wrt the next level
            # This only requires one orthogonalization
            s_factor = time.time()
            if level != L:
                output, R_fac = qr_factor_tensor_train(tensor_lst[level], outer= (level==0), side=0)
                tensor_lst[level] = output
                tensor_lst[level+1] = absorb_factor(R_fac, tensor_lst[level+1], side=0)
            e_factor = time.time()

            factor_time += e_factor - s_factor
        
        e = time.time()
        

        print('Time in gradient computation', grad_time)
        print('Time in reconstruction computation', recon_time)
        print('Time in QR factor computation', factor_time)
        print('Time in iteration', iters+1 ,':', e-s)
        s= time.time()
        # Same arguments here as above
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L-1)
        errors.append(error)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L-1)
        e = time.time()
        print('Time in error computation',e-s)
        print('Relative error in observed entries: ',error)
        print('Relative test error after', iters + 1,' iterations: ',test_error)
        print('-----------------')
        if iters + 1 >= 5 and error >= 3:
            print('Overfitting or error not reducing, stopping iterations')
            break
        if error < tol:
            print('converged')
            break
    
    return tensor_lst








def tensor_train_completion_v2(T_sparse, inds, T_test, inds_test, L, tensor_lst,
                                num_iters, tol, regu):
    """
    Tensor train completion using FastTTComputer.
    QTT has L+1 factors (levels 0..L), butterfly uses L_b = L-1.
    """
    L_b = L - 1

    if L == 0:
        print('------------------matrix completion----------------------------')
    else:
        print('------------------tensor train completion----------------------------')

    nnz = len(inds)
    print(f"Number of observed entries: {nnz}")
    print(f"Available memory: {get_available_memory()/1e9:.2f} GB")

    print("Initializing Numba (first call compiles)...")
    computer = FastTTComputer(tensor_lst, L_b)
    print("Done.")

    errors = []
    for iters in range(num_iters):
        s = time.time()
        print("Iteration", iters + 1, "/", num_iters)

        for level in range(L + 1):
            print(f'  Level {level}', end='')
            tensor_lst = tensor_train_ALS_solve_fast(
                T_sparse, inds, tensor_lst, level, L_b, regu, computer
            )
            print()

        e = time.time()
        print('Time in iteration', iters + 1, ':', e - s)

        s = time.time()
        error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L_b, computer)
        errors.append(error)
        test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L_b, computer)
        e = time.time()

        print('Time in error computation', e - s)
        print('Relative error in observed entries:', error)
        print('Relative test error after', iters + 1, 'iterations:', test_error)
        print('-----------------')

        if iters + 1 >= 5 and error >= 3:
            print('Overfitting or error not reducing, stopping iterations')
            break
        if error < tol:
            print('converged')
            break

    return tensor_lst

def tensor_train_ADF_v2(T_sparse, inds, T_test, inds_test, L, tensor_lst,
                         num_iters, tol, regu=0):
    """
    Tensor train ADF using FastTTComputer.
    """
    L_b = L - 1

    print('------------------tensor train ADF completion----------------------------')

    nnz = len(T_sparse)
    print(f"Number of observed entries: {nnz}")
    print(f"Available memory: {get_available_memory()/1e9:.2f} GB")

    print("Initializing Numba (first call compiles)...")
    computer = FastTTComputer(tensor_lst, L_b)
    print("Done.")

    errors = []
    inds_i64 = np.ascontiguousarray(inds.astype(np.int64))

    # Initial reconstruction and residual
    recon = computer.reconstruct(inds_i64)
    grad_tensor = T_sparse - recon

    s = time.time()
    error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L_b, computer)
    errors.append(error)
    test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L_b, computer)
    e = time.time()
    print('Time in error computation', e - s)
    print('Relative error in observed entries:', error)

    for iters in range(num_iters):
        print("Iteration", iters + 1, "/", num_iters)

        # Orthogonalize all factors wrt first
        tensor_lst = orthogonalize_all(tensor_lst, wrt=0)
        computer._set_tensors(tensor_lst)

        # Recompute residual after orthogonalization
        recon = computer.reconstruct(inds_i64)
        grad_tensor = T_sparse - recon

        grad_time = 0
        recon_time = 0
        factor_time = 0
        s = time.time()

        for level in range(L + 1):
            print(f'  Level {level}', end='')

            s_grad = time.time()
            N, unqs = tensor_train_gradient_fast(
                grad_tensor, inds, tensor_lst, level, L_b, regu, computer
            )
            e_grad = time.time()
            grad_time += e_grad - s_grad

            # Scatter into full tensor
            N_full = np.zeros_like(tensor_lst[level])
            N_full[unqs] = N

            new_lst = [t.copy() for t in tensor_lst]
            new_lst[level] = N_full.copy()

            # Reconstruct Z using temporary computer
            s_recon_time = time.time()
            temp_computer = FastTTComputer.__new__(FastTTComputer)
            temp_computer.L = L_b
            temp_computer.tensor_lst = new_lst
            temp_computer.start_tensor = np.ascontiguousarray(new_lst[0])
            temp_computer.end_tensor = np.ascontiguousarray(new_lst[L_b + 1])
            temp_computer.mid_tensors = _make_mid_list(new_lst, L_b)

            Z = temp_computer.reconstruct(inds_i64)
            e_recon_time = time.time()
            recon_time += e_recon_time - s_recon_time

            delta_fac, delta_grad = Update_fac_and_grad(N_full, Z, inds, level, L_b)

            tensor_lst[level] += delta_fac
            grad_tensor -= delta_grad

            computer.update_tensor(level, tensor_lst[level])

            # QR orthogonalization
            s_factor = time.time()
            if level != L:
                output, R_fac = qr_factor_tensor_train(
                    tensor_lst[level], outer=(level == 0), side=0
                )
                tensor_lst[level] = output
                tensor_lst[level + 1] = absorb_factor(R_fac, tensor_lst[level + 1], side=0)
                computer.update_tensor(level, tensor_lst[level])
                computer.update_tensor(level + 1, tensor_lst[level + 1])
            e_factor = time.time()
            factor_time += e_factor - s_factor

            print()

        e = time.time()

        print('Time in gradient computation', grad_time)
        print('Time in reconstruction computation', recon_time)
        print('Time in QR factor computation', factor_time)
        print('Time in iteration', iters + 1, ':', e - s)

        s = time.time()
        error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L_b, computer)
        errors.append(error)
        test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L_b, computer)
        e = time.time()

        print('Time in error computation', e - s)
        print('Relative error in observed entries:', error)
        print('Relative test error after', iters + 1, 'iterations:', test_error)
        print('-----------------')

        if iters + 1 >= 5 and error >= 3:
            print('Overfitting or error not reducing, stopping iterations')
            break
        if error < tol:
            print('converged')
            break

    return tensor_lst


def ADAM_tensor_train_v2(T_sparse, inds, T_test, inds_test, L, tensor_lst,
                          regu=1e-9, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,
                          max_iter=100, tol=1e-6):
    """
    ADAM optimizer for tensor train using FastTTComputer.
    """
    L_b = L - 1

    print('------------------Tensor Train ADAM----------------------------')

    nnz = len(T_sparse)
    print(f"Number of observed entries: {nnz}")
    print(f"Available memory: {get_available_memory()/1e9:.2f} GB")

    print("Initializing Numba (first call compiles)...")
    computer = FastTTComputer(tensor_lst, L_b)
    print("Done.")

    m = [np.zeros_like(x) for x in tensor_lst]
    v = [np.zeros_like(x) for x in tensor_lst]
    errors = []

    inds_i64 = np.ascontiguousarray(inds.astype(np.int64))

    # Precompute bias correction terms
    bias1 = 1.0
    bias2 = 1.0

    for t in range(1, max_iter + 1):
        print(f"Iteration {t} / {max_iter}")

        recon = computer.reconstruct(inds_i64)
        residual = T_sparse - recon
        del recon

        s = time.time()

        # Update bias correction terms
        bias1 *= beta1
        bias2 *= beta2
        lr_t = lr * np.sqrt(1 - bias2) / (1 - bias1)

        max_grad_norm = 0.0

        for level in range(L + 1):
            print(f'  Level {level}', end='')

            g_partial, unqs = tensor_train_gradient_fast(
                residual, inds, tensor_lst, level, L_b, regu, computer
            )

            # Scatter into full gradient
            g = np.zeros_like(tensor_lst[level])
            g[unqs] = g_partial

            g_norm = np.linalg.norm(g)
            max_grad_norm = max(max_grad_norm, g_norm)

            # Update moments in-place
            m[level] *= beta1
            m[level] += (1 - beta1) * g

            v[level] *= beta2
            if np.iscomplexobj(g):
                v[level] += (1 - beta2) * (g * np.conj(g)).real
            else:
                v[level] += (1 - beta2) * (g ** 2)

            # Update parameters in-place
            tensor_lst[level] += lr_t * m[level] / (np.sqrt(v[level]) + epsilon)

            computer.update_tensor(level, tensor_lst[level])

            del g
            print()

        del residual

        e = time.time()
        grad_time = e - s
        print('Time in gradient computation', grad_time)

        if max_grad_norm < tol:
            print(f"Converged in {t} iterations.")
            return tensor_lst, errors

        s1 = time.time()
        error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L_b, computer)
        errors.append(error)
        test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L_b, computer)
        e1 = time.time()

        print('Time in error computation', e1 - s1)
        print('Total time in iteration', t, ':', grad_time)
        print('Relative error in observed entries:', error)
        print('Relative test error after', t, 'iterations:', test_error)
        print('-----------------')

    print("Maximum iterations reached without convergence.")
    return tensor_lst, errors