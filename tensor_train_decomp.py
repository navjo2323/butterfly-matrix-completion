import numpy as np
import tensorly as tl
from tensorly.decomposition import matrix_product_state
import numpy.linalg as la
import time
from butterfly_tensor_train import tensor_train_ALS_solve, compute_sparse_butterfly


import numpy as np

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
        error = la.norm(T_sparse - compute_sparse_butterfly(inds, tensor_lst, L -1)) / la.norm(T_sparse)
        errors.append(error)
        test_error = la.norm(T_test - compute_sparse_butterfly(inds_test,tensor_lst,L - 1)) / la.norm(T_sparse)
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




