import numpy as np
import numpy.linalg as la
import time
import logging


def gen_tensor_train_list(L, c, ranks, rng, real=1):
    if(real==1):
        # Generate the initial tensor with complex numbers
        tensor_lst = [rng.uniform(-1, 1, size=(c * (2**L), ranks[0]))]

        # Generate tensors for the first half of the list with complex numbers
        for i in range(L // 2):
            tensor_lst.append(rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i+1])))

        # Generate tensors for the second half of the list with complex numbers
        for i in range(L // 2, 0, -1):
            tensor_lst.append(rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i-1])))

        # Generate the final tensor with complex numbers
        tensor_lst.append(rng.uniform(-1, 1, size=(c * 2**L, ranks[0])))
    else:
        # Generate the initial tensor with complex numbers
        tensor_lst = [rng.uniform(-1, 1, size=(c * (2**L), ranks[0])) + 
                    1j * rng.uniform(-1, 1, size=(c * (2**L), ranks[0]))]

        # Generate tensors for the first half of the list with complex numbers
        for i in range(L // 2):
            tensor_lst.append(rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i+1])) + 
                            1j * rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i+1])))

        # Generate tensors for the second half of the list with complex numbers
        for i in range(L // 2, 0, -1):
            tensor_lst.append(rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i-1])) + 
                            1j * rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i-1])))

        # Generate the final tensor with complex numbers
        tensor_lst.append(rng.uniform(-1, 1, size=(c * 2**L, ranks[0])) + 
                        1j * rng.uniform(-1, 1, size=(c * 2**L, ranks[0])))


    return tensor_lst

def create_inds(I, J, nnz, rng):
    unique_tuples = set()
    while len(unique_tuples) < nnz:
        # Generate a batch of random indices
        batch_size = nnz - len(unique_tuples)
        Is = rng.randint(low=0, high=I, size=batch_size)
        Js = rng.randint(low=0, high=J, size=batch_size)
        
        # Combine Is and Js into tuples and add them to the set
        new_tuples = zip(Is, Js)
        unique_tuples.update(new_tuples)
    
    return tuple(unique_tuples)
    
def reverse_from_binary_shape(binary_array):
    # Determine R from the last dimension of binary_array
    R = binary_array.shape[-1]
    
    # Determine c from the second last dimension of binary_array
    c = binary_array.shape[-2]
    
    # Determine the number of binary dimensions (L)
    binary_shape = binary_array.shape[:-2]  # Exclude c and R dimensions
    L = len(binary_shape)
    
    # Calculate N as M * c, where M = 2^L
    M = 2 ** L
    N = M * c
    
    # Initialize the output array with shape (N, R)
    if np.issubdtype(binary_array.dtype, np.floating):
        integer_array = np.zeros((N, R), dtype = np.float64)
    else:
        integer_array = np.zeros((N, R), dtype = np.complex128)
    
    # Precompute powers of 2 for bit conversion
    powers_of_2 = 1 << np.arange(L)[::-1]
    
    # Iterate over each possible combination of indices in binary dimensions
    for binary_indices in np.ndindex(binary_shape):
        # Convert binary indices to integer index
        int_index = np.dot(binary_indices, powers_of_2)
        
        # Iterate over each possible value of remainder_index (0 to c-1)
        for remainder_index in range(c):
            # Compute the linear index in the original array
            linear_index = int_index * c + remainder_index
            
            # Assign values from binary_array to integer_array
            index_tuple = binary_indices + (remainder_index, slice(None))
            integer_array[linear_index] = binary_array[index_tuple]
    
    return integer_array

def convert_to_binary_shape(array, c):
    N = array.shape[0]
    R = array.shape[-1]
    M = N // c
    L = int(np.log2(M))

    # Precompute powers of 2 for extracting most significant bits first
    powers_of_2 = 1 << np.arange(L)[::-1]

    # Initialize a new array with binary dimensions
    new_shape = (2,) * L + (c, R)
    if np.issubdtype(array.dtype, np.floating):
        new_array = np.zeros(new_shape, dtype=np.float64)
    else:
        new_array = np.zeros(new_shape, dtype=np.complex128)

    # Fill in the new array using binary indices
    for idx in range(N):
        # Calculate binary indices from idx (most significant bits first)
        binary_indices = ((idx // c) & powers_of_2) > 0
        remainder_index = idx % c
        
        # Assign values from original array to new_array
        index_tuple = tuple(binary_indices.astype(int)) + (remainder_index, slice(None))
        new_array[index_tuple] = array[idx]

    return new_array

def reverse_to_binary_array(integer_array):
    # Get the shape of the integer array
    N, R1, R2 = integer_array.shape
    
    # Determine the number of binary dimensions
    # The number of binary combinations is equal to the count of bits needed to represent N
    num_binary_dims = int(np.log2(N))
    
    # Initialize the binary array with the shape (2, 2, ..., 2, R, R)
    binary_shape = (2,) * num_binary_dims + (R1, R2)
    if np.issubdtype(integer_array.dtype, np.floating):
        binary_array = np.zeros(binary_shape, dtype= np.float64)
    else:
        binary_array = np.zeros(binary_shape, dtype= np.complex128)

    # Fill the binary array with values based on the integer indices
    for index in range(N):
        # Convert index back to binary representation
        binary_indices = np.array(list(np.binary_repr(index, width=num_binary_dims)), dtype=int)
        binary_array[tuple(binary_indices)] = integer_array[index]
    
    return binary_array

def convert_to_integer_array(binary_array):
    R2 = binary_array.shape[-1]  # Dimension R of the matrices
    R1 = binary_array.shape[-2]
    binary_shape = binary_array.shape[:-2]  # The binary dimensions
    N = np.prod(binary_shape)  # Total number of combinations from binary dims


    # Flatten the input array along the binary dimensions
    flattened = binary_array.reshape(N, R1, R2)



    # Calculate integer indices from binary dimensions
    num_dims = len(binary_shape)
    indices = np.indices(binary_shape).reshape(num_dims, -1).T
    int_indices = np.dot(indices, 1 << np.arange(num_dims - 1, -1, -1))


    # Ensure int_indices are within bounds
    if np.any(int_indices >= flattened.shape[0]):
        raise ValueError("Calculated integer indices exceed available range.")

    # Use the integer indices to reshape the array
    result = flattened[int_indices]
    return result


def convert_lst_to_3d(g_lst,h_lst, L, c):
    g_lst_new = []
    h_lst_new = []

    g0 = reverse_from_binary_shape(g_lst[0].copy())
    h0 = reverse_from_binary_shape(h_lst[0].copy())

    g_lst_new = [g0] + [convert_to_integer_array(arr) for arr in g_lst[1:]]
    h_lst_new = [h0] + [convert_to_integer_array(arr) for arr in h_lst[1:]]
    

    return g_lst_new,h_lst_new


def convert_lst_to_Nd(g_lst, h_lst, L, c):
    g_lst_new = []
    h_lst_new = []

    g0 = convert_to_binary_shape(g_lst[0].copy(), c)
    h0 = convert_to_binary_shape(h_lst[0].copy(), c)

    g_lst_new = [g0] + [reverse_to_binary_array(arr.copy()) for arr in g_lst[1:]]
    h_lst_new = [h0] + [reverse_to_binary_array(arr.copy()) for arr in h_lst[1:]]


    return g_lst_new, h_lst_new

def make_one_list(g_lst,h_lst):
    # Convert the list into tensor train format

    # We have g_lst in the right order such that g_lst[0] vectors multiply to the first index of
    # g_lst[1] matrix and so on
    # h_lst is in the wrong order, let us first reverse it
    h_lst = h_lst[::-1]

    # We still need to transpose each tensor in H, except the last one since it will be easier to 
    # index into rows
    h_lst[:-1] = [arr.conj().transpose(0, 2, 1) for arr in h_lst[:-1]]
    h_lst[-1] = h_lst[-1].conj()

    return g_lst + h_lst

def make_two_lists(tensor_lst):
    # Convert back to two lists for checking

    g_lst = tensor_lst[ : len(tensor_lst) // 2]

    h_lst = tensor_lst[ len(tensor_lst) // 2 : ]
    h_lst = h_lst[::-1]
    h_lst[1:] = [arr.conj().transpose(0, 2, 1) for arr in h_lst[1:]]
    h_lst[0] = h_lst[0].conj()

    return g_lst, h_lst



def encode_tuples(indices, L, c):
    indices = np.array(indices)
    I = indices[:, 0]
    J = indices[:, 1]
    
    # Calculate ind_is and ind_js by integer division
    ind_is = I // c
    ind_js = J // c
    
    # Convert ind_is and ind_js to binary representations using bit-shifting
    ind_is_binary = ((ind_is[:, None] >> np.arange(L-1, -1, -1)) & 1).astype(np.int32)
    ind_js_binary = ((ind_js[:, None] >> np.arange(L-1, -1, -1)) & 1).astype(np.int32)
    
    # Initialize array for storing intermediate results
    result = np.zeros((len(I), 2 + L), dtype=np.int64)
    
    # Calculate intermediate integers for the first part
    for i in range(1, L // 2 + 1):
        # Extract bits and concatenate
        combined_bits = np.hstack((ind_is_binary[:, :L - i + 1], ind_js_binary[:, :i]))
        
        # Convert combined bits to integer
        part = np.dot(combined_bits, 1 << np.arange(combined_bits.shape[1] - 1, -1, -1))
        result[:, i] = part

    # First column will store the original I values
    result[:, 0] = I
    
    # Calculate intermediate integers for the second part (similar to the second loop)
    for i in range(L // 2):
        # Extract bits and concatenate
        combined_bits = np.hstack((ind_is_binary[:, :i + 1], ind_js_binary[:, :L - i] ))
        
        # Convert combined bits to integer
        part = np.dot(combined_bits, 1 << np.arange(combined_bits.shape[1] - 1, -1, -1))
        result[:, L - i] = part
    
    # Last column will store the original J values
    result[:, L + 1] = J

    # Convert the result array to a list of tuples
    encoded_list = np.array([tuple(row) for row in result])
    
    return encoded_list



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

    return sorted_array, reordered_T


def compute_sparse_butterfly(inds, tensor_lst, L):
    vecs = tensor_lst[0][inds[:, 0]]
    for i in range(1,L+1):
        vecs = np.einsum('ir,irz->iz',vecs,tensor_lst[i][inds[:,i]],optimize=True)

    return np.einsum('iz,iz->i',vecs,tensor_lst[L+1][inds[:,L+1]],optimize=True)



def multiply_mats(inds_tups, tensor_lst, level, L, row_shape):
    num_tuples = len(inds_tups)

    if level == 0:
        # Pre-compute indices for the last tensor
        H = [tensor_lst[L+1][inds[:, L+1]] for inds in inds_tups]

        # Iterate in reverse order and apply einsum
        for i in range(L, 0, -1):
            H = [np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H[j],optimize=True) for j, inds in enumerate(inds_tups)]

    elif level == L + 1:
        # Pre-compute indices for the first tensor
        H = [tensor_lst[0][inds[:, 0]] for inds in inds_tups]

        # Iterate forwards and apply einsum
        for i in range(1, L + 1):
            H = [np.einsum('ir,irz->iz', H[j], tensor_lst[i][inds[:, i]],optimize=True) for j, inds in enumerate(inds_tups)]

    else:
        # Handle the case where level is between 0 and L+1
        H1 = [tensor_lst[0][inds[:, 0]] for inds in inds_tups]
        H2 = [tensor_lst[L+1][inds[:, L+1]] for inds in inds_tups]

        # Compute H1 by iterating forward up to 'level'
        for i in range(1, level):
            H1 = [np.einsum('ir,irz->iz', H1[j], tensor_lst[i][inds[:, i]],optimize=True) for j, inds in enumerate(inds_tups)]
        # Compute H2 by iterating backward from L down to 'level'
        for i in range(L, level, -1):
            H2 = [np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H2[j],optimize=True) for j, inds in enumerate(inds_tups)]

        # Combine H1 and H2
        H = [np.einsum('ir,iz->irz', H1[j], H2[j],optimize=True).reshape((len(inds), row_shape)) for j, inds in enumerate(inds_tups)]

    return H




'''
Faster algo: have sorted order of the indices.

now say we solve for side == 1 and some list index

each vector will be multiplied by a list of second matrices, which will then be multiplied by a list of third
and so on till the last index

similarly every list of matrix after solve will be multiplied by another list till we get all to the vectors

the remaining part will proceed as first list proceeded.

if it was vice versa then the remaining part will proceed as second list proceeded.
'''

def tensor_train_ALS_solve(T, inds, tensor_lst, level, L, regu):

    if level ==0 or level == L + 1:
        row_shape = tensor_lst[level].shape[-1]
    else:
        row_shape = np.prod(tensor_lst[level].shape[1:])

    s = time.time()

    sorted_tuples, T_new = sort_inds_and_T(inds, T, level)

    e = time.time()


    #print('Time in sorting',e-s)

    s = time.time()

    I = regu*np.eye(row_shape)

    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index = True, return_counts = True)

    inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]


    # Further can be optimized based on sorted indices
    # For now lets keep it this way
    Hs = multiply_mats(inds_tups, tensor_lst, level, L, row_shape) 


    RHS = np.array([np.dot(T_new[starts[i]: starts[i] + counts[i] ], Hs[i].conj()) for i in range(len(unqs))])


    LHS = np.array([np.dot(H.conj().T ,H) + I for H in Hs])

    result = la.solve(LHS , RHS)

    if level ==0 or level == L + 1:
        tensor_lst[level][unqs] = result 
    else:
        tensor_lst[level][unqs] = result.reshape( (len(unqs),) + tensor_lst[level].shape[1:])
        
    return tensor_lst


def butterfly_tensor_train_completer(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu):
    if(L==0):
        print('------------------matrix completion----------------------------')
    else:
        print('------------------butterfly completion----------------------------')
    nnz = len(inds)
    print("Number of observed entries:",nnz)
    
    errors = []
    for iters in range(num_iters):
        s = time.time()
        print("Iteration", iters+1,"/",num_iters)

        for level in range(L+2):
            print('At level: ',level)
            tensor_lst = tensor_train_ALS_solve(T_sparse, inds, tensor_lst, level, L, regu=regu)
        
        e = time.time()
        print('Time in iteration', iters+1 ,':', e-s)
        
        s= time.time()
        error = la.norm(T_sparse - compute_sparse_butterfly(inds, tensor_lst, L)) / la.norm(T_sparse)
        errors.append(error)
        test_error = la.norm(T_test - compute_sparse_butterfly(inds_test,tensor_lst,L)) / la.norm(T_sparse)
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

