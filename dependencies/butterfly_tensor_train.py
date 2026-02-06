import itertools
import numpy as np
import numpy.linalg as la
import time
import scipy.linalg as sla
import logging
import psutil
# do pip install psutil if not already installed


import numba
from numba import njit, prange
from numba.typed import List as NumbaList
import time

def gen_tensor_train_list(L, c, ranks, rng, real=1):
    if(real==1):
        # Generate the initial tensor 
        tensor_lst = [rng.uniform(-1, 1, size=(c * (2**L), ranks[0]))]

        # Generate tensors for the first half 
        for i in range(L // 2):
            tensor_lst.append(rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i+1])))

        # Generate tensors for the second half
        for i in range(L // 2, 0, -1):
            tensor_lst.append(rng.uniform(-1, 1, size=(2**(L+1), ranks[i], ranks[i-1])))

        # Generate the final tensor 
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

    # If nnz equals the total number of indices, return all index pairs.
    if nnz == I * J:
        # Using np.indices to generate a grid of indices:
        row_indices, col_indices = np.indices((I, J))
        # Reshape and stack into a (I*J, 2) array.
        return np.column_stack((row_indices.ravel(), col_indices.ravel()))

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
    THIS IS STABLE
    """
    if k is None:
        sorted_indices = np.lexsort(np.fliplr(tuples).T)
    else:
        sorted_indices = np.argsort(tuples[:, k])

    sorted_array = tuples[sorted_indices]
    reordered_T = T[sorted_indices]

    return sorted_array, reordered_T


def qr_factor_flat(factor, L, level, side):
    """
    QR/LQ factorization for arrays in 2D (N, R) or 3D (M, R1, R2) form.
    
    For side=0 (left to right sweep):
        - 2D: Standard QR, R absorbs rightward
        - 3D: Pair matrices based on bit at position `level`
              Concatenate along left rank (axis=1) to form (M/2, 2*R1, R2)
              Standard QR decomposition: A = QR
              Returns Q with orthonormal columns, R to absorb rightward
        
    For side=1 (right to left sweep):
        - 2D: Standard QR, then transpose R to L for leftward absorption
        - 3D: Pair matrices based on bit at position 0 (LSB)
              Concatenate along right rank (axis=2) to form (M/2, R1, 2*R2)
              LQ decomposition (via QR of transpose): A = LQ^T
              Returns Q with orthonormal rows, L to absorb leftward
    
    Parameters:
    -----------
    factor : ndarray
        2D array of shape (N, R) for outer factors, or
        3D array of shape (M, R1, R2) for inner factors
    L : int
        Total number of inner levels (number of 3D factors = L)
    level : int, in range [0, L+1]
        - level = 0: left outer factor (2D)
        - level in [1, L]: inner factors (3D)
        - level = L+1: right outer factor (2D)
    side : int, 0 or 1
        - side = 0: sweep left to right (QR, absorb R rightward)
        - side = 1: sweep right to left (LQ, absorb L leftward)
    
    Returns:
    --------
    output : ndarray
        Orthogonalized factor with same structure as input
    remainder : ndarray
        For side=0: R matrices of shape (batch, k, R_right) to absorb rightward
        For side=1: L matrices of shape (batch, R_left, k) to absorb leftward
    """
    
    if level == 0 or level == L + 1:
        # ===== 2D CASE: Outer factors =====
        # Shape: (N, R) where N = c * 2^L
        # Reshape to (M, c, R) where M = 2^L blocks of size (c, R)
        #
        # Both outer factors are stored the same way:
        #   - Rows index into the tensor (grouped by c)
        #   - Columns are the rank connecting to adjacent inner factor
        
        N, R = factor.shape
        c = N // (2 ** L)
        M = N // c  # M = 2^L
        
        reshaped = factor.reshape(M, c, R)
        
        # Always do standard QR on each (c, R) block
        # Q: (M, c, k), R_matrices: (M, k, R) where k = min(c, R)
        Q, R_matrices = la.qr(reshaped, mode='reduced')
        
        output = Q.reshape(N, -1)  # (N, k)
        
        if side == 0:
            # Absorb rightward: R @ next
            # R_matrices shape: (M, k, R) contracts with next's left rank R
            return output, R_matrices
        else:
            # Absorb leftward: prev @ L
            # Need L_matrices shape: (M, R, k) to contract with prev's right rank R
            # Simply transpose R_matrices: (M, k, R) -> (M, R, k)
            L_matrices = R_matrices.transpose(0, 2, 1)
            return output, L_matrices
    
    else:
        # ===== 3D CASE: Inner factors =====
        # Shape: (M, R1, R2) where M = 2^{L+1}
        # Each index m encodes binary tuple (i_{L-level}, ..., i_0, j_0, ..., j_{level-1})
        # where leftmost bit is MSB
        
        M, R1, R2 = factor.shape
        
        # Determine bit position for pairing based on sweep direction
        if side == 0:
            # Left to right: pair over bit at position `level`
            # This corresponds to the i_{L-level} index being peeled off
            bit_pos = level
        else:
            # Right to left: pair over bit at position 0 (LSB)
            # This corresponds to the j_{level-1} index being peeled off
            bit_pos = 0
        
        # Create index pairs: idx0 has bit=0 at bit_pos, idx1 has bit=1
        mask = 1 << bit_pos
        all_indices = np.arange(M)
        idx0 = all_indices[(all_indices >> bit_pos) & 1 == 0]  # (M/2,)
        idx1 = idx0 | mask  # Corresponding indices with bit=1
        
        # Extract paired matrices
        mat0 = factor[idx0]  # (M/2, R1, R2)
        mat1 = factor[idx1]  # (M/2, R1, R2)
        
        if side == 0:
            # ===== SIDE 0: QR decomposition =====
            # Concatenate along left rank (axis=1): (M/2, 2*R1, R2)
            # This stacks the two R1-dimensional blocks vertically
            concat = np.concatenate([mat0, mat1], axis=1)
            
            # Standard QR: A = QR where Q has orthonormal columns
            Q, R_matrices = la.qr(concat, mode='reduced')
            # Q: (M/2, 2*R1, k), R_matrices: (M/2, k, R2)
            # where k = min(2*R1, R2)
            
            k = Q.shape[-1]
            
            # Allocate output array
            if np.issubdtype(factor.dtype, np.floating):
                output = np.zeros((M, R1, k), dtype=np.float64)
            else:
                output = np.zeros((M, R1, k), dtype=np.complex128)
            
            # Split Q back along axis=1 and assign to correct indices
            output[idx0] = Q[:, :R1, :]   # First R1 rows
            output[idx1] = Q[:, R1:, :]   # Second R1 rows
            
            return output, R_matrices
        
        else:
            # ===== SIDE 1: LQ decomposition =====
            # Concatenate along right rank (axis=2): (M/2, R1, 2*R2)
            # This stacks the two R2-dimensional blocks horizontally
            concat = np.concatenate([mat0, mat1], axis=2)
            
            # LQ decomposition via QR of transpose: A = LQ^T
            # If A^T = Q'R', then A = R'^T Q'^T = L Q^T
            # where L = R'^T has shape matching for leftward absorption
            #
            # concat: (M/2, R1, 2*R2)
            # concat^T: (M/2, 2*R2, R1)
            Qt, Lt = la.qr(concat.transpose(0, 2, 1), mode='reduced')
            # Qt: (M/2, 2*R2, k), Lt: (M/2, k, R1)
            # where k = min(2*R2, R1)
            
            # Transpose back to get L and Q
            # Q = Qt^T has orthonormal rows: (M/2, k, 2*R2)
            # L = Lt^T: (M/2, R1, k) - this is what we absorb leftward
            Q = Qt.transpose(0, 2, 1)
            L_matrices = Lt.transpose(0, 2, 1)
            
            k = Q.shape[1]
            
            # Allocate output array
            if np.issubdtype(factor.dtype, np.floating):
                output = np.zeros((M, k, R2), dtype=np.float64)
            else:
                output = np.zeros((M, k, R2), dtype=np.complex128)
            
            # Split Q back along axis=2 and assign to correct indices
            output[idx0] = Q[:, :, :R2]   # First R2 columns
            output[idx1] = Q[:, :, R2:]   # Second R2 columns
            
            return output, L_matrices


def absorb_R(R_matrices, next_factor, L, level, side):
    """
    Absorb R/L matrices from QR/LQ factorization into the adjacent factor.
    
    For side=0 (left to right):
        - R_matrices has shape (batch, k, R_contracted)
        - Absorb into next factor's LEFT rank: new = R @ next
        - Index pairing always uses bit_pos=0 (LSB) in next factor
          because the newly added j-index appears at the LSB
        
    For side=1 (right to left):
        - L_matrices has shape (batch, R_contracted, k)
        - Absorb into prev factor's RIGHT rank: new = prev @ L
        - Index pairing depends on level:
            - level > L//2: use bit_pos=L (MSB) 
            - level <= L//2: use bit_pos=level-1
          because the index structure changes at the halfway point
    
    Parameters:
    -----------
    R_matrices : ndarray
        For side=0: R matrices of shape (batch, k, R_contracted)
        For side=1: L matrices of shape (batch, R_contracted, k)
    next_factor : ndarray
        Factor to absorb into:
        - For side=0: the factor to the RIGHT (will multiply R @ next)
        - For side=1: the factor to the LEFT (will multiply prev @ L)
    L : int
        Total number of inner levels
    level : int
        Level of the factor that was just QR/LQ decomposed
    side : int, 0 or 1
        - side = 0: absorb rightward into next factor's left rank
        - side = 1: absorb leftward into prev factor's right rank
    
    Returns:
    --------
    new_next_factor : ndarray
        Updated factor after absorption, same structure as next_factor
    """
    
    if level == 0 or level == L + 1:
        # ===== 2D → ? ABSORPTION =====
        # R_matrices: (M, k, R) or (M, R, k) from QR/LQ of outer factor
        # M = 2^L (number of blocks in outer factor)
        
        M = R_matrices.shape[0]
        
        if next_factor.ndim == 3:
            # ----- 2D → 3D: Outer to Inner -----
            # next_factor: (2M, R_left, R_right) - inner factor has twice as many matrices
            # Each R[m] or L[m] applies to a pair of matrices in next_factor
            
            M_next, R_left, R_right = next_factor.shape
            
            if side == 0:
                # Index correspondence: outer block m → inner blocks 2m, 2m+1
                # This is because the new j-index (j_0) appears at LSB (bit 0)
                idx0 = 2 * np.arange(M)  # [0, 2, 4, ...]
                idx1 = idx0 + 1           # [1, 3, 5, ...]
                
                # R @ next: contract R's right dim with next's left dim
                # R: (M, k, R_left), next: (M, R_left, R_right) -> (M, k, R_right)
                k = R_matrices.shape[1]
                
                if np.issubdtype(next_factor.dtype, np.floating):
                    new_next = np.zeros((M_next, k, R_right), dtype=np.float64)
                else:
                    new_next = np.zeros((M_next, k, R_right), dtype=np.complex128)
                
                new_next[idx0] = np.matmul(R_matrices, next_factor[idx0])
                new_next[idx1] = np.matmul(R_matrices, next_factor[idx1])
            
            else:
                # Pairing over bit L (MSB) because the new i-index appears at MSB
                bit_pos = L
                mask = 1 << bit_pos
                all_indices = np.arange(M_next)
                idx0 = all_indices[(all_indices >> bit_pos) & 1 == 0]
                idx1 = idx0 | mask
                
                # prev @ L: contract prev's right dim with L's left dim
                # L: (M, R_right, k), prev: (M, R_left, R_right) -> (M, R_left, k)
                k = R_matrices.shape[2]
                
                if np.issubdtype(next_factor.dtype, np.floating):
                    new_next = np.zeros((M_next, R_left, k), dtype=np.float64)
                else:
                    new_next = np.zeros((M_next, R_left, k), dtype=np.complex128)
                
                new_next[idx0] = np.matmul(next_factor[idx0], R_matrices)
                new_next[idx1] = np.matmul(next_factor[idx1], R_matrices)
            
            return new_next
        
        elif next_factor.ndim == 2:
            # ----- 2D → 2D: Outer to Outer (only when L=0) -----
            # next_factor: (N, R_outer)
            
            N, R_outer = next_factor.shape
            c = N // (2 ** L)
            M_outer = N // c  # Should equal M
            
            # Reshape to (M, c, R_outer) for block operations
            next_reshaped = next_factor.reshape(M_outer, c, R_outer)
            
            if side == 0:
                # outer @ R^T: (M, c, R_outer) @ (M, R_outer, k) -> (M, c, k)
                k = R_matrices.shape[1]
                R_T = R_matrices.transpose(0, 2, 1)  # (M, R, k)
                new_next = np.matmul(next_reshaped, R_T)
            else:
                # outer @ L: (M, c, R_outer) @ (M, R_outer, k) -> (M, c, k)
                k = R_matrices.shape[2]
                new_next = np.matmul(next_reshaped, R_matrices)
            
            return new_next.reshape(M_outer * c, k)
    
    else:
        # ===== 3D → ? ABSORPTION =====
        # R_matrices from QR: (M/2, k, R_right)
        # L_matrices from LQ: (M/2, R_left, k)
        
        M_half = R_matrices.shape[0]
        M = 2 * M_half  # Size of next_factor's first dimension
        
        # Determine bit position for index pairing in the target factor
        # This is critical and depends on the index structure at each level
        if side == 0:
            # Absorbing rightward: new j-index always appears at LSB in next factor
            # So we always pair over bit 0
            bit_pos = 0
        else:
            # Absorbing leftward: the bit position bit position is level-1
            # if level > L // 2:
            #     bit_pos = L
            # else:
            bit_pos = level - 1
        
        # Create index pairs in the target factor
        mask = 1 << bit_pos
        all_indices = np.arange(M)
        idx0 = all_indices[(all_indices >> bit_pos) & 1 == 0]
        idx1 = idx0 | mask
        
        if next_factor.ndim == 3:
            # ----- 3D → 3D: Inner to Inner -----
            _, R_left, R_right = next_factor.shape
            
            if side == 0:
                # R @ next: (M/2, k, R_left) @ (M/2, R_left, R_right) -> (M/2, k, R_right)
                k = R_matrices.shape[1]
                
                if np.issubdtype(next_factor.dtype, np.floating):
                    new_next = np.zeros((M, k, R_right), dtype=np.float64)
                else:
                    new_next = np.zeros((M, k, R_right), dtype=np.complex128)
                
                # R[b] applies to both next[idx0[b]] and next[idx1[b]]
                new_next[idx0] = np.matmul(R_matrices, next_factor[idx0])
                new_next[idx1] = np.matmul(R_matrices, next_factor[idx1])
            
            else:
                # prev @ L: (M/2, R_left, R_right) @ (M/2, R_right, k) -> (M/2, R_left, k)
                k = R_matrices.shape[2]
                
                if np.issubdtype(next_factor.dtype, np.floating):
                    new_next = np.zeros((M, R_left, k), dtype=np.float64)
                else:
                    new_next = np.zeros((M, R_left, k), dtype=np.complex128)
                
                # L[b] applies to both prev[idx0[b]] and prev[idx1[b]]
                new_next[idx0] = np.matmul(next_factor[idx0], R_matrices)
                new_next[idx1] = np.matmul(next_factor[idx1], R_matrices)
            
            return new_next
        
        elif next_factor.ndim == 2:
            # ----- 3D → 2D: Inner to Outer -----
            # next_factor: (N, R_outer)
            
            N, R_outer = next_factor.shape
            c = N // (2 ** L)
            M_outer = N // c  # Should equal M_half
            
            # Reshape to (M_outer, c, R_outer) for block operations
            next_reshaped = next_factor.reshape(M_outer, c, R_outer)
            
            if side == 0:
                # outer @ R^T: (M_outer, c, R_outer) @ (M_half, R_outer, k) -> (M_outer, c, k)
                k = R_matrices.shape[1]
                R_T = R_matrices.transpose(0, 2, 1)
                new_next = np.matmul(next_reshaped, R_T)
            else:
                # outer @ L: (M_outer, c, R_outer) @ (M_half, R_outer, k) -> (M_outer, c, k)
                k = R_matrices.shape[2]
                new_next = np.matmul(next_reshaped, R_matrices)
            
            return new_next.reshape(M_outer * c, k)


def orthogonalize_sweep(factors, L, wrt_side):
    """
    Orthogonalize all factors from one side, absorbing remainder into the last factor.
    
    For side=0 (left to right):
        - QR factors 0, 1, ..., L in order
        - Absorb each R into the next factor
        - Final factor (L+1) contains all absorbed R matrices
        
    For side=1 (right to left):
        - LQ factors L+1, L, ..., 1 in order
        - Absorb each L into the previous factor
        - First factor (0) contains all absorbed L matrices
    
    Parameters:
    -----------
    factors : list of ndarrays
        factors[0]: left outer, shape (N, R_1), 2D
        factors[1] to factors[L]: inner factors, shape (M, R_i, R_{i+1}), 3D
        factors[L+1]: right outer, shape (N, R_{L+2}), 2D
    L : int
        Number of inner levels (len(factors) = L + 2)
    side : int, 0 or 1
        - side = 1: sweep left to right
        - side = 0: sweep right to left
    
    Returns:
    --------
    new_factors : list of ndarrays
        Orthogonalized factors with remainder absorbed into the last
    """
    new_factors = [f.copy() for f in factors]
    
    if wrt_side == 1:
        # Left to right: QR levels 0, 1, 2, ..., L
        # Absorb final R into level L+1
        for level in range(L + 1):
            Q, R_matrices = qr_factor_flat(new_factors[level], L, level, side=0)
            new_factors[level] = Q
            new_factors[level + 1] = absorb_R(R_matrices, new_factors[level + 1], L, level, side=0)
    
    else:
        # Right to left: LQ levels L+1, L, L-1, ..., 1
        # Absorb final L into level 0
        for level in range(L + 1, 0, -1):
            Q, L_matrices = qr_factor_flat(new_factors[level], L, level, side=1)
            new_factors[level] = Q
            new_factors[level - 1] = absorb_R(L_matrices, new_factors[level - 1], L, level, side=1)
    
    return new_factors


def reconstruct_sparse_single(inds, tensor_lst, level, L):
    """Reconstruct values for entries sharing a single unique index at 'level'."""
    
    if level == 0:
        H = tensor_lst[L+1][inds[:, L+1]].copy()
        for i in range(L, 0, -1):
            H = np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H, optimize=True)
        # Final contraction with level-0 core (which is 1D: shape (r,))
        # But we need the unique index value - this is handled in batched version
        return H  # Return H, contraction done outside
    
    elif level == L + 1:
        H = tensor_lst[0][inds[:, 0]].copy()
        for i in range(1, L + 1):
            H = np.einsum('ir,irz->iz', H, tensor_lst[i][inds[:, i]], optimize=True)
        return H
    
    else:
        H1 = tensor_lst[0][inds[:, 0]].copy()
        for i in range(1, level):
            H1 = np.einsum('ir,irz->iz', H1, tensor_lst[i][inds[:, i]], optimize=True)
        
        H2 = tensor_lst[L+1][inds[:, L+1]].copy()
        for i in range(L, level, -1):
            H2 = np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H2, optimize=True)
        
        return H1, H2


def estimate_reconstruct_memory(counts, tensor_lst, level, L):
    """Estimate memory needed for H matrices in reconstruction."""
    total_entries = np.sum(counts)
    itemsize = np.dtype(tensor_lst[0].dtype).itemsize
    
    if level == 0:
        # H shape: (n_entries, r) where r is rank at level 1
        r = tensor_lst[1].shape[0] if len(tensor_lst[1].shape) > 1 else tensor_lst[1].shape[-1]
        return total_entries * r * itemsize
    elif level == L + 1:
        r = tensor_lst[L].shape[-1] if len(tensor_lst[L].shape) > 2 else tensor_lst[L].shape[-1]
        return total_entries * r * itemsize
    else:
        # Need H1 and H2 simultaneously
        r1 = tensor_lst[level].shape[1] if len(tensor_lst[level].shape) > 2 else tensor_lst[level].shape[0]
        r2 = tensor_lst[level].shape[-1] if len(tensor_lst[level].shape) > 2 else tensor_lst[level].shape[-1]
        return total_entries * (r1 + r2) * itemsize


def reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups, tensor_lst, level, L, memory_fraction=0.5):
    """Memory-aware sparse reconstruction with automatic batching."""
    
    num_tuples = len(inds_tups)
    
    if np.issubdtype(tensor_lst[0].dtype, np.floating):
        Xs = np.zeros(nnz, dtype=np.float64)
    else:
        Xs = np.zeros(nnz, dtype=np.complex128)
    
    # Estimate memory
    estimated_mem = estimate_reconstruct_memory(counts, tensor_lst, level, L)
    available_mem = get_available_memory(memory_fraction)
    
    if estimated_mem < available_mem:
        # Original fast path
        Xs = _reconstruct_sparse_butterfly_full(unqs, starts, counts, nnz, inds_tups, tensor_lst, level, L)
    else:
        # Batched path
        batch_size = _compute_reconstruct_batch_size(counts, tensor_lst, level, L, memory_fraction)
        print(f"  [Reconstruct] Memory limit: batching {num_tuples} unique indices in batches of {batch_size}")
        
        for batch_start in range(0, num_tuples, batch_size):
            batch_end = min(batch_start + batch_size, num_tuples)
            
            if level == 0:
                # Compute H for this batch
                for idx in range(batch_start, batch_end):
                    inds = inds_tups[idx]
                    H = tensor_lst[L+1][inds[:, L+1]].copy()
                    for i in range(L, 0, -1):
                        H = np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H, optimize=True)
                    Xs[starts[idx]: starts[idx] + counts[idx]] = np.einsum('iz,z->i', H, tensor_lst[level][unqs[idx]], optimize=True)
                    del H
            
            elif level == L + 1:
                for idx in range(batch_start, batch_end):
                    inds = inds_tups[idx]
                    H = tensor_lst[0][inds[:, 0]].copy()
                    for i in range(1, L + 1):
                        H = np.einsum('ir,irz->iz', H, tensor_lst[i][inds[:, i]], optimize=True)
                    Xs[starts[idx]: starts[idx] + counts[idx]] = np.einsum('iz,z->i', H, tensor_lst[level][unqs[idx]], optimize=True)
                    del H
            
            else:
                for idx in range(batch_start, batch_end):
                    inds = inds_tups[idx]
                    
                    H1 = tensor_lst[0][inds[:, 0]].copy()
                    for i in range(1, level):
                        H1 = np.einsum('ir,irz->iz', H1, tensor_lst[i][inds[:, i]], optimize=True)
                    
                    H2 = tensor_lst[L+1][inds[:, L+1]].copy()
                    for i in range(L, level, -1):
                        H2 = np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H2, optimize=True)
                    
                    Xs[starts[idx]: starts[idx] + counts[idx]] = np.einsum(
                        'ir,iz,rz->i', H1, H2, tensor_lst[level][unqs[idx], :, :], optimize=True
                    )
                    del H1, H2
    
    return Xs


def _reconstruct_sparse_butterfly_full(unqs, starts, counts, nnz, inds_tups, tensor_lst, level, L):
    """Original non-batched reconstruction (when memory is sufficient)."""
    
    if np.issubdtype(tensor_lst[0].dtype, np.floating):
        Xs = np.zeros(nnz, dtype=np.float64)
    else:
        Xs = np.zeros(nnz, dtype=np.complex128)
    
    if level == 0:
        H = [tensor_lst[L+1][inds[:, L+1]] for inds in inds_tups]
        for i in range(L, 0, -1):
            H = [np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H[j], optimize=True) 
                 for j, inds in enumerate(inds_tups)]
        for i in range(len(counts)):
            Xs[starts[i]: starts[i] + counts[i]] = np.einsum('iz,z->i', H[i], tensor_lst[level][unqs[i]], optimize=True)
    
    elif level == L + 1:
        H = [tensor_lst[0][inds[:, 0]] for inds in inds_tups]
        for i in range(1, L + 1):
            H = [np.einsum('ir,irz->iz', H[j], tensor_lst[i][inds[:, i]], optimize=True) 
                 for j, inds in enumerate(inds_tups)]
        for i in range(len(counts)):
            Xs[starts[i]: starts[i] + counts[i]] = np.einsum('iz,z->i', H[i], tensor_lst[level][unqs[i]], optimize=True)
    
    else:
        H1 = [tensor_lst[0][inds[:, 0]] for inds in inds_tups]
        H2 = [tensor_lst[L+1][inds[:, L+1]] for inds in inds_tups]
        
        for i in range(1, level):
            H1 = [np.einsum('ir,irz->iz', H1[j], tensor_lst[i][inds[:, i]], optimize=True) 
                  for j, inds in enumerate(inds_tups)]
        for i in range(L, level, -1):
            H2 = [np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H2[j], optimize=True) 
                  for j, inds in enumerate(inds_tups)]
        
        for i in range(len(counts)):
            Xs[starts[i]: starts[i] + counts[i]] = np.einsum(
                'ir,iz,rz->i', H1[i], H2[i], tensor_lst[level][unqs[i], :, :], optimize=True
            )
    
    return Xs


def _compute_reconstruct_batch_size(counts, tensor_lst, level, L, memory_fraction=0.5):
    """Compute batch size for reconstruction based on available memory."""
    available = get_available_memory(memory_fraction)
    itemsize = np.dtype(tensor_lst[0].dtype).itemsize
    
    avg_count = np.mean(counts)
    
    if level == 0:
        r = tensor_lst[1].shape[-1] if len(tensor_lst[1].shape) > 1 else tensor_lst[0].shape[-1]
        mem_per_unique = avg_count * r * itemsize
    elif level == L + 1:
        r = tensor_lst[L].shape[-1]
        mem_per_unique = avg_count * r * itemsize
    else:
        r1 = tensor_lst[level].shape[1] if len(tensor_lst[level].shape) > 2 else tensor_lst[level].shape[0]
        r2 = tensor_lst[level].shape[-1]
        mem_per_unique = avg_count * (r1 + r2) * itemsize
    
    # Safety factor for intermediates
    mem_per_unique *= 3
    
    batch_size = max(1, int(available / mem_per_unique))
    return min(batch_size, len(counts))


def compute_error_sparse(T, inds, tensor_lst, L, no_batch_lr=False, memory_fraction=0.5, returnmore=None):
    """Memory-aware error computation with automatic batching."""
    
    level = 0
    sorted_tuples, T_new = sort_inds_and_T(inds, T, level)
    nnz = len(sorted_tuples)
    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index=True, return_counts=True)
    
    if no_batch_lr:
        # Original no_batch_lr path - for matrix completion when rank is very large
        recon = np.zeros_like(T_new)

        for i in range(len(unqs)):
            inds_for_row = sorted_tuples[starts[i]: starts[i] + counts[i]]
            H1 = tensor_lst[-1][inds_for_row[:, L+1]]      # N x R 
            H2 = tensor_lst[0][inds_for_row[:, 0]]         # N x R

            recon[starts[i]: starts[i] + counts[i]] = np.einsum('ir,ir->i', H1, H2, optimize=True)
        
        if returnmore is not None:
            return la.norm(T_new - recon) / la.norm(T_new), sorted_tuples, recon
        else:
            return la.norm(T_new - recon) / la.norm(T_new)
    
    # Memory-aware path
    estimated_mem = estimate_reconstruct_memory(counts, tensor_lst, level, L)
    available_mem = get_available_memory(memory_fraction)
    
    if estimated_mem < available_mem:
        # Original fast path
        inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
        recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups, tensor_lst, level, L, memory_fraction)
    else:
        # Batched path - compute reconstruction incrementally
        print(f"  [Error] Memory limit: using batched reconstruction")
        recon = np.zeros_like(T_new)
        
        batch_size = _compute_reconstruct_batch_size(counts, tensor_lst, level, L, memory_fraction)
        num_unqs = len(unqs)
        
        for batch_start in range(0, num_unqs, batch_size):
            batch_end = min(batch_start + batch_size, num_unqs)
            
            for idx in range(batch_start, batch_end):
                inds_for_row = sorted_tuples[starts[idx]: starts[idx] + counts[idx]]
                
                # Compute reconstruction for this unique index
                H = tensor_lst[L+1][inds_for_row[:, L+1]].copy()
                for i in range(L, 0, -1):
                    H = np.einsum('irz,iz->ir', tensor_lst[i][inds_for_row[:, i]], H, optimize=True)
                
                recon[starts[idx]: starts[idx] + counts[idx]] = np.einsum(
                    'iz,z->i', H, tensor_lst[level][unqs[idx]], optimize=True
                )
                del H
    
    if returnmore is not None:
        return la.norm(T_new - recon) / la.norm(T_new), sorted_tuples, recon
    else:
        return la.norm(T_new - recon) / la.norm(T_new)


def reconstruct_sparse_from_tensorlist(inds, tensor_lst, L, memory_fraction=0.5):
    """Memory-aware reconstruction from tensor list."""
    
    level = 0
    sorted_indices = np.argsort(inds[:, level])
    sorted_tuples = inds[sorted_indices]
    nnz = len(sorted_tuples)
    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index=True, return_counts=True)
    inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    
    recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups, tensor_lst, level, L, memory_fraction)
    
    return sorted_tuples, recon





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



def get_fullmat_from_sparse(T, inds, I, J):

    if np.issubdtype(T.dtype, np.floating):
        mat = np.zeros((I, J), dtype = np.float64)
    else:
        mat = np.zeros((I, J), dtype = np.complex128)

    for idx in range(len(inds)):
        mat[inds[idx][0],inds[idx][1]] = T[idx]
    return mat 


def get_masked_fullmat_from_sparse(T, inds, I, J):
    if np.issubdtype(T.dtype, np.floating):
        mat = np.zeros((I, J), dtype = np.float64)
    else:
        mat = np.zeros((I, J), dtype = np.complex128)
    
    mask = np.zeros_like(mat, dtype=bool)
    mask[:] = True
    for idx in range(len(inds)):
        mat[inds[idx][0],inds[idx][1]] = T[idx]
        mask[inds[idx][0],inds[idx][1]] = False
    mat_masked = np.ma.masked_array(mat, mask=mask)
    return mat_masked 




# def compute_sparse_butterfly(inds, tensor_lst, L):
#     vecs = tensor_lst[0][inds[:, 0]]
#     for i in range(1,L+1):
#         vecs = np.einsum('ir,irz->iz',vecs,tensor_lst[i][inds[:,i]],optimize=True)

#     return np.einsum('iz,iz->i',vecs,tensor_lst[L+1][inds[:,L+1]],optimize=True)



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



def estimate_H_memory(nnz, row_shape, dtype):
    """Estimate memory needed for H matrices in bytes."""
    itemsize = np.dtype(dtype).itemsize
    return nnz * row_shape * itemsize

def get_available_memory(fraction=0.7):
    """Get available memory in bytes, using only a fraction to be safe."""
    return int(psutil.virtual_memory().available * fraction)

def compute_batch_size(counts, row_shape, dtype, memory_fraction=0.5):
    """
    Compute how many unique indices we can process at once.
    Returns number of unique indices per batch.
    """
    available = get_available_memory(memory_fraction)
    itemsize = np.dtype(dtype).itemsize
    
    # Estimate memory per unique index (average entries per unique × row_shape)
    avg_count = np.mean(counts)
    mem_per_unique = avg_count * row_shape * itemsize
    
    # Account for multiple arrays (H, LHS, RHS, intermediates) - use 4x safety factor
    mem_per_unique *= 4
    
    batch_size = max(1, int(available / mem_per_unique))
    return min(batch_size, len(counts))  # Don't exceed total unique indices


def multiply_mats_single(inds, tensor_lst, level, L, row_shape):
    """Compute H matrix for a single group of indices sharing the same level index."""
    
    if level == 0:
        H = tensor_lst[L+1][inds[:, L+1]].copy()
        for i in range(L, 0, -1):
            H = np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H, optimize=True)
    
    elif level == L + 1:
        H = tensor_lst[0][inds[:, 0]].copy()
        for i in range(1, L + 1):
            H = np.einsum('ir,irz->iz', H, tensor_lst[i][inds[:, i]], optimize=True)
    
    else:
        H1 = tensor_lst[0][inds[:, 0]].copy()
        for i in range(1, level):
            H1 = np.einsum('ir,irz->iz', H1, tensor_lst[i][inds[:, i]], optimize=True)
        
        H2 = tensor_lst[L+1][inds[:, L+1]].copy()
        for i in range(L, level, -1):
            H2 = np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H2, optimize=True)
        
        H = np.einsum('ir,iz->irz', H1, H2, optimize=True).reshape(len(inds), row_shape)
        del H1, H2
    
    return H


def multiply_mats_batched(inds_tups, tensor_lst, level, L, row_shape, batch_indices):
    """Compute H matrices for a batch of unique indices."""
    Hs = []
    for i in batch_indices:
        H = multiply_mats_single(inds_tups[i], tensor_lst, level, L, row_shape)
        Hs.append(H)
    return Hs


def tensor_train_ALS_solve(T, inds, tensor_lst, level, L, regu, no_batch_lr=False, memory_fraction=0.5):
    """Memory-aware ALS solve with automatic batching."""
    
    if level == 0 or level == L + 1:
        row_shape = tensor_lst[level].shape[-1]
    else:
        row_shape = np.prod(tensor_lst[level].shape[1:])

    I = regu * np.eye(row_shape, dtype=tensor_lst[level].dtype)


    sorted_tuples, T_new = sort_inds_and_T(inds, T, level)
    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index=True, return_counts=True)
    
    num_unqs = len(unqs)
    total_nnz = len(sorted_tuples)

    if no_batch_lr:
        # Original no_batch_lr path - process one row at a time
        # This is only for matrix completion when rank is very large
        # and we have a lot of nonzeros
        for i in range(len(unqs)):
            LHS = np.zeros((row_shape, row_shape), dtype=T_new.dtype)
            RHS = np.zeros((row_shape), dtype=T_new.dtype)

            inds_for_row = sorted_tuples[starts[i]: starts[i] + counts[i]]
            if level == 0:
                H = tensor_lst[-1][inds_for_row[:, L+1]]        # N x R
            else:
                H = tensor_lst[0][inds_for_row[:, 0]]           # N x R

            LHS = np.dot(H.conj().T, H) + I                     # R x R

            
            RHS = np.dot(T_new[starts[i]: starts[i] + counts[i]], H.conj())  # R

            tensor_lst[level][unqs[i]] = la.solve(LHS, RHS)
        
        return tensor_lst

    # Memory-aware path
    inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    
    # Estimate memory and decide on batching
    estimated_mem = estimate_H_memory(total_nnz, row_shape, tensor_lst[level].dtype)
    available_mem = get_available_memory(memory_fraction)
    
    if estimated_mem < available_mem:
        # Original approach - process all at once
        Hs = multiply_mats(inds_tups, tensor_lst, level, L, row_shape)
        
        RHS = np.array([np.dot(T_new[starts[i]: starts[i] + counts[i]], Hs[i].conj()) 
                        for i in range(num_unqs)])
        LHS = np.array([np.dot(H.conj().T, H) + I for H in Hs])
        
        result = la.solve(LHS, RHS)
        
        if level == 0 or level == L + 1:
            tensor_lst[level][unqs] = result
        else:
            tensor_lst[level][unqs] = result.reshape((num_unqs,) + tensor_lst[level].shape[1:])
    
    else:
        # Batched approach
        batch_size = compute_batch_size(counts, row_shape, tensor_lst[level].dtype, memory_fraction)
        print(f"  [ALS] Memory limit: batching {num_unqs} unique indices in batches of {batch_size}")
        
        for batch_start in range(0, num_unqs, batch_size):
            batch_end = min(batch_start + batch_size, num_unqs)
            batch_indices = list(range(batch_start, batch_end))
            batch_unqs = unqs[batch_start:batch_end]
            
            # Compute H for this batch
            Hs = multiply_mats_batched(inds_tups, tensor_lst, level, L, row_shape, batch_indices)
            
            # Solve for this batch
            RHS = np.array([np.dot(T_new[starts[i]: starts[i] + counts[i]], Hs[j].conj()) 
                            for j, i in enumerate(batch_indices)])
            LHS = np.array([np.dot(H.conj().T, H) + I for H in Hs])
            
            result = la.solve(LHS, RHS)
            
            if level == 0 or level == L + 1:
                tensor_lst[level][batch_unqs] = result
            else:
                tensor_lst[level][batch_unqs] = result.reshape((len(batch_indices),) + tensor_lst[level].shape[1:])
            
            # Free memory
            del Hs, RHS, LHS, result
    
    return tensor_lst


def tensor_train_gradient(tensor, inds, tensor_lst, level, L, regu, memory_fraction=0.5):
    """Memory-aware gradient computation with automatic batching."""
    
    if level == 0 or level == L + 1:
        row_shape = tensor_lst[level].shape[-1]
    else:
        row_shape = np.prod(tensor_lst[level].shape[1:])

    sorted_tuples, tensor_new = sort_inds_and_T(inds, tensor, level)
    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index=True, return_counts=True)
    inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    
    num_unqs = len(unqs)
    total_nnz = len(sorted_tuples)
    
    # Pre-allocate gradient array
    grad_shape = (num_unqs,) + tensor_lst[level].shape[1:]
    neg_grad = np.zeros(grad_shape, dtype=tensor_lst[level].dtype)
    
    # Estimate memory and decide on batching
    estimated_mem = estimate_H_memory(total_nnz, row_shape, tensor_lst[level].dtype)
    available_mem = get_available_memory(memory_fraction)
    
    if estimated_mem < available_mem:
        # Original approach - process all at once
        Hs = multiply_mats(inds_tups, tensor_lst, level, L, row_shape)
        
        neg_grad_flat = np.array([np.dot(tensor_new[starts[i]: starts[i] + counts[i]], Hs[i].conj()) 
                                   for i in range(num_unqs)])
        neg_grad = neg_grad_flat.reshape(grad_shape)
    
    else:
        # Batched approach
        batch_size = compute_batch_size(counts, row_shape, tensor_lst[level].dtype, memory_fraction)
        print(f"  [Gradient] Memory limit: batching {num_unqs} unique indices in batches of {batch_size}")
        
        for batch_start in range(0, num_unqs, batch_size):
            batch_end = min(batch_start + batch_size, num_unqs)
            batch_indices = list(range(batch_start, batch_end))
            
            # Compute H for this batch
            Hs = multiply_mats_batched(inds_tups, tensor_lst, level, L, row_shape, batch_indices)
            
            # Compute gradient contributions for this batch
            for j, i in enumerate(batch_indices):
                grad_contrib = np.dot(tensor_new[starts[i]: starts[i] + counts[i]], Hs[j].conj())
                if level == 0 or level == L + 1:
                    neg_grad[i] = grad_contrib
                else:
                    neg_grad[i] = grad_contrib.reshape(tensor_lst[level].shape[1:])
            
            # Free memory
            del Hs
    
    neg_grad -= regu * tensor_lst[level]
    return neg_grad



def ADAM_butterfly(T_sparse, inds, T_test, inds_test, L, tensor_lst, 
    regu=1e-9, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-6):
    """
    Memory-optimized ADAM optimizer for butterfly factorization.
    """
    m = [np.zeros_like(x) for x in tensor_lst]
    v = [np.zeros_like(x) for x in tensor_lst]
    errors = []

    inds, T_sparse = sort_inds_and_T(inds, T_sparse, 0)
    unqs, starts, counts = np.unique(inds[:, 0], return_index=True, return_counts=True)
    inds_tups = [inds[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    nnz = len(T_sparse)
    
    # Precompute bias correction terms (updated iteratively)
    bias1 = 1.0
    bias2 = 1.0

    for t in range(1, max_iter + 1):
        recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups, tensor_lst, 0, L)
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
            g = tensor_train_gradient(residual, inds, tensor_lst, level, L, regu)
            
            # Track gradient norm for convergence
            g_norm = np.linalg.norm(g)
            max_grad_norm = max(max_grad_norm, g_norm)
            
            # Update moments IN-PLACE
            m[level] *= beta1
            m[level] += (1 - beta1) * g
            
            v[level] *= beta2
            # Handle complex case properly
            if np.iscomplexobj(g):
                v[level] += (1 - beta2) * (g * np.conj(g)).real
            else:
                v[level] += (1 - beta2) * (g ** 2)
            
            # Update parameters IN-PLACE (bias correction folded into lr_t)
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
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L)
        errors.append(error)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L)
        e1 = time.time()
        
        print('Time in error computation', e1 - s1)
        print('Total time in iteration', t, ':', grad_time)
        print('Relative error in observed entries:', error)
        print('Relative test error after', t, 'iterations:', test_error)
    
    print("Maximum iterations reached without convergence.")
    return tensor_lst, errors






def butterfly_tensor_train_completer(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu, no_batch_lr=False):
    if(L==0):
        print('------------------matrix completion----------------------------')
    else:
        print('------------------butterfly tensor train completion----------------------------')
    nnz = len(inds)
    print("Number of observed entries:",nnz)
    
    errors = []
    for iters in range(num_iters):
        s = time.time()
        print("Iteration", iters+1,"/",num_iters)

        for level in range(L+2):
            print('At level: ',level)
            tensor_lst = tensor_train_ALS_solve(T_sparse, inds, tensor_lst, level, L, regu=regu, no_batch_lr=no_batch_lr)
        
        e = time.time()
        print('Time in iteration', iters+1 ,':', e-s)
        
        s= time.time()
        #error = la.norm(T_sparse - compute_sparse_butterfly(inds, tensor_lst, L)) / la.norm(T_sparse)
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L,no_batch_lr=no_batch_lr)
        errors.append(error)
        #test_error = la.norm(T_test - compute_sparse_butterfly(inds_test,tensor_lst,L)) / la.norm(T_test)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L,no_batch_lr=no_batch_lr)
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


def butterfly_ADF(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu=0, no_batch_lr=False):
    print('------------------Butterfly ADF----------------------------')
    errors = []
    inds,  T_sparse = sort_inds_and_T(inds,  T_sparse, 0)
    unqs, starts, counts = np.unique(inds[:, 0], return_index = True, return_counts = True)
    inds_tups = [inds[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    nnz = len(T_sparse)
    print("Number of observed entries:",nnz)
    recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,0, L)
    grad_tensor = T_sparse - recon

    s = time.time()
    error = compute_error_sparse(T_sparse, inds, tensor_lst, L)
    errors.append(error)
    test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L)
    e = time.time()
    print('Time in error computation',e-s)
    print('Relative error in observed entries: ',error)


    for iters in range(num_iters):
        print("Iteration", iters+1,"/",num_iters)
        #Since we are going to start from 0 , we will orthogonalize all factors wrt first, i.e., absorb all weight into first
        tensor_lst = orthogonalize_sweep(tensor_lst, L, 0)
        grad_time = 0
        recon_time = 0
        factor_time = 0
        s = time.time()
        for level in range(len(tensor_lst)):
            print('At level: ',level)
            s_grad = time.time()
            N = tensor_train_gradient(grad_tensor, inds, tensor_lst, level, L, regu) # N as used in paper algo 5
            e_grad = time.time()
            
            grad_time += e_grad - s_grad

            new_lst = [t.copy() for t in tensor_lst] 
            new_lst[level] = N.copy()

            s_recon_time = time.time()
            Z = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,new_lst,0, L) # paper algo 5
            e_recon_time = time.time()

            recon_time += e_recon_time - s_recon_time

            delta_fac, delta_grad = Update_fac_and_grad(N, Z, inds, level, L)
            # alpha = la.norm(N)**2 / (la.norm(Z))**2 # We may need to change alpha for each slice in the level
            # delta_fac = alpha*N
            # delta_grad = alpha*Z    

            tensor_lst[level] += delta_fac
            grad_tensor -= delta_grad


            # Now we have to orthogonalize wrt the next level
            # This only requires one orthogonalization
            s_factor = time.time()
            if level != len(tensor_lst)-1:
                output, R_fac = qr_factor_flat(tensor_lst[level], L, level, 0)
                tensor_lst[level] = output
                tensor_lst[level+1] = absorb_R(R_fac, tensor_lst[level+1], L, level, 0)
            e_factor = time.time()

            factor_time += e_factor - s_factor
        
        e = time.time()
            

        print('Time in gradient computation', grad_time)
        print('Time in reconstruction computation', recon_time)
        print('Time in QR factor computation', factor_time)
        print('Time in iteration', iters+1 ,':', e-s)
        s= time.time()
        # Same arguments here as above
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L)
        errors.append(error)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L)
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




@njit
def _compute_H_level0(inds, end_tensor, mid_tensors, L):
    nnz = inds.shape[0]
    if L == 0:
        out_R = end_tensor.shape[1]
    else:
        out_R = mid_tensors[0].shape[1]

    H = np.empty((nnz, out_R), dtype=end_tensor.dtype)

    for row in range(nnz):
        h = end_tensor[inds[row, L + 1]].copy()
        for j in range(L, 0, -1):
            M = np.ascontiguousarray(mid_tensors[j - 1][inds[row, j]])
            h = np.dot(M, h)
        H[row] = h

    return H


@njit
def _compute_H_levelLp1(inds, start_tensor, mid_tensors, L):
    nnz = inds.shape[0]
    if L == 0:
        out_R = start_tensor.shape[1]
    else:
        out_R = mid_tensors[L - 1].shape[2]

    H = np.empty((nnz, out_R), dtype=start_tensor.dtype)

    for row in range(nnz):
        h = start_tensor[inds[row, 0]].copy()
        for j in range(1, L + 1):
            M = np.ascontiguousarray(mid_tensors[j - 1][inds[row, j]])
            h = np.dot(h, M)
        H[row] = h

    return H


@njit
def _compute_H_middle(inds, start_tensor, end_tensor, mid_tensors, level, L):
    nnz = inds.shape[0]

    if level == 1:
        R1 = start_tensor.shape[1]
    else:
        R1 = mid_tensors[level - 2].shape[2]

    if level == L:
        R2 = end_tensor.shape[1]
    else:
        R2 = mid_tensors[level].shape[1]

    row_shape = R1 * R2
    H = np.empty((nnz, row_shape), dtype=start_tensor.dtype)

    for row in range(nnz):
        h1 = start_tensor[inds[row, 0]].copy()
        for j in range(1, level):
            M = np.ascontiguousarray(mid_tensors[j - 1][inds[row, j]])
            h1 = np.dot(h1, M)

        h2 = end_tensor[inds[row, L + 1]].copy()
        for j in range(L, level, -1):
            M = np.ascontiguousarray(mid_tensors[j - 1][inds[row, j]])
            h2 = np.dot(M, h2)

        H[row] = np.outer(h1, h2).ravel()

    return H


@njit
def _reconstruct_all(inds, start_tensor, end_tensor, mid_tensors, L):
    nnz = inds.shape[0]
    recon = np.empty(nnz, dtype=start_tensor.dtype)

    for row in range(nnz):
        h = start_tensor[inds[row, 0]].copy()
        for j in range(1, L + 1):
            M = np.ascontiguousarray(mid_tensors[j - 1][inds[row, j]])
            h = np.dot(h, M)
        recon[row] = np.dot(h, end_tensor[inds[row, L + 1]])

    return recon





def _make_mid_list(tensor_lst, L):
    """Build a numba typed list of the middle tensors."""
    if L == 0:
        dtype = tensor_lst[0].dtype
        nb_dtype = numba.from_dtype(dtype)
        mid = NumbaList.empty_list(
            item_type=nb_dtype[:, :, :]
        )
    else:
        mid = NumbaList()
        for i in range(1, L + 1):
            mid.append(np.ascontiguousarray(tensor_lst[i]))
    return mid


class FastTTComputer:
    def __init__(self, tensor_lst, L):
        self.L = L
        self._set_tensors(tensor_lst)
        self._warmup()

    def _set_tensors(self, tensor_lst):
        self.tensor_lst = tensor_lst
        self.start_tensor = np.ascontiguousarray(tensor_lst[0])
        self.end_tensor = np.ascontiguousarray(tensor_lst[self.L + 1])
        self.mid_tensors = _make_mid_list(tensor_lst, self.L)

    def _warmup(self):
        dummy = np.zeros((1, self.L + 2), dtype=np.int64)

        _ = _compute_H_level0(dummy, self.end_tensor, self.mid_tensors, self.L)
        _ = _compute_H_levelLp1(dummy, self.start_tensor, self.mid_tensors, self.L)
        if self.L > 0:
            _ = _compute_H_middle(dummy, self.start_tensor, self.end_tensor,
                                  self.mid_tensors, 1, self.L)
        _ = _reconstruct_all(dummy, self.start_tensor, self.end_tensor,
                             self.mid_tensors, self.L)

    def update_tensor(self, level, new_tensor):
        new_tensor = np.ascontiguousarray(new_tensor)
        self.tensor_lst[level] = new_tensor

        if level == 0:
            self.start_tensor = new_tensor
        elif level == self.L + 1:
            self.end_tensor = new_tensor
        else:
            self.mid_tensors[level - 1] = new_tensor

    def _compute_H_single(self, inds, level):
        if level == 0:
            return _compute_H_level0(inds, self.end_tensor, self.mid_tensors, self.L)
        elif level == self.L + 1:
            return _compute_H_levelLp1(inds, self.start_tensor, self.mid_tensors, self.L)
        else:
            return _compute_H_middle(inds, self.start_tensor, self.end_tensor,
                                     self.mid_tensors, level, self.L)

    def compute_H(self, inds, level, batch_size=None):
        inds = np.ascontiguousarray(inds.astype(np.int64))

        if batch_size is None or len(inds) <= batch_size:
            return self._compute_H_single(inds, level)

        H_list = []
        for start in range(0, len(inds), batch_size):
            end = min(start + batch_size, len(inds))
            inds_batch = np.ascontiguousarray(inds[start:end])
            H_list.append(self._compute_H_single(inds_batch, level))

        return np.vstack(H_list)

    def reconstruct(self, inds, batch_size=None):
        inds = np.ascontiguousarray(inds.astype(np.int64))

        if batch_size is None or len(inds) <= batch_size:
            return _reconstruct_all(inds, self.start_tensor, self.end_tensor,
                                    self.mid_tensors, self.L)

        recon_list = []
        for start in range(0, len(inds), batch_size):
            end = min(start + batch_size, len(inds))
            inds_batch = np.ascontiguousarray(inds[start:end])
            recon_list.append(_reconstruct_all(inds_batch, self.start_tensor,
                                               self.end_tensor,
                                               self.mid_tensors, self.L))

        return np.concatenate(recon_list)


def sort_inds_and_T_short(inds, T, level):
    """Sort indices and T values by the level column."""
    sort_idx = np.argsort(inds[:, level])
    return inds[sort_idx], T[sort_idx]


import psutil

def get_available_memory(fraction=0.7):
    """Get available memory in bytes, using only a fraction to be safe."""
    return int(psutil.virtual_memory().available * fraction)


def compute_batch_size(nnz, row_shape, dtype=np.float64, min_batch=1000, max_batch=500000):
    """
    Compute batch size based on available memory.
    
    Returns None if full H fits in memory, otherwise returns batch size.
    """
    itemsize = np.dtype(dtype).itemsize
    full_H_bytes = nnz * row_shape * itemsize
    available = get_available_memory(fraction=0.7)
    
    # If full H fits with 2x headroom, use full memory mode
    if full_H_bytes * 2 < available:
        return None
    
    # Otherwise compute batch size
    # Each batch needs: H_batch (batch_size x row_shape) + intermediate for H^T @ H
    bytes_per_row = row_shape * itemsize * 3  # 3x for safety
    batch_size = available // bytes_per_row
    
    return int(np.clip(batch_size, min_batch, max_batch))


def get_row_shape(tensor_lst, level, L):
    """Get the row shape for H at a given level."""
    if level == 0 or level == L + 1:
        return tensor_lst[level].shape[-1]
    else:
        return np.prod(tensor_lst[level].shape[1:])


def tensor_train_ALS_solve_fast(T, inds, tensor_lst, level, L, regu, computer):
    """
    ALS solve with automatic batching based on available memory.
    
    Logic:
    1. Compute how much memory full H matrix would need: nnz * row_shape * 8 bytes
    2. Check available RAM
    3. If full H fits with 2x headroom -> compute all at once (fast)
    4. Otherwise -> accumulate H^T H and H^T b in batches (low memory)
    """
    row_shape = get_row_shape(tensor_lst, level, L)
    I = regu * np.eye(row_shape, dtype=tensor_lst[level].dtype)
    
    sorted_tuples, T_new = sort_inds_and_T_short(inds, T, level)
    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index=True, return_counts=True)
    num_unqs = len(unqs)
    
    # Auto-determine batch size
    batch_size = compute_batch_size(len(sorted_tuples), row_shape, dtype=T_new.dtype)
    
    if batch_size is None:
        # FULL MEMORY MODE: compute entire H matrix at once
        # Fast because we do one big matrix multiply
        H_all = computer.compute_H(sorted_tuples, level)
        
        LHS = np.empty((num_unqs, row_shape, row_shape), dtype=T_new.dtype)
        RHS = np.empty((num_unqs, row_shape), dtype=T_new.dtype)
        
        for i in range(num_unqs):
            H = H_all[starts[i]:starts[i] + counts[i]]
            LHS[i] = H.conj().T @ H + I
            RHS[i] = T_new[starts[i]:starts[i] + counts[i]] @ H.conj()
    else:
        # BATCHED MODE: for each k, accumulate H^T H and H^T b in chunks
        # Never allocates full H, only H_batch of size (batch_size x row_shape)
        print(f"    [batched mode: batch_size={batch_size}]")
        
        LHS = np.empty((num_unqs, row_shape, row_shape), dtype=T_new.dtype)
        RHS = np.empty((num_unqs, row_shape), dtype=T_new.dtype)
        
        for i in range(num_unqs):
            start_k = starts[i]
            count_k = counts[i]
            
            # Accumulate normal equations: (H^T H) x = H^T b
            HtH = np.zeros((row_shape, row_shape), dtype=T_new.dtype)
            Htb = np.zeros(row_shape, dtype=T_new.dtype)
            
            for batch_start in range(0, count_k, batch_size):
                batch_end = min(batch_start + batch_size, count_k)
                
                inds_batch = np.ascontiguousarray(
                    sorted_tuples[start_k + batch_start:start_k + batch_end].astype(np.int64)
                )
                b_batch = T_new[start_k + batch_start:start_k + batch_end]
                
                H_batch = computer._compute_H_single(inds_batch, level)
                
                # Accumulate: H^T H += H_batch^T @ H_batch
                #             H^T b += H_batch^T @ b_batch
                HtH += H_batch.conj().T @ H_batch
                Htb += H_batch.conj().T @ b_batch
            
            LHS[i] = HtH + I
            RHS[i] = Htb
    
    result = la.solve(LHS, RHS)
    
    if level == 0 or level == L + 1:
        tensor_lst[level][unqs] = result
    else:
        tensor_lst[level][unqs] = result.reshape((num_unqs,) + tensor_lst[level].shape[1:])
    
    computer.update_tensor(level, tensor_lst[level])
    return tensor_lst


def tensor_train_gradient_fast(tensor, inds, tensor_lst, level, L, regu, computer):
    """
    Gradient computation using FastTTComputer with automatic batching.
    
    Logic mirrors tensor_train_ALS_solve_fast:
    1. If full H fits in memory -> compute all at once
    2. Otherwise -> batch over rows and accumulate H^T b
    """
    row_shape = get_row_shape(tensor_lst, level, L)
    
    sorted_tuples, T_new = sort_inds_and_T_short(inds, tensor, level)
    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index=True, return_counts=True)
    num_unqs = len(unqs)
    
    grad_shape = (num_unqs,) + tensor_lst[level].shape[1:]
    
    batch_size = compute_batch_size(len(sorted_tuples), row_shape, dtype=T_new.dtype)
    
    if batch_size is None:
        # FULL MEMORY MODE
        H_all = computer.compute_H(sorted_tuples, level)
        
        neg_grad_flat = np.empty((num_unqs, row_shape), dtype=T_new.dtype)
        for i in range(num_unqs):
            H = H_all[starts[i]:starts[i] + counts[i]]
            neg_grad_flat[i] = T_new[starts[i]:starts[i] + counts[i]] @ H.conj()
        
        neg_grad = neg_grad_flat.reshape(grad_shape)
    else:
        # BATCHED MODE
        print(f"    [gradient batched mode: batch_size={batch_size}]")
        
        neg_grad_flat = np.empty((num_unqs, row_shape), dtype=T_new.dtype)
        
        for i in range(num_unqs):
            start_k = starts[i]
            count_k = counts[i]
            
            Htb = np.zeros(row_shape, dtype=T_new.dtype)
            
            for batch_start in range(0, count_k, batch_size):
                batch_end = min(batch_start + batch_size, count_k)
                
                inds_batch = np.ascontiguousarray(
                    sorted_tuples[start_k + batch_start:start_k + batch_end].astype(np.int64)
                )
                b_batch = T_new[start_k + batch_start:start_k + batch_end]
                
                H_batch = computer._compute_H_single(inds_batch, level)
                
                Htb += b_batch @ H_batch.conj()
            
            neg_grad_flat[i] = Htb
        
        neg_grad = neg_grad_flat.reshape(grad_shape)
    
    neg_grad -= regu * tensor_lst[level][unqs]
    
    return neg_grad, unqs


def compute_error_sparse_fast(T, inds, tensor_lst, L, computer, returnmore=None):
    """Compute reconstruction error. Auto-batches if needed."""
    sorted_tuples, T_new = sort_inds_and_T_short(inds, T, 0)
    
    # Check if reconstruction fits in memory
    recon_bytes = len(sorted_tuples) * np.dtype(T_new.dtype).itemsize * 3
    available = get_available_memory(fraction=0.5)
    
    if recon_bytes < available:
        batch_size = None
    else:
        batch_size = int(available // (np.dtype(T_new.dtype).itemsize * 3))
        batch_size = max(1000, min(batch_size, 500000))
    
    recon = computer.reconstruct(sorted_tuples, batch_size=batch_size)
    error = la.norm(T_new - recon) / la.norm(T_new)
    
    if returnmore is not None:
        return error, sorted_tuples, recon
    return error


def butterfly_tensor_train_completion_v2(T_sparse, inds, T_test, inds_test, L, tensor_lst, 
                                            num_iters, tol, regu):
    """
    Tensor train completion with automatic memory management.
    
    Memory strategy:
    - Checks available RAM before each level
    - If full H matrix fits: compute all at once (faster)
    - If not: accumulate H^T H and H^T b in batches (uses less memory)
    
    This is mathematically equivalent - batching just splits:
        H^T H = sum_i (H_i^T H_i)
        H^T b = sum_i (H_i^T b_i)
    """
    if L == 0:
        print('------------------matrix completion----------------------------')
    else:
        print('------------------butterfly tensor train completion----------------------------')
    
    nnz = len(inds)
    print(f"Number of observed entries: {nnz}")
    print(f"Available memory: {get_available_memory()/1e9:.2f} GB")
    
    print("Initializing Numba (first call compiles)...")
    computer = FastTTComputer(tensor_lst, L)
    print("Done.")

    errors = []
    for iters in range(num_iters):
        s = time.time()
        print("Iteration", iters + 1, "/", num_iters)
        
        for level in range(L + 2):
            print(f'  Level {level}', end='')
            tensor_lst = tensor_train_ALS_solve_fast(
                T_sparse, inds, tensor_lst, level, L, regu, computer
            )
            print()
        
        e = time.time()
        print('Time in iteration', iters + 1, ':', e - s)
        
        s = time.time()
        error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L, computer)
        errors.append(error)
        test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L, computer)
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


def butterfly_ADF_v2(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu=0, no_batch_lr=False):
    print('------------------Butterfly ADF----------------------------')
    errors = []
    inds,  T_sparse = sort_inds_and_T_short(inds,  T_sparse, 0)
    unqs, starts, counts = np.unique(inds[:, 0], return_index = True, return_counts = True)
    inds_tups = [inds[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    nnz = len(T_sparse)
    print("Number of observed entries:",nnz)
    recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,0, L)
    grad_tensor = T_sparse - recon

    s = time.time()
    error = compute_error_sparse(T_sparse, inds, tensor_lst, L)
    errors.append(error)
    test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L)
    e = time.time()
    print('Time in error computation',e-s)
    print('Relative error in observed entries: ',error)


    for iters in range(num_iters):
        print("Iteration", iters+1,"/",num_iters)
        #Since we are going to start from 0 , we will orthogonalize all factors wrt first, i.e., absorb all weight into first
        tensor_lst = orthogonalize_sweep(tensor_lst, L, 0)
        grad_time = 0
        recon_time = 0
        factor_time = 0
        s = time.time()
        for level in range(len(tensor_lst)):
            print('At level: ',level)
            s_grad = time.time()
            N = tensor_train_gradient(grad_tensor, inds, tensor_lst, level, L, regu) # N as used in paper algo 5
            e_grad = time.time()
            
            grad_time += e_grad - s_grad

            new_lst = [t.copy() for t in tensor_lst] 
            new_lst[level] = N.copy()

            s_recon_time = time.time()
            Z = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,new_lst,0, L) # paper algo 5
            e_recon_time = time.time()

            recon_time += e_recon_time - s_recon_time

            delta_fac, delta_grad = Update_fac_and_grad(N, Z, inds, level, L)
            # alpha = la.norm(N)**2 / (la.norm(Z))**2 # We may need to change alpha for each slice in the level
            # delta_fac = alpha*N
            # delta_grad = alpha*Z    

            tensor_lst[level] += delta_fac
            grad_tensor -= delta_grad


            # Now we have to orthogonalize wrt the next level
            # This only requires one orthogonalization
            s_factor = time.time()
            if level != len(tensor_lst)-1:
                output, R_fac = qr_factor_flat(tensor_lst[level], L, level, 0)
                tensor_lst[level] = output
                tensor_lst[level+1] = absorb_R(R_fac, tensor_lst[level+1], L, level, 0)
            e_factor = time.time()

            factor_time += e_factor - s_factor
        
        e = time.time()
            

        print('Time in gradient computation', grad_time)
        print('Time in reconstruction computation', recon_time)
        print('Time in QR factor computation', factor_time)
        print('Time in iteration', iters+1 ,':', e-s)
        s= time.time()
        # Same arguments here as above
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L)
        errors.append(error)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L)
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


def butterfly_ADF_v2(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu=0):
    print('------------------Butterfly ADF----------------------------')
    
    nnz = len(T_sparse)
    print(f"Number of observed entries: {nnz}")
    print(f"Available memory: {get_available_memory()/1e9:.2f} GB")
    
    print("Initializing Numba (first call compiles)...")
    computer = FastTTComputer(tensor_lst, L)
    print("Done.")
    
    errors = []
    
    # Initial reconstruction and residual
    recon = computer.reconstruct(np.ascontiguousarray(inds.astype(np.int64)))
    grad_tensor = T_sparse - recon
    
    s = time.time()
    error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L, computer)
    errors.append(error)
    test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L, computer)
    e = time.time()
    print('Time in error computation', e - s)
    print('Relative error in observed entries:', error)
    
    for iters in range(num_iters):
        print("Iteration", iters + 1, "/", num_iters)
        
        # Orthogonalize all factors wrt first
        tensor_lst = orthogonalize_sweep(tensor_lst, L, 0)
        computer._set_tensors(tensor_lst)
        
        # Recompute residual after orthogonalization (tensors changed)
        recon = computer.reconstruct(np.ascontiguousarray(inds.astype(np.int64)))
        grad_tensor = T_sparse - recon
        
        grad_time = 0
        recon_time = 0
        factor_time = 0
        s = time.time()
        
        for level in range(len(tensor_lst)):
            print(f'  Level {level}', end='')
            
            s_grad = time.time()
            N, unqs = tensor_train_gradient_fast(grad_tensor, inds, tensor_lst, level, L, regu, computer)
            e_grad = time.time()
            grad_time += e_grad - s_grad
            
            # Build full N tensor (scatter unqs back)
            N_full = np.zeros_like(tensor_lst[level])
            N_full[unqs] = N
            
            new_lst = [t.copy() for t in tensor_lst]
            new_lst[level] = N_full.copy()
            
            # Reconstruct Z using a temporary computer
            s_recon_time = time.time()
            temp_computer = FastTTComputer.__new__(FastTTComputer)
            temp_computer.L = L
            temp_computer.tensor_lst = new_lst
            temp_computer.start_tensor = np.ascontiguousarray(new_lst[0])
            temp_computer.end_tensor = np.ascontiguousarray(new_lst[L + 1])
            temp_computer.mid_tensors = _make_mid_list(new_lst, L)
            
            Z = temp_computer.reconstruct(np.ascontiguousarray(inds.astype(np.int64)))
            e_recon_time = time.time()
            recon_time += e_recon_time - s_recon_time
            
            delta_fac, delta_grad = Update_fac_and_grad(N_full, Z, inds, level, L)
            
            tensor_lst[level] += delta_fac
            grad_tensor -= delta_grad
            
            # Update computer with modified tensor
            computer.update_tensor(level, tensor_lst[level])
            
            # QR orthogonalization
            s_factor = time.time()
            if level != len(tensor_lst) - 1:
                output, R_fac = qr_factor_flat(tensor_lst[level], L, level, 0)
                tensor_lst[level] = output
                tensor_lst[level + 1] = absorb_R(R_fac, tensor_lst[level + 1], L, level, 0)
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
        error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L, computer)
        errors.append(error)
        test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L, computer)
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


def ADAM_butterfly_v2(T_sparse, inds, T_test, inds_test, L, tensor_lst,
                      regu=1e-9, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-6):
    """
    Memory-optimized ADAM optimizer for butterfly factorization.
    Uses FastTTComputer for accelerated gradient and reconstruction.
    """
    print('------------------Butterfly ADAM----------------------------')

    nnz = len(T_sparse)
    print(f"Number of observed entries: {nnz}")
    print(f"Available memory: {get_available_memory()/1e9:.2f} GB")

    print("Initializing Numba (first call compiles)...")
    computer = FastTTComputer(tensor_lst, L)
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

        for level in range(len(tensor_lst)):
            print(f'  Level {level}', end='')

            g_partial, unqs = tensor_train_gradient_fast(residual, inds, tensor_lst, level, L, regu, computer)

            # Scatter into full gradient
            g = np.zeros_like(tensor_lst[level])
            g[unqs] = g_partial

            # Track gradient norm for convergence
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

            # Update computer with modified tensor
            computer.update_tensor(level, tensor_lst[level])

            del g
            print()

        del residual

        e = time.time()
        grad_time = e - s
        print('Time in gradient computation', grad_time)

        # Check convergence
        if max_grad_norm < tol:
            print(f"Converged in {t} iterations.")
            return tensor_lst, errors

        s1 = time.time()
        error = compute_error_sparse_fast(T_sparse, inds, tensor_lst, L, computer)
        errors.append(error)
        test_error = compute_error_sparse_fast(T_test, inds_test, tensor_lst, L, computer)
        e1 = time.time()

        print('Time in error computation', e1 - s1)
        print('Total time in iteration', t, ':', grad_time)
        print('Relative error in observed entries:', error)
        print('Relative test error after', t, 'iterations:', test_error)
        print('-----------------')

    print("Maximum iterations reached without convergence.")
    return tensor_lst, errors