import itertools
import numpy as np
import numpy.linalg as la
import time
import scipy.linalg as sla
import logging


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
    
    For side=0 (left to right):
        - Pair matrices based on bit at position `level`
        - Concatenate along left rank (axis=1) to form (M/2, 2*R1, R2)
        - Standard QR decomposition: A = QR
        - Returns Q with orthonormal columns, R to absorb rightward
        
    For side=1 (right to left):
        - Pair matrices based on bit at position 0 (LSB)
        - Concatenate along right rank (axis=2) to form (M/2, R1, 2*R2)
        - LQ decomposition (via QR of transpose): A = LQ
        - Returns Q with orthonormal rows, L to absorb leftward
    
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
        - side = 0: sweep left to right
        - side = 1: sweep right to left
    
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
        
        N, R = factor.shape
        c = N // (2 ** L)
        M = N // c  # M = 2^L
        
        reshaped = factor.reshape(M, c, R)
        
        # Standard QR on each (c, R) block
        # Q: (M, c, k), R_matrices: (M, k, R) where k = min(c, R)
        Q, R_matrices = np.linalg.qr(reshaped, mode='reduced')
        
        output = Q.reshape(N, -1)  # (N, k)
        return output, R_matrices
    
    else:
        # ===== 3D CASE: Inner factors =====
        # Shape: (M, R1, R2) where M = 2^{L+1}
        # Each index m encodes (i_{L-level}, ..., i_0, j_0, ..., j_{level-1})
        
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
            Q, R_matrices = np.linalg.qr(concat, mode='reduced')
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
            
            # LQ decomposition via QR of transpose: A = LQ
            # A^T = Q^T L^T, so QR(A^T) gives us Q^T and L^T
            # concat^T: (M/2, 2*R2, R1)
            Qt, Lt = np.linalg.qr(concat.transpose(0, 2, 1), mode='reduced')
            # Qt: (M/2, 2*R2, k), Lt: (M/2, k, R1)
            # where k = min(2*R2, R1)
            
            # Transpose back to get L and Q
            Q = Qt.transpose(0, 2, 1)      # (M/2, k, 2*R2) - rows are orthonormal
            L_matrices = Lt.transpose(0, 2, 1)  # (M/2, R1, k)
            
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
            # Absorbing leftward: the bit position changes at the halfway point
            # For levels > L//2 (upper half): new i-index appears at MSB
            # For levels <= L//2 (lower half): bit position is level-1
            if level > L // 2:
                bit_pos = L
            else:
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


def orthogonalize_sweep(factors, L, side):
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
        - side = 0: sweep left to right
        - side = 1: sweep right to left
    
    Returns:
    --------
    new_factors : list of ndarrays
        Orthogonalized factors with remainder absorbed into the last
    """
    new_factors = [f.copy() for f in factors]
    
    if side == 0:
        # Left to right: QR levels 0, 1, 2, ..., L
        # Absorb final R into level L+1
        for level in range(L + 1):
            Q, R_matrices = qr_factor_flat(new_factors[level], L, level, side)
            new_factors[level] = Q
            new_factors[level + 1] = absorb_R(R_matrices, new_factors[level + 1], L, level, side)
    
    else:
        # Right to left: LQ levels L+1, L, L-1, ..., 1
        # Absorb final L into level 0
        for level in range(L + 1, 0, -1):
            Q, L_matrices = qr_factor_flat(new_factors[level], L, level, side)
            new_factors[level] = Q
            new_factors[level - 1] = absorb_R(L_matrices, new_factors[level - 1], L, level, side)
    
    return new_factors


def reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,level, L):

    num_tuples = len(inds_tups)
    if np.issubdtype(tensor_lst[0][0].dtype, np.floating):
        Xs = np.zeros(nnz, dtype= np.float64)
    else:
        Xs = np.zeros(nnz, dtype= np.complex128)


    # A conjugate needs to be done based on level    
    if level == 0:
        # Pre-compute indices for the last tensor
        H = [tensor_lst[L+1][inds[:, L+1]] for inds in inds_tups]

        # Iterate in reverse order and apply einsum
        for i in range(L, 0, -1):
            H = [np.einsum('irz,iz->ir', tensor_lst[i][inds[:, i]], H[j],optimize=True) for j, inds in enumerate(inds_tups)]

        for i in range(len(counts)):
            Xs[starts[i]: starts[i] + counts[i]] = np.einsum('iz,z->i',H[i],tensor_lst[level][unqs[i]],optimize=True)

    elif level == L + 1:
        # Pre-compute indices for the first tensor
        H = [tensor_lst[0][inds[:, 0]] for inds in inds_tups]

        # Iterate forwards and apply einsum
        for i in range(1, L + 1):
            H = [np.einsum('ir,irz->iz', H[j], tensor_lst[i][inds[:, i]],optimize=True) for j, inds in enumerate(inds_tups)]

        for i in range(len(counts)):
            Xs[starts[i]: starts[i] + counts[i]] = np.einsum('iz,z->i',H[i],tensor_lst[level][unqs[i]],optimize=True)

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
        for i in range(len(counts)):
            Xs[starts[i]: starts[i] + counts[i]] = np.einsum('ir,iz,rz->i',H1[i],H2[i],tensor_lst[level][unqs[i],:,:],optimize=True)
        
    return Xs


def compute_error_sparse(T, inds, tensor_lst, L, no_batch_lr=False, returnmore=None):

    level = 0
    s = time.time()

    sorted_tuples, T_new = sort_inds_and_T(inds, T, level)

    e = time.time()


    #print('Time in sorting',e-s)

    nnz = len(sorted_tuples)

    s = time.time()


    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index = True, return_counts = True)


    if no_batch_lr:
        # This is only for matrix completion when rank is very large
        # and we have a lot of nonzeros
        recon = np.zeros_like(T_new)

        for i in range(len(unqs)):

            inds_for_row = sorted_tuples[starts[i]: starts[i] + counts[i]]
            H1 = tensor_lst[-1][inds_for_row[:,L+1]]      # N x R 
            
            H2 = tensor_lst[0][inds_for_row[:,0]]         # N x R

            recon[starts[i]: starts[i] + counts[i] ] = np.einsum('ir,ir->i',H1,H2,optimize=True)


    else:

        inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]


        recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,level, L)
        
    if(returnmore is not None):
        return la.norm(T_new - recon)/la.norm(T_new), sorted_tuples, recon
    else: 
        return la.norm(T_new - recon)/la.norm(T_new)



def reconstruct_sparse_from_tensorlist(inds, tensor_lst, L):

    level = 0
    sorted_indices = np.argsort(inds[:, level])
    sorted_tuples = inds[sorted_indices]
    nnz = len(sorted_tuples)
    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index = True, return_counts = True)
    inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,level, L)

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

def tensor_train_ALS_solve(T, inds, tensor_lst, level, L, regu, no_batch_lr=False):

    if level ==0 or level == L + 1:
        row_shape = tensor_lst[level].shape[-1]
    else:
        row_shape = np.prod(tensor_lst[level].shape[1:])


    I = regu*np.eye(row_shape)

    s = time.time()

    sorted_tuples, T_new = sort_inds_and_T(inds, T, level)

    e = time.time()


    #print('Time in sorting',e-s)


    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index = True, return_counts = True)

    inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]


    if no_batch_lr:
        # This is only for matrix completion when rank is very large
        # and we have a lot of nonzeros
        for i in range(len(unqs)):
            LHS = np.zeros((row_shape,row_shape),dtype = T_new.dtype)
            RHS = np.zeros((row_shape),dtype = T_new.dtype)

            inds_for_row = sorted_tuples[starts[i]: starts[i] + counts[i]]
            if level == 0:
                H = tensor_lst[-1][inds_for_row[:,L+1]]        # N x R
            else:
                H = tensor_lst[0][inds_for_row[:,0]]         # N x R

            LHS = np.dot(H.conj().T,H) + I                                           # R x R
            RHS = np.dot(T_new[starts[i]: starts[i] + counts[i] ], H.conj())        # R

            tensor_lst[level][unqs[i]] = la.solve(LHS,RHS)


    else:

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


def tensor_train_gradient(tensor, inds, tensor_lst, level, L, regu):
    if level ==0 or level == L + 1:
        row_shape = tensor_lst[level].shape[-1]
    else:
        row_shape = np.prod(tensor_lst[level].shape[1:])

    s = time.time()

    sorted_tuples, tensor_new = sort_inds_and_T(inds, tensor, level)

    e = time.time()


    #print('Time in sorting',e-s)

    nnz = len(sorted_tuples)

    s = time.time()


    unqs, starts, counts = np.unique(sorted_tuples[:, level], return_index = True, return_counts = True)

    inds_tups = [sorted_tuples[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]


    #recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,level, L)


    #tensor = T_new - recon

    Hs = multiply_mats(inds_tups, tensor_lst, level, L, row_shape) 

    neg_grad = np.array([np.dot(tensor_new[starts[i]: starts[i] + counts[i] ], Hs[i].conj()) for i in range(len(unqs))])

    neg_grad = neg_grad.reshape( (len(unqs),) + tensor_lst[level].shape[1:])

    neg_grad -= regu*tensor_lst[level]

    return neg_grad



def ADAM_tensor_train(T_sparse, inds, T_test, inds_test, L, tensor_lst, 
    regu=1e-9, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100, tol=1e-6):
    """
    ADAM optimizer for unconstrained optimization.
    
    """
    m = [np.zeros_like(x) for x in tensor_lst]          # First moment vector (mean of gradients)
    v = [np.zeros_like(x) for x in tensor_lst]          # Second moment vector (uncentered variance of gradients)
    errors = []

    inds, T_sparse = sort_inds_and_T(inds, T_sparse, 0)
    unqs, starts, counts = np.unique(inds[:, 0], return_index = True, return_counts = True)
    inds_tups = [inds[starts[i]: starts[i] + counts[i]] for i in range(len(unqs))]
    nnz = len(T_sparse)

    for t in range(1, max_iter + 1):
        recon = reconstruct_sparse_butterfly(unqs, starts, counts, nnz, inds_tups,tensor_lst,0, L)
        tensor = T_sparse - recon

        s = time.time()
        grads = []
        
        for level in range(len(tensor_lst)):
            grads.append(tensor_train_gradient(tensor, inds, tensor_lst, level, L, regu))
        # Update biased first moment estimate
        m = [beta1*x + (1 - beta1)*g for x, g in zip(m, grads)]
        
        # Update biased second raw moment estimate
        v = [beta2*x + (1 - beta2)* (g**2) for x, g in zip(v,grads)]

        
        # Correct bias in first and second moments
        m_hat = [x / (1 - beta1 ** t) for x in m]
        v_hat = [x / (1 - beta2 ** t) for x in v]
        
        # Update parameters
        tensor_lst = [x + lr * x1 / (np.sqrt(x2) + epsilon) for x, x1, x2 in zip(tensor_lst, m_hat, v_hat)]

        e = time.time()
        print('Time in gradient computation', e-s)
        grad_time = e - s

        
        s = time.time()
        # Check convergence based on gradient norm
        if max([la.norm(g) for g in grads]) < tol:
            print(f"Converged in {t} iterations.")
            return tensor_lst
        e = time.time()


        s1= time.time()
        error = compute_error_sparse(T_sparse, inds, tensor_lst, L)
        errors.append(error)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L)
        e1 = time.time()
        print('Time in error computation',e1-s1)
        print('Total time in iteration', t, e-s + grad_time)
        print('Relative error in observed entries: ',error)
        print('Relative test error after', t,' iterations: ',test_error)
    print("Maximum iterations reached without convergence.")
    return tensor_lst



def butterfly_tensor_train_completer(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu, no_batch_lr=False):
    if(L==0):
        print('------------------matrix completion----------------------------')
    else:
        print('------------------butterfly/ tensor train completion----------------------------')
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




def butterfly_tensor_train_completer(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu, no_batch_lr=False):
    if(L==0):
        print('------------------matrix completion----------------------------')
    else:
        print('------------------butterfly/ tensor train completion----------------------------')
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


def butterfly_ADF(T_sparse, inds, T_test, inds_test, L, tensor_lst, num_iters, tol, regu, no_batch_lr=False):
    print('------------------Butterfly ADF----------------------------')
    nnz = len(inds)
    print("Number of observed entries:",nnz)
    iters = 0
    errors = []
    s = time.time()
    error = compute_error_sparse(T_sparse, inds, tensor_lst, L,no_batch_lr=no_batch_lr)
    errors.append(error)
    #test_error = la.norm(T_test - compute_sparse_butterfly(inds_test,tensor_lst,L)) / la.norm(T_test)
    test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L,no_batch_lr=no_batch_lr)
    e = time.time()
    print('Time in error computation',e-s)
    print('Relative error in observed entries: ',error)
    print('Relative test error after', iters + 1,' iterations: ',test_error)
    print('-----------------')
    #side = 0
    side = 1

    #for level in range(L + 1):
    for level in range(L + 1, 0, -1):
        print('At level: ',level)
        Q, R_matrices = qr_factor_flat(tensor_lst[level],  L, level, side)
        tensor_lst[level] = Q
        #tensor_lst[level + 1] = absorb_R(R_matrices, tensor_lst[level + 1], L, level, side)
        tensor_lst[level - 1] = absorb_R(R_matrices, tensor_lst[level - 1], L, level, side)
        s= time.time()

        error = compute_error_sparse(T_sparse, inds, tensor_lst, L,no_batch_lr=no_batch_lr)
        errors.append(error)
        #test_error = la.norm(T_test - compute_sparse_butterfly(inds_test,tensor_lst,L)) / la.norm(T_test)
        test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L,no_batch_lr=no_batch_lr)
        e = time.time()
        print('Time in error computation',e-s)
        print('Relative error in observed entries: ',error)
        print('Relative test error after', iters + 1,' iterations: ',test_error)
        print('-----------------')
    # for iters in range(num_iters):
    #     s = time.time()
    #     print("Iteration", iters+1,"/",num_iters)
        


    #     e = time.time()
    #     print('Time in iteration', iters+1 ,':', e-s)
        
    #     s= time.time()
    #     #error = la.norm(T_sparse - compute_sparse_butterfly(inds, tensor_lst, L)) / la.norm(T_sparse)
    #     error = compute_error_sparse(T_sparse, inds, tensor_lst, L,no_batch_lr=no_batch_lr)
    #     errors.append(error)
    #     #test_error = la.norm(T_test - compute_sparse_butterfly(inds_test,tensor_lst,L)) / la.norm(T_test)
    #     test_error = compute_error_sparse(T_test, inds_test, tensor_lst, L,no_batch_lr=no_batch_lr)
    #     e = time.time()
    #     print('Time in error computation',e-s)
    #     print('Relative error in observed entries: ',error)
    #     print('Relative test error after', iters + 1,' iterations: ',test_error)
    #     print('-----------------')
    #     if iters + 1 >= 5 and error >= 3:
    #         print('Overfitting or error not reducing, stopping iterations')
    #         break
    #     if error < tol:
    #         print('converged')
    #         break
    
    return tensor_lst