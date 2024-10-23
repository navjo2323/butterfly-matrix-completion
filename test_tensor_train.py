import numpy as np
from butterfly_tensor_train import *
from butterfly_helper_functions import index_convert, gen_tensor_inputs, ALS_solve
from butterfly_decomposition import butterfly_decompose_low_rank
from kdtree_ordering import generate_kd_tree
import time
import math
import numpy.linalg as la



def numerical_rank(matrix, tol):
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Compute the threshold based on the relative tolerance
    threshold = tol * S[0]
    
    # Count the number of singular values greater than the threshold
    rank = np.sum(S > threshold)
    
    return rank




def butterfly_rank(matrix, block_size,tol):
    # Get the dimensions of the matrix
    rows, cols = matrix.shape
    
    # Check if the matrix can be evenly divided into block_size x block_size blocks
    if rows % block_size != 0 or cols % block_size != 0:
        raise ValueError("Matrix dimensions must be divisible by the block size")
    
    # Initialize a list to store the ranks of each block
    block_ranks = []
    
    # Loop over the matrix and partition it into blocks
    for i in range(0, rows, block_size):
        row_blocks = []
        for j in range(0, cols, block_size):
            # Extract the MxM block
            block = matrix[i:i + block_size, j:j + block_size]
            
            # Compute the numerical rank of the block
            rank = numerical_rank(block, tol)
            # print(int(i/block_size),int(j/block_size),rank)
            row_blocks.append(rank)
        block_ranks.append(row_blocks)
    
    return block_ranks


def is_perfect_square(n):
    return n == int(math.isqrt(n))**2

def get_greens_kernel(c, L, ppw, inds=None,real=1):
    # Validate inputs
    assert is_perfect_square(c), f"{c} should be a perfect square"
    assert L % 2 == 0, f"{L} is not an even number"

    # Calculate parameters
    denominator = np.sqrt(c) * 2**(int(L/2))
    wavelen = ppw / denominator
    ds = wavelen / ppw
    Nperdim = int(np.ceil(1.0 / ds))
    
    print('Generating Greens kernel with ppw', ppw)
    print("Number per dim is", Nperdim)
    
    # Generate point coordinates
    pts = [(x, y) for x in np.linspace(0, 1, Nperdim) for y in np.linspace(0, 1, Nperdim)]
    waven = 2 * np.pi / wavelen

    pts = generate_kd_tree(pts)
    
    if inds is None:
        # Initialize the Green's function matrix
        G = np.zeros((Nperdim * Nperdim, Nperdim * Nperdim))
        
        # Compute pairwise distances using broadcasting
        delta = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(delta**2, axis=-1) + 1)
        
        # Compute Green's function values
        if(real==1):
            G = (np.cos(-1 * waven * dist))/ dist
        else:
            G = (np.cos(-1 * waven * dist)+ 1j* np.sin(-1 * waven * dist))/ dist
        
        return G
    else:
        # Vectorized computation for T_sparse
        inds = np.array(inds)
        p = pts[inds[:, 0]]  # Extract points for ind_i
        q = pts[inds[:, 1]]  # Extract points for ind_j
        
        # Calculate distances using vectorized operations
        dist = np.sqrt(np.sum((p - q)**2, axis=1) + 1)
        
        # Compute T_sparse using vectorized operations
        if(real==1):
            T_sparse = (np.cos(-1 * waven * dist)) / dist
        else:
            T_sparse = (np.cos(-1 * waven * dist) + 1j*np.sin(-1 * waven * dist)) / dist
        
        return T_sparse




def get_2dradon_kernel(c, L, inds=None,real=1):
    # Validate inputs
    assert is_perfect_square(c), f"{c} should be a perfect square"
    assert L % 2 == 0, f"{L} is not an even number"

    # Calculate parameters
    Nperdim = int(np.sqrt(c) * 2**(int(L/2)))

    print('Generating 2D Radon kernel')
    print("Number per dim is", Nperdim)
    
    # Generate point coordinates
    pts = [(x/Nperdim, y/Nperdim) for x in range(0, Nperdim) for y in range(0, Nperdim)]

    pts = generate_kd_tree(pts)
    
    if inds is None:
        # Initialize the Green's function matrix
        G = np.zeros((Nperdim * Nperdim, Nperdim * Nperdim))
        
        x = pts
        x2 = x**2
        y = pts
        y = y*Nperdim-Nperdim/2.0
        y2 = y**2
        c1 = (2+np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1]))/16.0
        c1 = c1**2
        c2 = (2+np.cos(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1]))/16.0
        c2 = c2**2
        phi = np.sqrt(np.outer(c1,y2[:,0]) + np.outer(c2,y2[:,1])) + np.dot(x, y.T)

        # Compute Radon transform values
        if(real==1):
            G = np.cos(2*np.pi*phi)
        else:
            G = np.cos(2*np.pi*phi) + 1j*np.sin(2*np.pi*phi)    
        return G
    else:
        # Vectorized computation for T_sparse
        inds = np.array(inds)
        x = pts[inds[:, 0]]  # Extract points for ind_i
        y = pts[inds[:, 1]]  # Extract points for ind_j
        y = y * Nperdim - Nperdim / 2.0  # Adjust y coordinates
        y2 = y**2

        c1 = (2 + np.sin(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])) / 16.0
        c1 = c1**2
        c2 = (2 + np.cos(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])) / 16.0
        c2 = c2**2
        
        # Calculate phi ensuring shape is (num_pts,)
        phi = np.sqrt(c1 * y2[:, 0] + c2 * y2[:, 1]) + np.sum(x * y, axis=1)

        # Compute T_sparse using the new shape of phi
        if(real==1):
            T_sparse = np.cos(2 * np.pi * phi)
        else:
            T_sparse = np.cos(2 * np.pi * phi)  + 1j*np.sin(2 * np.pi * phi)    
        return T_sparse



def get_1dradon_kernel(c, L, inds=None,real=1):
    # Validate inputs
    assert is_perfect_square(c), f"{c} should be a perfect square"
    assert L % 2 == 0, f"{L} is not an even number"

    # Calculate parameters
    Nperdim = c * 2**(L)

    print('Generating 1D Radon kernel')
    print("Number per dim is", Nperdim)
    
    # Generate point coordinates
    pts = [[x/Nperdim] for x in range(0, Nperdim)]

    pts = generate_kd_tree(pts)
    
    if inds is None:
        # Initialize the Green's function matrix
        G = np.zeros((Nperdim , Nperdim ))
        
        x = pts
        y = pts
        y = y*Nperdim-Nperdim/2.0
        yabs = np.abs(y)
        c = (2+np.sin(2*np.pi*x))/8.0
        phi = np.dot(c, yabs.T) + np.dot(x, y.T)

        # Compute Radon transform values
        if(real==1):
            G = np.cos(2*np.pi*phi) 
        else:
            G = np.cos(2*np.pi*phi) + 1j*np.sin(2*np.pi*phi)
        
        return G
    else:
        # Vectorized computation for T_sparse
        inds = np.array(inds)
        x = pts[inds[:, 0]]  # Extract points for ind_i
        y = pts[inds[:, 1]]  # Extract points for ind_j
        y = y*Nperdim-Nperdim/2.0
        yabs = np.abs(y)
        c = (2+np.sin(2*np.pi*x))/8.0
        phi = x*y + c*yabs

        # Compute T_sparse using vectorized operations
        if(real==1):
            T_sparse = np.cos(2*np.pi*phi)
        else:
            T_sparse = np.cos(2*np.pi*phi) + 1j*np.sin(2*np.pi*phi)
        
        
        return T_sparse.reshape(-1)

rng = np.random.RandomState(np.random.randint(1000))

kernel=3 # 1: Green's function 2: 2D Radon transform 3: 1D Radon transform
real=0 # 1: real-valued kernels, 0: complex-valued kernels
get_true_rank=1
lowrank_only=0
errorcheck_lr2bf=1
c = 4 # 4 9
#Should be perfect square, 4 and 9 options

L = 10

#Should be even, becomes too slow after 10 for this version of code

tol=1e-3
ppw=10

lc = int(L/2) 
I = c*2**L
J = c*2**L


r_BF= 11
ranks_lr = [2*r_BF] # [r_BF*10]
if(lowrank_only==0):
    nnz = min(int(6*(r_BF)*I*np.log2(I)),I**2)
else:
    nnz = min(10*(ranks_lr[0])*I,I**2)
ranks = [r_BF for _ in range(L- L//2+1 )] 

for i in range(len(ranks)):
    if i==0:
        ranks[0] = min(ranks[0],c)
    else:
        ranks[i] = min(2*ranks[i-1],ranks[i])
print('ranks for butterfly completion are ', ranks)


if(kernel==1):
    if(real==1):
        print('Testing real-valued Green function')
    else:
        print('Testing complex-valued Green function')
elif(kernel==2):
    if(real==1):
        print('Testing real-valued 2D Radon transform')
    else:
        print('Testing complex-valued 2D Radon transform')
elif(kernel==3):
    if(real==1):
        print('Testing real-valued 1D Radon transform')
    else:
        print('Testing complex-valued 1D Radon transform')


if(get_true_rank==1):
    s = time.time()
    if(kernel==1):
        mat= get_greens_kernel(c,L,ppw=ppw,real=real)
    elif(kernel==2):
        mat= get_2dradon_kernel(c,L,real=real)
    elif(kernel==3):
        mat= get_1dradon_kernel(c,L,real=real)

    e = time.time()
    #np.save('greens_matN-48ppw15.npy',mat)

    #mat = np.load('greens_matN-48ppw15.npy')
    print('full mat generated of shape',I)
    print('--time in full mat generation:',e-s)

    s = time.time()

    true_r_lr = numerical_rank(mat,tol)
    e = time.time()
    print('--time in computing the LR rank of mat:',e-s, ' rank:',true_r_lr, ' with tolerence', tol)

    s = time.time()
    blocksize = int(c*2**(L/2))
    true_bf_ranks = butterfly_rank(mat,blocksize,tol)
    e = time.time()
    print('--time in computing the BF rank of mat:',e-s, ' min/max rank:',np.min(true_bf_ranks),np.max(true_bf_ranks), ' with tolerence', tol)





print('matrix shape is',I)
print('nnz is',nnz)
print('ratio of nonzeros is',nnz/(I*J))
s = time.time()
indices = create_inds(I, J, nnz,rng)
indices_test = create_inds(I, J, nnz,rng)
e = time.time()
print('--time in creating indices:',e-s)




s = time.time()
if(kernel==1):
    T_sparse = get_greens_kernel(c,L,ppw=ppw,inds=indices,real=real)
    T_sparse_test = get_greens_kernel(c,L,ppw=ppw,inds=indices_test,real=real)
elif(kernel==2):
    T_sparse = get_2dradon_kernel(c,L,inds=indices,real=real)
    T_sparse_test = get_2dradon_kernel(c,L,inds=indices_test,real=real)    
elif(kernel==3):
    T_sparse = get_1dradon_kernel(c,L,inds=indices,real=real)
    T_sparse_test = get_1dradon_kernel(c,L,inds=indices_test,real=real)    


e = time.time()
print('--time in entry generation:',e-s)




##### TEST CODE FOR GREENS ########
L_lr = 0
c_lr = I
num_iter_lr = 10
print('--matrix completion rank:',ranks_lr)

s = time.time()
inds_tt_lr = encode_tuples(indices, L_lr, c_lr)
inds_tt_test_lr = encode_tuples(indices_test, L_lr, c_lr)
e = time.time()
print('--time in index conversion:',e-s)

s= time.time()
tensor_lst_lr = gen_tensor_train_list(L_lr, c_lr, ranks_lr ,rng, real=real )
e = time.time()
print('--time to generate inputs for matrix completion',e-s)


s = time.time()
tensor_lst_lr = butterfly_tensor_train_completer(T_sparse, inds_tt_lr, T_sparse_test, inds_tt_test_lr, L_lr, tensor_lst_lr, num_iter_lr, tol, regu=1e-4)
left_mat = tensor_lst_lr[0]
right_mat = tensor_lst_lr[1].conj()
e = time.time()
print('--time for matrix completion',e-s)


if(lowrank_only==0):

    s = time.time()
    g_lst,h_lst = gen_tensor_inputs(I, J, L, L//2, ranks, rng)
    g_lst, h_lst = butterfly_decompose_low_rank(left_mat,right_mat,L,ranks,g_lst,h_lst,errorcheck_lr2bf)
    e= time.time()
    print('--time for low-rank to butterfly conversion', e-s)


    num_iters = 20

    s= time.time()
    g_lst3d, h_lst3d = convert_lst_to_3d(g_lst,h_lst, L, c)
    tensor_lst = make_one_list(g_lst3d,h_lst3d)
    e= time.time()
    print('--time for converting lists',e-s)

    s = time.time()
    inds_tt = encode_tuples(indices, L, c)
    inds_tt_test = encode_tuples(indices_test, L, c)
    e = time.time()
    print('--time in index conversion 2 :',e-s)

    s= time.time()
    tensor_lst = butterfly_tensor_train_completer(T_sparse, inds_tt, T_sparse_test, inds_tt_test, L, tensor_lst, num_iters, tol, regu=1e-10)
    e= time.time()
    print('--time for butterfly completion',e-s)




### TEST CODE FOR GREENS #######



# #TEST CODE FOR CHECKING SOLVES ###########

# s = time.time()
# inds = index_convert(indices, I, J, L, c)
# e = time.time()
# print('--time in index conversion 1:',e-s)

# g_lst,h_lst = gen_tensor_inputs(I,J,L,L//2,ranks,rng)

# g_lst3d, h_lst3d = convert_lst_to_3d(g_lst,h_lst, L, c)

# tensor_lst = make_one_list(g_lst3d,h_lst3d)


# regu = 1e-7
# for iters in range(2):
#     for level in range(L,L//2-1,-1):
#         for r_c in range(2):
#             g_lst, h_lst = ALS_solve(T_sparse, inds, g_lst, h_lst, level, L, L//2, r_c, regu)
#             print(la.norm(h_lst[0]))

# print('---------------')

# for iters in range(10):
#     for level_tt in range(L+2):
#         tensor_lst = tensor_train_ALS_solve(T_sparse, inds_tt, tensor_lst, level_tt, L, regu=1e-7)
#     val = compute_sparse_butterfly(inds_tt, tensor_lst, L)
#     print('error norm is', (la.norm(T_sparse - val)/la.norm(T_sparse)))

# g_lst3d, h_lst3d = make_two_lists(tensor_lst)

# g_lst_tt, h_lst_tt = convert_lst_to_Nd(g_lst3d, h_lst3d, L, c)


# for i in range(len(h_lst)):
#     print(la.norm(g_lst[i] - g_lst_tt[i]))
#     print(la.norm(h_lst[i] - h_lst_tt[i]))

############### TEST CODE FOR CHECKING SOLVES ###############



