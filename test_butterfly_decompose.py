import numpy as np 
import numpy.linalg as la
from butterfly_helper_functions import *
from low_rank_test import *
from kdtree_ordering import generate_kd_tree
from butterfly_decomposition import butterfly_decompose_low_rank
import math
import time
import copy


def is_perfect_square(n):
    return n == int(math.isqrt(n))**2
# def get_kd_tree_ordering():
    

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
            row_blocks.append(rank)
        block_ranks.append(row_blocks)
    
    return block_ranks



def get_greens_kernel(c,L,ppw,inds=None):
    # Define the number of points and the wavenumber
    #wavelen = 0.35/(2 ** 2.5)
    
    #Nperdim^2 should become c*2^L
    # That is Nperdim = sqrt(c 2*L) should be a perfect square
    
    assert is_perfect_square(c), f"{c} should be a perfect square"
    
    assert  L % 2 == 0, f"{L} is not an even number"
    
    denominator = np.sqrt(c)* 2**(int(L/2))
    
    wavelen = ppw/denominator
    
    ds = wavelen/ppw
    Nperdim = int(np.ceil(1.0/ds))
    
    print('generating Greens kernel with ppw',ppw)
    
    print("number per dim is",Nperdim)
    
    Nperdimx = Nperdim
    Nperdimy = Nperdim


    pts = [(x,y) for x in np.linspace(0,1,Nperdimx) for y in np.linspace(0,1,Nperdimy)]
    
    waven = 2*np.pi/wavelen
    # Compute the Green's function matrix
    
    # pts_arr = np.array(pts)
    # # # usual ordering
    # for i in range(len(pts_arr)):
    #     for j in range(len(pts_arr)):
    #         dist = np.sqrt( np.sum((pts_arr[i] - pts_arr[j])**2 +1))
    #         G[i,j] = np.cos(-1 * waven* dist) / dist
    
    
    order_pts = generate_kd_tree(pts)

    if(inds is None):
        # Initialize the Green's function matrix
        G = np.zeros((Nperdimx*Nperdimy, Nperdimx*Nperdimy))
        i = 0
        for p in order_pts:
            j=0
            for q in order_pts:
                dist = np.sqrt(la.norm(p-q)**2 + 1)   #np.sqrt( np.sum((p - q)**2) +1)
                G[i,j] = np.cos(-1 * waven* dist) / dist
                j+=1
            i+=1
        return G
    else:
        T_sparse = np.zeros(len(inds))
        i=0
        for ind in inds:
            ind_i = ind[0]
            ind_j = ind[1]  
            p=order_pts[ind_i] 
            q=order_pts[ind_j]
            dist = np.sqrt(la.norm(p-q)**2 + 1)   #np.sqrt( np.sum((p - q)**2) +1)
            T_sparse[i] = np.cos(-1 * waven* dist) / dist             
            i+=1
        return T_sparse



def create_omega_from_indices(indices,I,J):
    Omega = np.zeros((I,J))
    for ind in indices:
        Omega[ind] = 1
    return Omega




rng = np.random.RandomState(np.random.randint(1000))

get_true_rank=0
c = 4
#Should be perfect square, 4 and 9 options

L = 10

#Should be even, becomes too slow after 10 for this version of code

tol=1e-3
ppw=10

lc = int(L/2) 
I = c*2**L
J = c*2**L


if(get_true_rank==1):
    s = time.time()
    mat= get_greens_kernel(c,L,ppw=ppw)
    e = time.time()
    #np.save('greens_matN-48ppw15.npy',mat)

    #mat = np.load('greens_matN-48ppw15.npy')
    print('greens mat generated of shape',I)
    print('--time in greens mat generation:',e-s)

    s = time.time()

    true_r_lr = numerical_rank(mat,tol)
    e = time.time()
    print('--time in computing the LR rank of mat:',e-s, ' rank:',true_r_lr, ' with tolerence', tol)

    s = time.time()
    blocksize = int(c*2**(L/2))
    true_bf_ranks = butterfly_rank(mat,blocksize,tol)
    e = time.time()
    print('--time in computing the BF rank of mat:',e-s, ' min/max rank:',np.min(true_bf_ranks),np.max(true_bf_ranks), ' with tolerence', tol)





m = I
n= m

r_BF= 10
ranks = [r_BF for _ in range(L-lc+1 )] 

for i in range(len(ranks)):
    if i==0:
        ranks[0] = min(ranks[0],c)
    else:
        ranks[i] = min(2*ranks[i-1],ranks[i])
print('ranks for butterfly completion are ', ranks)

# Can give assymetric ranks, if needed








nnz = 5*int((r_BF)*n*np.log2(n))

print('m*n is',m*n)
print('nnz is',nnz)
print('ratio of nonzeros is',nnz/(m*n))
s = time.time()
indices = create_inds(I, J, nnz,rng)
indices_test = create_inds(I, J, nnz,rng)
e = time.time()
print('--time in creating indices:',e-s)
s = time.time()
inds = index_convert(indices, I, J, L, c)
inds_test = index_convert(indices_test, I, J, L, c)
e = time.time()
print('--time in index conversion:',e-s)
s = time.time()
T_sparse = get_greens_kernel(c,L,ppw=ppw,inds=indices)
T_sparse_test = get_greens_kernel(c,L,ppw=ppw,inds=indices_test)
e = time.time()
print('--time in entry generation:',e-s)


num_iters = 1
L_LR = 0
r_LR = r_BF
ranks_LR = [r_LR]
inds_LR = index_convert(indices, I, J, L_LR,I)
inds_test_LR = index_convert(indices_test, I, J, L_LR,I)
T_sparse_LR = T_sparse
T_sparse_test_LR = T_sparse_test

print('Low-rank completion (initial guess for butterlfy completion) rank',r_LR)
s = time.time()
rng = np.random.RandomState(np.random.randint(1000))
g_lst,h_lst = gen_tensor_inputs(m,n,L_LR,0,ranks_LR,rng)
g_lst,h_lst = butterfly_completer3(T_sparse_LR,inds_LR,T_sparse_test_LR, inds_test_LR, L_LR, g_lst, h_lst, num_iters=num_iters, tol=tol)
e = time.time()
left_mat = g_lst[0]
right_mat = h_lst[0]
print('--time in low-rank completion:',e-s)


rng = np.random.RandomState(np.random.randint(1000))
g_lst,h_lst = gen_tensor_inputs(m,n,L,lc,ranks,rng)
# Unoptimized way of doing butterfly decomposition
# s = time.time()
# left,g_lst,h_lst,right = butterfly_completer(T_sparse,T_mat,Omega,L,left,g_lst,h_lst,right,num_iters,tol=tol)
# e = time.time()
# print('--time in low-rank to butterfly conversion:',e-s)
s=time.time()
g_lst,h_lst = butterfly_decompose_low_rank(left_mat,right_mat,L,ranks,g_lst,h_lst)
e = time.time()
print('--time in low-rank to butterfly conversion with new:',e-s)


print('starting butterfly completion now')
num_iters = 10
s = time.time()
g_lst,h_lst = butterfly_completer3(T_sparse,inds,T_sparse_test, inds_test, L, g_lst, h_lst, num_iters=num_iters, tol=tol)
e = time.time()
print('--time in butterfly completion:',e-s)



