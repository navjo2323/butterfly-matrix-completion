import numpy as np
from butterfly_tensor_train import *
from butterfly_helper_functions import index_convert, gen_tensor_inputs, ALS_solve
from butterfly_decomposition import butterfly_decompose_low_rank
from kdtree_ordering import generate_kd_tree
import time
import math
import numpy.linalg as la





def is_perfect_square(n):
    return n == int(math.isqrt(n))**2

def get_greens_kernel(c, L, ppw, inds=None):
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
        G = np.cos(-1 * waven * dist) / dist
        
        return G
    else:
        # Vectorized computation for T_sparse
        inds = np.array(inds)
        p = pts[inds[:, 0]]  # Extract points for ind_i
        q = pts[inds[:, 1]]  # Extract points for ind_j
        
        # Calculate distances using vectorized operations
        dist = np.sqrt(np.sum((p - q)**2, axis=1) + 1)
        
        # Compute T_sparse using vectorized operations
        T_sparse = np.cos(-1 * waven * dist) / dist
        
        return T_sparse



rng = np.random.RandomState(np.random.randint(1000))

get_true_rank=0
c = 4
#Should be perfect square, 4 and 9 options

L = 12

#Should be even, becomes too slow after 10 for this version of code

tol=1e-3
ppw=10

lc = int(L/2) 
I = c*2**L
J = c*2**L


r_BF= 11
ranks = [r_BF for _ in range(L- L//2+1 )] 

for i in range(len(ranks)):
    if i==0:
        ranks[0] = min(ranks[0],c)
    else:
        ranks[i] = min(2*ranks[i-1],ranks[i])
print('ranks for butterfly completion are ', ranks)


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




nnz = int(3*(r_BF)*I*np.log2(I))

print('matrix shape is',I)
print('nnz is',nnz)
print('ratio of nonzeros is',nnz/(I*J))
s = time.time()
indices = create_inds(I, J, nnz,rng)
indices_test = create_inds(I, J, nnz,rng)
e = time.time()
print('--time in creating indices:',e-s)




s = time.time()
T_sparse = get_greens_kernel(c,L,ppw=ppw,inds=indices)
T_sparse_test = get_greens_kernel(c,L,ppw=ppw,inds=indices_test)
e = time.time()
print('--time in entry generation:',e-s)




##### TEST CODE FOR GREENS ########

ranks_lr = [r_BF*10]
L_lr = 0
c_lr = I
num_iter_lr = 10

s = time.time()
inds_tt_lr = encode_tuples(indices, L_lr, c_lr)
inds_tt_test_lr = encode_tuples(indices_test, L_lr, c_lr)
e = time.time()
print('--time in index conversion:',e-s)

s= time.time()
tensor_lst_lr = gen_tensor_train_list(L_lr, c_lr, ranks_lr ,rng )
e = time.time()
print('--time to generate inputs for matrix completion',e-s)


s = time.time()
tensor_lst_lr = butterfly_tensor_train_completer(T_sparse, inds_tt_lr, T_sparse_test, inds_tt_test_lr, L_lr, tensor_lst_lr, num_iter_lr, tol)
left_mat = tensor_lst_lr[0]
right_mat = tensor_lst_lr[1]
e = time.time()
print('--time for matrix completion',e-s)

s = time.time()
g_lst,h_lst = gen_tensor_inputs(I, J, L, L//2, ranks, rng)
g_lst, h_lst = butterfly_decompose_low_rank(left_mat,right_mat,L,ranks,g_lst,h_lst)
e= time.time()
print('--time for butterfly decomposition', e-s)



num_iters = 10

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
tensor_lst = butterfly_tensor_train_completer(T_sparse, inds_tt, T_sparse_test, inds_tt_test, L, tensor_lst, num_iters, tol)
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



