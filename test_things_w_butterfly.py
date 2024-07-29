import numpy as np 
import numpy.linalg as la
from tensor_formulated_butterfly import * 
from low_rank_test import *
from kdtree_ordering import generate_kd_tree
import math
import time


def is_perfect_square(n):
    return n == int(math.isqrt(n))**2
# def get_kd_tree_ordering():
    

def get_greens_kernel(c,L,ppw):
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

    # Initialize the Green's function matrix
    G = np.zeros((Nperdimx*Nperdimy, Nperdimx*Nperdimy))

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
    
    i = 0
    for p in order_pts:
        j=0
        for q in order_pts:
            dist = np.sqrt(la.norm(p-q)**2 + 1)   #np.sqrt( np.sum((p - q)**2) +1)
            G[i,j] = np.cos(-1 * waven* dist) / dist
            j+=1
        i+=1
            

    return G




rng = np.random.RandomState(np.random.randint(1000))

c = 4
#Should be perfect square, 4 and 9 options

L = 8

#Should be even, becomes too slow after 10 for this version of code



lc = int(L/2) 
s = time.time()
mat= get_greens_kernel(c,L,ppw=10)
e = time.time()
#np.save('greens_matN-48ppw15.npy',mat)

#mat = np.load('greens_matN-48ppw15.npy')



print('greens mat generated of shape',mat.shape)
print('--time in greens mat generation:',e-s)

m = mat.shape[0]
n= m

ranks = [8 for _ in range(L-lc+1 )] # Increasing this  makes things very slow for current implementation, R^2 x R^2 solves are not parallel

print('ranks for butterfly completion are ', ranks)

# Can give assymetric ranks, if needed

r = 25
nnz = int((r)*n*np.log2(n))

print('m*n is',m*n)
print('nnz is',nnz)
print('ratio of nonzeros is',nnz/(m*n))
s = time.time()
T = get_butterfly_tens_from_mat(mat,L,lc,c)


omega = create_omega(mat.shape,nnz)

mat_sparse = mat*omega

print('Low-rank completion rank',r)

left,right = matrix_completion(mat,mat_sparse,omega, r=r,num_iter = 10)

mat = left@right.T
e = time.time()
print('--time in low-rank completion:',e-s)

T_mat = get_butterfly_tens_from_mat(left@right.T,L,lc,c)   # Get tensor from low rank matrix





# # recon = recon_butterfly_tensor(left,g_lst,h_lst,right,L,lc)
# # error = la.norm(T - recon)/la.norm(T)
# # sparse_error = la.norm(T_sparse - Omega*recon)
# # errors =[]
# # errors.append(error)
# # print('sparse error is',sparse_error)
# print('relative error after',0,'is',error)
errors = []

nnz_n = np.prod(T_mat.shape)       #All entries
Omega = create_omega(T_mat.shape,nnz_n)    # ALl ones

print('sum should be m*n',np.sum(Omega))

rng = np.random.RandomState(np.random.randint(1000))

left,g_lst,h_lst,right = gen_tensor_inputs(m,n,L,lc,ranks,rng)

num_iters = 1
T_sparse = Omega*T_mat


# Unoptimized way of doing butterfly decomposition
s = time.time()
left,g_lst,h_lst,right = butterfly_completer(T_sparse,T_mat,Omega,L,left,g_lst,h_lst,right,num_iters,tol=1e-3)
e = time.time()
print('--time in low-rank to butterfly conversion:',e-s)

print('starting completion now')

Omega = get_butterfly_tens_from_mat(omega,L,lc,c)

T_sparse = T*Omega

num_iters = 20
s = time.time()
left,g_lst,h_lst,right = butterfly_completer(T_sparse,T,Omega,L,left,g_lst,h_lst,right,num_iters,tol=1e-3)
e = time.time()
print('--time in butterfly completion:',e-s)


