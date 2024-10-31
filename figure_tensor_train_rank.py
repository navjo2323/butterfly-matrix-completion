import numpy as np
from tensor_train_decomp import *
from kdtree_ordering import generate_kd_tree
import time
import math
import numpy.linalg as la
import tensorly as tl
from tensorly import tucker_to_tensor
from tensorly.decomposition import matrix_product_state, tucker
from tensorly.contrib.decomposition import tensor_train_cross
import time

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

kernel= 1 # 1: Green's function 2: 2D Radon transform 3: 1D Radon transform
real = 1 # 1: real-valued kernels, 0: complex-valued kernels
get_true_rank=1
lowrank_only= 1
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


T = reshape_matrix_to_tensor_QTT(mat, L, c)

# has_nan = np.isnan(T).any()
# print(f"Contains NaN: {has_nan}")

# # Check for Inf
# has_inf = np.isinf(T).any()
# print(f"Contains Inf: {has_inf}")
# r_TT = 8
# small = 6

# ranks_TT = [1] + [small] +  [r_TT for _ in range(L-1)] + [1]


start = 7
intermediate = [start + 2 * i for i in range(L)]

intermediate = []

ranks_TT = [1] + intermediate + [1]


l_shape = int(c**2)
r_shape = 4**L
max_r = l_shape
for i in range(1,len(ranks_TT)-1):
    ranks_TT[i] = min(max_r, ranks_TT[i])
    l_shape = l_shape*4
    r_shape = r_shape/4
    max_r = int(min(l_shape,r_shape))

# ranks_Tucker = [3 for _ in range(L+1)]
# ranks_Tucker[0] = 6
print(ranks_TT)

# print(ranks_Tucker)


# core, factors = tucker(tl.tensor(T), rank=ranks_Tucker)


# Perform TT-cross decomposition with automatic rank selection via tolerance
factors = matrix_product_state(T, rank=ranks_TT )

reconstructed_tensor = tl.tt_to_tensor(factors)

# reconstructed_tensor = tucker_to_tensor((core, factors))

# Calculate the reconstruction error
error = tl.norm(tl.tensor(T) - reconstructed_tensor) / tl.norm(tl.tensor(T))
print("Reconstruction Error:", error)
# if error < tol2:
#     print('ranks are',ranks_TT)


