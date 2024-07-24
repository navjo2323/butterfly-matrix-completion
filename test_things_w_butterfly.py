import numpy as np 
import numpy.linalg as la
from tensor_formulated_butterfly import * 
from low_rank_test import *

def get_index(i,j,L,c):
	ind_i = i
	ind_j = j
	left = []
	right = []
	num1 = 2**L
	num2 = 2**L
	for m in range(L):
		val1 = int(ind_i >= num1//2)
		val2 = int(ind_j >= num2//2)
		left.append(val1)
		right.append(val2)
		num1 = num1//2
		num2 = num2//2
		if val1:
		    ind_i -= num1
		if val2:
		    ind_j -= num2
	return left,right

def get_butterfly_mat_from_tens(T,L,lc,c):
	# T is constructed from the lst
	big_side = c*2**L
	mat = np.zeros((big_side,big_side))
	for i in range(2**L):
		for j in range(2**L):
			left, right = get_index(i,j,L,c)
			mat[c*i:c*(i+1),c*j:c*(j+1) ] = T[tuple(left +[slice(None)] + right + [slice(None)])]
	return mat


def get_butterfly_tens_from_mat(mat,L,lc,c):
	block_m = int(m/2**L)
	block_n = int(n/2**L)
	shape = [2 for l in range(L)]
	shape.append(block_m)
	shape += [2 for l in range(L)]
	shape.append(block_n)
	T = np.zeros(shape)
	for i in range(2**L):
		for j in range(2**L):
			left,right = get_index(i,j,L,c)
			T[tuple(left +[slice(None)] + right + [slice(None)])] = mat[c*i:c*(i+1),c*j:c*(j+1) ]
	return T

def get_greens_kernel():
	# Define the number of points and the wavenumber
	#wavelen = 0.35/(2 ** 2.5)
	wavelen = 15/48
	ppw = 15
	ds = wavelen/ppw
	Nperdim = int(np.ceil(1.0/ds))

	print(Nperdim)
	# Initialize the Green's function matrix
	G = np.zeros((Nperdim**2, Nperdim**2))

	pts = np.linspace(0,1,Nperdim)


	waven = 2*np.pi/wavelen
	# Compute the Green's function matrix
	for i in range(Nperdim):
		for j in range(Nperdim):
			for k in range(Nperdim):
				for l in range(Nperdim):
					m = i*Nperdim + j
					n = k*Nperdim + l
					dist = np.sqrt( (pts[i] - pts[k])**2 + (pts[j] - pts[l])**2 + 1 )
					G[m, n] = np.cos(-1 * waven* dist) / dist

	return G



rng = np.random.RandomState(np.random.randint(1000))

#mat= get_greens_kernel()

#np.save('greens_matN-48ppw15.npy',mat)

mat = np.load('greens_matN-48ppw15.npy')

print('greens mat generated of shape',mat.shape)

m = mat.shape[0]

c = 9
L = 8
lc = int(L/2) 

m = c*2**L
n = m
ranks = [7 for _ in range(L-lc+1)]

r = 20
nnz = int((r)*n*np.log2(n))
#nnz = m*n

print('m*n is',m*n)
print('nnz is',nnz)
print('ratio is',nnz/(m*n))

T = get_butterfly_tens_from_mat(mat,L,lc,c)

# lst = gen_tensor_inputs(m,n,L,lc,ranks,rng)

# T, lst2 = const_butterfly_tensor(m,n,L,lc,ranks,rng)
# mat = get_butterfly_mat_from_tens(T,L,lc,c)


#rank = c
# A = np.random.randn(m,rank)
# B = np.random.randn(n,rank)

# mat = A@B.T

omega = create_omega(mat.shape,nnz)

mat_sparse = mat*omega


left,right = matrix_completion(mat,mat_sparse,omega, r=r,num_iter = 10)
#left,right = subspace_iteration(mat,mat_sparse,omega, r=120,num_iter = 10,left=left,right=right)

mat = left@right.T


T_mat = get_butterfly_tens_from_mat(left@right.T,L,lc,c)





# # recon = recon_butterfly_tensor(left,g_lst,h_lst,right,L,lc)
# # error = la.norm(T - recon)/la.norm(T)
# # sparse_error = la.norm(T_sparse - Omega*recon)
# # errors =[]
# # errors.append(error)
# # print('sparse error is',sparse_error)
# print('relative error after',0,'is',error)
errors = []

nnz_n = np.prod(T_mat.shape)
Omega = create_omega(T_mat.shape,nnz_n)

print('sum should be m*n',np.sum(Omega))

rng = np.random.RandomState(np.random.randint(1000))

left,g_lst,h_lst,right = gen_tensor_inputs(m,n,L,lc,ranks,rng)

num_iters = 3
T_sparse = Omega*T_mat

for iters in range(num_iters):
	left,trig = solve_for_outer(0,L,T_sparse,Omega,left,g_lst,h_lst,right)
	if trig:
		print('trig')

	right,trig = solve_for_outer(1,L,T_sparse,Omega,left,g_lst,h_lst,right)
	if trig:
		print('trig')

	for l in range(L-1,int(L/2)-1,-1):
		g_lst =  solve_for_inner(0,L,l,T_sparse,Omega,left,g_lst,h_lst,right)

	for l in range(int(L/2),L,1):
		h_lst =  solve_for_inner(1,L,l,T_sparse,Omega,left,g_lst,h_lst,right)

	recon = recon_butterfly_tensor(left,g_lst,h_lst,right,L,lc)
	error = la.norm(T - recon)/la.norm(T)
	errors.append(error)
	print('relative error after',iters +1,'is',error)
	if iters+1 >=10 and error >=4:
		break
	if error < 1e-05:
		print('converged')
		conv+=1
		break





Omega = get_butterfly_tens_from_mat(omega,L,lc,c)

T_sparse = T*Omega



num_iters = 20

for iters in range(num_iters):
	left,trig = solve_for_outer(0,L,T_sparse,Omega,left,g_lst,h_lst,right)
	if trig:
		print('trig')

	right,trig = solve_for_outer(1,L,T_sparse,Omega,left,g_lst,h_lst,right)
	if trig:
		print('trig')

	for l in range(L-1,int(L/2)-1,-1):
		g_lst =  solve_for_inner(0,L,l,T_sparse,Omega,left,g_lst,h_lst,right)

	for l in range(int(L/2),L,1):
		h_lst =  solve_for_inner(1,L,l,T_sparse,Omega,left,g_lst,h_lst,right)

	recon = recon_butterfly_tensor(left,g_lst,h_lst,right,L,lc)
	error = la.norm(T - recon)/la.norm(T)
	errors.append(error)
	sparse_error = la.norm(T_sparse - Omega*recon)
	print('sparse error is',sparse_error)
	print('relative error after',iters +1,'is',error)
	if iters+1 >=10 and error >=4:
		break
	if error < 1e-05:
		print('converged')
		break




