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

c = 9
L = 8
lc = int(L/2) 

rng = np.random.RandomState(np.random.randint(1000))

m = c*2**L
n = m
ranks = [3 for _ in range(L-lc+1)]

T, lst2 = const_butterfly_tensor(m,n,L,lc,ranks,rng)

def get_greens_kernel(wavelen=5):
	# Define the number of points and the wavenumber
	wavelen = 3/32
	ppw = 3
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


#mat = get_butterfly_mat_from_tens(T,L,lc,c)
mat = get_greens_kernel()
print('generated butterfly mat')

U,s,Vt = la.svd(mat)

print('len of s',len(s))
print(sum(s<1e-9))