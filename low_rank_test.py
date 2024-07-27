import numpy as np 
import numpy.linalg as la
from kdtree_ordering import generate_kd_tree

def create_omega(shape,nnz,seed=123):
	omega = np.zeros(np.prod(shape))
	omega[:int(nnz)] = 1
	np.random.shuffle(omega)
	omega = omega.reshape(shape)
	return omega

def create_T(m,n,rank):
	A = np.random.uniform(-1,1,size=(m,r))
	B = np.random.uniform(-1,1,size=(n,r))

	return A@B.T


def get_greens_kernel1():
    # Define the number of points and the wavenumber
    #wavelen = 0.35/(2 ** 2.5)
    wavelen = 10/48
    ppw = 10
    ds = wavelen/ppw
    Nperdim = int(np.ceil(1.0/ds))
    
    Nperdimx = Nperdim
    Nperdimy = Nperdim

    # Initialize the Green's function matrix
    G = np.zeros((Nperdimx*Nperdimy, Nperdimx*Nperdimy))

    pts = [(x,y) for x in np.linspace(0,1,Nperdimx) for y in np.linspace(0,1,Nperdimy)]
    
    waven = 2*np.pi/wavelen
    # Compute the Green's function matrix
    
    pts_arr = np.array(pts)
    # # usual ordering
    # for i in range(len(pts_arr)):
    #     for j in range(len(pts_arr)):
    #         dist = np.sqrt( np.sum((pts_arr[i] - pts_arr[j])**2 +1))
    #         G[i,j] = np.cos(-1 * waven* dist) / dist
    
    

    order_pts = generate_kd_tree(pts)
    
    i = 0
    for p in order_pts:
        j=0
        for q in order_pts:
            dist = np.sqrt( np.sum((p - q)**2 +1))
            G[i,j] = np.cos(-1 * waven* dist) / dist
            j+=1
        i+=1
            

    return G



# #np.save('green_func',mat)

# U,s,Vt = la.svd(mat)
# print(np.sum(s>s[0]*1e-4))
# m = 130*2**6
# n = m
# rank = 2
# num_iter = 20
# c= 3
# tries = 20
# nnz = n*np.log(n)*c
# lmb = 1e-06
# r= rank

# T =create_T(m,n,rank)
# omega = create_omega((m,n),nnz = nnz)
# T_sparse = np.einsum('ij,ij->ij',T,omega,optimize=True)
# conv = 0




def matrix_completion(T,T_sparse,omega,r,num_iter):
	guess1 = np.random.uniform(-1,1,size=(T.shape[0],r))
	guess2 = np.random.uniform(-1,1,size=(T.shape[1],r))
	lmb = 1e-05

	error = la.norm(T - guess1@guess2.T)/la.norm(T)
	print('error before is',error)
	weight = 1

	for iters in range(num_iter):

		E = T_sparse - np.einsum('ij,ia,ja->ij',omega,guess1,guess2,optimize=True)

		LHS = np.einsum('ij,jr,jz->irz',omega,guess2,guess2,optimize=True)
		RHS = np.einsum('ij,jr->ir',E,guess2,optimize=True)

		update = np.zeros_like(guess1)

		for rows in range(LHS.shape[0]):
			if not np.allclose(RHS[rows,:],np.zeros_like(RHS[rows,:])): 
				update[rows,:] = la.solve(LHS[rows] + lmb*np.eye(RHS[rows,:].shape[0]),RHS[rows,:])

		guess1 += weight*update


		E = T_sparse - np.einsum('ij,ia,ja->ij',omega,guess1,guess2,optimize=True)

		LHS = np.einsum('ij,ir,iz->jrz',omega,guess1,guess1,optimize=True)
		RHS = np.einsum('ij,ir->jr',E,guess1,optimize=True)

		update = np.zeros_like(guess2)
		for rows in range(LHS.shape[0]):
			if not np.allclose(RHS[rows,:],np.zeros_like(RHS[rows,:])): 
				update[rows,:] = la.solve(LHS[rows]+lmb*np.eye(RHS[rows,:].shape[0]),RHS[rows,:])

		guess2 += weight*update

		error = la.norm(T - guess1@guess2.T)/la.norm(T)
		print('error after',iters+1,'is',error)
		if error< 1e-06:
			print('converged')
			conv = 1
			break
	return guess1,guess2

def subspace_iteration(T,T_sparse,omega,r,num_iter,left,right):
	# left,_ = la.qr(np.random.uniform(-1,1,size=(T.shape[0],r)))
	# right,_ = la.qr(np.random.uniform(-1,1,size=(T.shape[1],r)))
	# mid = np.random.randn(r,r)
	left,mid1 = la.qr(left)
	right,mid2 = la.qr(right)
	mid = mid1@mid2.T

	error = la.norm(T - np.einsum('ia,ab,jb->ij',left,mid,right,optimize=True))/la.norm(T)
	print('error before is',error)

	t= 1
	lmb = 1e-6

	for iters in range(num_iter):
		E = T_sparse - np.einsum('ij,ia,ab,jb->ij',omega,left,mid,right,optimize=True)

		left,_ = la.qr( left+ t*np.einsum('ij,jb,ba->ia',E,right,la.pinv(mid),optimize=True) )

		right,_ = la.qr(right + t*np.einsum('ij,ia,ba->jb',E,left,la.pinv(mid),optimize=True))

		lhs_mid = np.einsum('ij,ia,il,jb,jm->albm',omega,left,left,right,right,optimize=True)
		rhs_mid = np.einsum('ij,ia,jb->ab',T_sparse,left,right,optimize=True)

		print('check norm of rhs')
		print(la.norm(rhs_mid))

		lhs_mid = lhs_mid.reshape((r*r,r*r))
		mid = la.solve(lhs_mid + lmb*np.eye(r*r),rhs_mid.reshape(-1)).reshape((r,r))

		error = la.norm(T - np.einsum('ia,ab,jb->ij',left,mid,right,optimize=True))/la.norm(T)
		print('error after',iters,'is',error)

		if error< 1e-06:
			print('converged')
			conv = 1
			break
	return left@mid, right

#def matrix_Kronecker_completion(T,T_sparse,omega,

#lhs,rhs = matrix_completion(T,T_sparse,omega,r)
