import numpy as np
import numpy.linalg as la
import sys
import time

def const_butterfly_mat(shape,rank,layer=1):
	m,n = shape
	mat = np.zeros(shape)
	As = [np.random.uniform(low = -1, high =1, size=(int(m/2),rank)) for i in range(2)]
	Bs = [np.random.uniform(low = -1, high =1, size=(int(n/2),rank)) for i in range(2)]
	D = np.random.uniform(low=-1,high =1,size=(2*rank,2*rank))
	for i in range(2):
		for j in range(2):
			mat[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)] = np.einsum('ir,rz,jz->ij',As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Bs[j],optimize=True)
	return mat,[As,D,Bs]


def create_low_rank(shape,rank):
	m,n = shape
	A = np.random.uniform(low=-1,high = 1, size=(m,rank))
	B =  np.random.uniform(low=-1,high = 1, size=(n,rank))
	T =  np.einsum('ir,jr->ij',A,B,optimize=True)
	return T

def create_omega(shape,sp_frac,seed=123):
	np.random.seed(seed)
	omega = np.zeros(np.prod(shape))
	omega[:int(sp_frac*np.prod(shape))] = 1
	np.random.shuffle(omega)
	omega = omega.reshape(shape)
	return omega

m = 100
n = 80

rank = 5

T,originals = const_butterfly_mat((m,n), rank = rank)

As = [np.random.uniform(low=-1,high = 1, size=(int(m/2),rank)) for i in range(2)]
D = np.random.uniform(low=-1,high = 1, size=(2*rank,2*rank))
Bs = [np.random.uniform(low=-1,high = 1, size=(int (n/2),rank)) for i in range(2)]


Omega = create_omega(T.shape,sp_frac=0.4)

T_sparse = np.einsum('ij,ij->ij',T,Omega,optimize=True)
num_iter= 100
regu = 0

recon = np.zeros_like(T)
for i in range(2):
	for j in range(2):
		recon[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)] = np.einsum('ir,rz,jz->ij',As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Bs[j],optimize=True)
error = la.norm(T - recon)
print('Initial relative error is ',error/la.norm(T))





for iters in range(num_iter):
	#Solve for A
	for i in range(2):
		LHS = np.zeros((int(T.shape[0]/2),rank,rank))
		RHS = np.zeros((int(T.shape[0]/2),rank))
		for j in range(2):
			LHS += np.einsum('jr,zr,ij,jp,lp->izl',Bs[j],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
				Bs[j],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
			RHS += np.einsum('ij,jr,lr->il',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],Bs[j],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
		
		for p in range(LHS.shape[0]):
			As[i][p,:] =la.solve(LHS[p] + regu*np.eye(rank),RHS[p,:]) #SPD Solve


	# Solve for B
	for j in range(2):
		LHS = np.zeros((int(T.shape[1]/2),rank,rank))
		RHS = np.zeros((int(T.shape[1]/2),rank))
		for i in range(2):
			LHS += np.einsum('ir,rz,ij,ip,pl->jzl',As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
				As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
			RHS += np.einsum('ij,ir,rl->jl',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
		
		for p in range(LHS.shape[0]):
			Bs[j][p,:] =la.solve(LHS[p] + regu*np.eye(rank),RHS[p,:]) #SPD Solve
	
	# Solve for D
	for i in range(2):
		for j in range(2):
			LHS = np.einsum('iz,jr,ij,il,jm->zrlm',As[i],Bs[j],Omega[i*int(m/2):(i+1)*int(m/2),
				j*int(n/2):(j+1)*int(n/2)],As[i],Bs[j],optimize=True).reshape((rank*rank,rank*rank))
			RHS = np.einsum('ij,iz,jr->zr',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
				As[i],Bs[j],optimize=True).reshape(-1)
			D[i*rank:(i+1)*rank,j*rank:(j+1)*rank] = la.solve(LHS + regu*np.eye(rank*rank),RHS).reshape((rank,rank)) #SPD Solve

	
	recon = np.zeros_like(T)
	for i in range(2):
		for j in range(2):
			recon[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)] = np.einsum('ir,rz,jz->ij',As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Bs[j],optimize=True)
	error = la.norm(T - recon)
	print('Relative error is ',error/la.norm(T))

	rel_err = error/la.norm(T)
	if rel_err < 1e-14:
		print('Converged in',iters)
		break

