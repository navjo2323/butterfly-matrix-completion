import numpy as np
import numpy.linalg as la
import sys
import time


def compute_matrix_with_butterfly(As,D,Bs):
	blocks = len(As)
	rank = As[0].shape[1]
	m = blocks*As[0].shape[0]
	n = blocks*Bs[0].shape[0]
	mat = np.zeros((m,n))
	for i in range(blocks):
		for j in range(blocks):
			mat[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)] = np.einsum('ir,rz,jz->ij',As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Bs[j],optimize=True)
	return mat


def gen_all_matrices(shape,ranks,L,lc):
	m,n = shape
	assert len(ranks) == (L-lc+1), 'length of ranks should be the same as number of layers'
	lst_A = []
	lst_B = []
	p = 0
	l1 = m
	l2 = n
	l3 = ranks[0]
	blocks = 2**L

	As = [np.random.uniform(low = -1, high =1, size=(int(m/blocks),ranks[0])) for j in range(blocks)]
	Bs = [np.random.uniform(low = -1, high =1, size=(int(n/blocks),ranks[0])) for j in range(blocks)]
	lst_A.append(As)
	lst_B.append(Bs)
	for i in range(L-1,lc-1,-1):
		blocks = 2**i
		As = [[np.random.uniform(low = -1, high =1, size=(ranks[p],ranks[p+1]))  for j in range(blocks)] for k in range(4**(p+1))]
		Bs = [[ np.random.uniform(low = -1, high =1, size=(ranks[p],ranks[p+1])) for j in range(blocks)] for k in range(4**(p+1))]
		lst_A.append(As)
		lst_B.append(Bs)
		if i == lc:
			lst_D = [ np.random.uniform(low=-1,high =1,size=(ranks[-1],ranks[-1])) for i in range(4**(p+1))]
		p+=1
	return lst_A,lst_D,lst_B

def get_mats(inds_A,ind_D,inds_B,lst_A,lst_D,lst_B):
	mats_A = [lst_A[0][inds_A[0]]]
	mats_B = [lst_B[0][inds_B[0]]]
	for i in range(1,len(inds_A)):
		mats_A.append(lst_A[i][inds_A[i][0]][inds_A[i][1]])
		mats_B.append(lst_B[i][inds_A[i][0]][inds_A[i][1]])

	
	return mats_A, lst_D[ind_D], mats_B


def get_indices(i,j,L,lc):
	# For now assume, lc = L/2
	inds_A = [i]
	inds_B = [j]
	prev = [0]
	for l in range(L-1,lc-1,-1):
		# Figure out the block update the index
		# Multiply all previous by 4 at each level and then add it to the current index
		prev = [4*i for i in prev]
		if i < 2**l and j < 2**l:
			# Top left
			index = 0 + np.sum(prev)
			inds_A.append([index,i])
			inds_B.append([index,j])
			prev.append(0)
		elif i < 2**l and j >= 2**l:
			# top right
			index = 1 + np.sum(prev)
			inds_A.append([index,i])
			inds_B.append([index,j%2**l])
			j= j%2**l
			prev.append(1)
		elif i >= 2**l and j < 2**l:
			# bottom left
			index = 2 + np.sum(prev)
			inds_A.append([index,i%2**l])
			inds_B.append([index,j])
			i = i%2**l
			prev.append(2)
		else:
			#Bottom right
			index = 3 + np.sum(prev)
			inds_A.append([index,i%2**l])
			inds_B.append([index,j%2**l])
			i = i%2**l
			j = j%2**l
			prev.append(3)

	return inds_A,index,inds_B




def gen_einstr(length):
	A_str = ""
	B_str = ""
	einstr = ""
	for i in range(length):
		if i != length-1:
			A_str += chr(ord('a')+i) + chr(ord('a')+i+1) + ','
			B_str += chr(ord('a')+length+i) + chr(ord('a')+length+i+1) + ','
		else:
			A_str += chr(ord('a')+i) + 'y,'
			B_str += chr(ord('a')+length+i) + 'z'
	einstr += A_str + "yz,"+ B_str +"->" + "a"+chr(ord('a')+length)

	return einstr


def gen_solve_einstr(which,solve_layer,L,lc):
	LHS = ""
	RHS = ""
	A_str = ""
	B_str = ""
	einstr = ""
	length = L-lc+1
	s_l = L-solve_layer 
	for i in range(length):
		if i != length-1:
			if i != s_l:
				A_str += chr(ord('a')+i) + chr(ord('a')+i+1) + ','
				B_str += chr(ord('a')+length+i) + chr(ord('a')+length+i+1) + ','
			else:
				# Skip the string which is to be solved
				if which==0:
					B_str += chr(ord('a')+length+i) + chr(ord('a')+length+i+1) + ','
				elif which ==1:
					A_str += chr(ord('a')+i) + chr(ord('a')+i+1) + ','
		else:
			if i != s_l:
				A_str += chr(ord('a')+i) + 'y,'
				B_str += chr(ord('a')+length+i) + 'z'
			else:
				if which==0:
					B_str += chr(ord('a')+length+i) + 'z'
				else:
					B_str = B_str[:-1]
					A_str += chr(ord('a')+i) + 'y,'

	einstr += "a"+chr(ord('a')+length) +','+ A_str + "yz,"+ B_str 
	if which ==0:
		if s_l != length -1:
			RHS = einstr +"->"+ chr(ord('a')+s_l) + chr(ord('a')+s_l+1)
		else:
			RHS = einstr +"->"+ chr(ord('a')+s_l) + 'y'
	else:
		if s_l != length-1:
			RHS = einstr +"->"+ chr(ord('a')+s_l+length) + chr(ord('a')+s_l+length+1)
		else:
			RHS = einstr +"->"+ chr(ord('a')+s_l+length) + 'z'

	A_str = ""
	B_str = ""
	for i in range(length):
		if i != length-1:
			if i != s_l:
				if i==0:
					A_str += chr(ord('a')+i) + chr(ord('a')+2*length+i) + ','
					B_str += chr(ord('a')+length+i) + chr(ord('a')+3*length) + ','
				else:
					A_str += chr(ord('a')+2*length+i-1) + chr(ord('a')+2*length+i) + ','
					B_str += chr(ord('a')+3*length+i -1) + chr(ord('a')+3*length+i) + ','
			else:
				# Skip the string which is to be solved
				if which==0:
					B_str += chr(ord('a')+3*length+i -1) + chr(ord('a')+3*length+i) + ','
				elif which ==1:
					A_str += chr(ord('a')+2*length+i -1) + chr(ord('a')+2*length+i) + ','
		else:
			if i != s_l:
				A_str += chr(ord('a')+2*length+i-1) + 'w,'
				B_str += chr(ord('a')+3*length+i-1) + 'x'
			else:
				if which ==0:
					B_str += chr(ord('a')+3*length+i -1) + 'x'
				else:
					B_str = B_str[:-1]
					A_str += chr(ord('a')+ 2*length+i-1) + 'w,'

	LHS = einstr + ',' + A_str + "wx,"+ B_str + '->'

	if which == 0:
		if s_l != length - 1 and s_l != 0:
			LHS += chr(ord('a')+s_l) + chr(ord('a')+s_l+1) + chr(ord('a')+2*length+ s_l -1) + chr(ord('a')+ 2*length+ s_l)
		elif s_l ==length -1:
			LHS += chr(ord('a')+s_l) + 'y' + chr(ord('a')+2*length+ s_l -1) + 'w'
		else:
			LHS += chr(ord('a')+0) + chr(ord('a')+s_l+1) +  chr(ord('a')+ 2*length+ s_l)
	else: 
		if s_l != length - 1 and s_l != 0: 
			LHS += chr(ord('a')+length+ s_l) + chr(ord('a')+ length+ s_l+1) + chr(ord('a')+3*length+ s_l-1) + chr(ord('a')+ 3*length+ s_l)
		elif s_l ==length -1:
			LHS += chr(ord('a')+length+s_l) + 'z' + chr(ord('a')+3*length+ s_l -1) + 'x'
		else:
			LHS += chr(ord('a')+length) + chr(ord('a')+length+s_l+1) +  chr(ord('a')+ 3*length+ s_l)
	print(RHS)
	print(LHS)
	return LHS,RHS

def figure_indices(solve_layer,solve_inds,which,L,lc):
	'''
	which is 0 for A
	1 for B
	'''
	inds_A = []
	inds_B = []
	inds_D = []
	if which ==0:
		# solving for A, which means all of the B in that level will be used in solve
		inds_A.append(solve_inds)
		if solve_layer != L:
			inds_B.append([solve_inds[0],[i for i in range(2**solve_layer)]])
		else:
			inds_B.append([i for i in range(2**solve_layer)])

	elif which ==1:
		# solving for B, which means all of the A in that level wil be used in solve
		if solve_layer != L:
			inds_A.append([solve_inds[0],[i for i in range(2**solve_layer)]])
		else:
			inds_A.append([i for i in range(2**solve_layer)])
		inds_B.append(solve_inds)

	for l in range(solve_layer+1,L+1,1):

		# shift should be calculated based on solving for A or B
		if which == 0:
			# This should give me the group number for both A and B
			if solve_layer != L:
				group = inds_A[0][0]//4 # Which quadrant?
				# Below now gives if first half or second half
				num = inds_A[0][0]%4 # 0,1,2,3 (Z ordering)--> 0,1 means top blocks, 2,3 means bottom blocks
			else:
				inds_A[0]%4
			shift = int((num/2)*2**(l-1))  # 2**blocks shift for bottom blocks
			if l != L:
				inds_A.insert(0,[group,inds_A[0][1]+shift])  # insert at the first position always
				inds_B.insert(0,[group,[ids+shift for ids in inds_B[0][1]]])
			else:
				inds_A.insert(0,inds_A[0][1]+shift)  # insert at the first position always
				inds_B.insert(0,[ids+shift for ids in inds_B[0][1]])
		else:
			# This should give me the group number for both A and B
			if solve_layer != L :
				group = inds_B[0][0]//4 # Which quadrant?
				# Below now gives if first half or second half
				num = inds_B[0][0]%4 # 0,1,2,3 (Z ordering)--> 0,1 means top blocks, 2,3 means bottom blocks
			else:
				num = inds_B[0]%4

			shift = int((num%2)*2**(l-1))  # 2**blocks shift for blocks 1 and 3
			if l != L:  
				inds_B.insert(0,[group,inds_B[0][1]+shift])	 # insert at the first position always
				inds_A.insert(0,[group,[ids+shift for ids in inds_A[0][1]]])
			else:
				inds_B.insert(0,inds_B[0][1]+shift)	 # insert at the first position always
				inds_A.insert(0,[ids+shift for ids in inds_A[0][1]])


	for l in range(solve_layer-1,lc-1,-1):

		#isinstance(variable, list)
		if isinstance(inds_A[-1],list) and isinstance(inds_A[-1][0],list) and which ==0:
			groups = []
			for gr in inds_A[-1][0]:
				group_start = gr*4  # group and index to decide the groups and indices of As. B will be all of it
				if inds_A[-1][1] < 2**l:
					grps = [group_start, group_start + 1] # Top blocks
				else:
					grps = [group_start + 2, group_start + 3]# Bottom blocks
				groups.extend(grps)
			index = inds_A[-1][1]%2**l
			inds_A.append([groups,index])
			inds_B.append([groups,[ids for ids in range(2**l)]])
		elif isinstance(inds_B[-1],list) and isinstance(inds_B[-1][0],list) and which ==1:
			groups = []
			for gr in inds_A[-1][0]:
				group_start = gr*4
				if inds_B[-1][1] < 2**l:
					grps = [group_start+2, group_start + 3] # Top blocks
				else:
					grps = [group_start , group_start + 1]# Bottom blocks
				groups.extend(grps)
			index = inds_B[-1][1]%2**l
			inds_B.append([groups,index])
			inds_A.append([groups,[ids for ids in range(2**l)]])

		else:
			if which ==0:
				if l+1 != L:
					group_start = inds_A[-1][0]*4  # group and index to decide the groups and indices of As. B will be all of it
					cond = inds_A[-1][1] < 2**l
				else:
					group_start = 0
					cond = inds_A[-1] < 2**l
				if cond:
					groups = [group_start, group_start + 1] # Top blocks
				else:
					groups = [group_start + 2, group_start + 3]# Bottom blocks
				if l+1 != L:
					index = inds_A[-1][1]%2**l
				else:
					index = inds_A[-1]%2**l
				inds_A.append([groups,index])
				inds_B.append([groups,[ids for ids in range(2**l)]])
			else:
				if l+1 !=L:
					group_start = inds_B[-1][0]*4
					cond = inds_B[-1][1] < 2**l
				else:
					group_start = 0
					cond = inds_B[-1] < 2**l
				if cond:
					groups = [group_start+2, group_start + 3] # Top blocks
				else:
					groups = [group_start , group_start + 1]# Bottom blocks
				if l+1 != L:
					index = inds_B[-1][1]%2**l
				else:
					index = inds_B[-1]%2**l
				inds_B.append([groups,index])
				inds_A.append([groups,[ids for ids in range(2**l)]])

	return inds_A,inds_B


def construct_butterfly_mat(shape,ranks,L,lc):
	lst_A,lst_D,lst_B = gen_all_matrices(shape,ranks,L,lc)
	blocks = len(lst_A[0])
	m = lst_A[0][0].shape[0]*blocks
	n = lst_B[0][0].shape[0]*blocks
	big_mat = np.zeros((m,n))
	for i in range(blocks):
		for j in range(blocks):
			#Figure out the i,j for each block and compute submatrices for the final result
			inds_A,ind_D,inds_B = get_indices(i,j,L,lc)
			#print('for i',i,' and j',j,'the indices for A',inds_A)
			#print('ind for B',inds_B)
			#print('ind for D',ind_D)
			#print('-------')
			# Get the corresponding matrices
			mats_A,mat_D,mats_B = get_mats(inds_A,ind_D,inds_B,lst_A,lst_D,lst_B)
			# Construct an einsum string (not really needed, all mat-mats)
			einstr = gen_einstr(len(mats_A))
			# Get the submatrix
			big_mat[i*int(m/blocks):(i+1)*int(m/blocks),
			j*int(n/blocks):(j+1)*int(n/blocks)] = np.einsum(einstr,*mats_A,mat_D,*mats_B,optimize=True)
			# The above will only be correct if lc= L/2, else we need to change
			# to sum the contributions of smaller blocks, would need more einsums
			# And add them up together. 
	return big_mat,[lst_A,lst_D,lst_B]


def const_butterfly_mat(shape,rank,L=1):
	m,n = shape
	#mat = np.zeros(shape)
	blocks = 2**L
	As = [np.random.uniform(low = -1, high =1, size=(int(m/blocks),rank)) for i in range(blocks)]
	Bs = [np.random.uniform(low = -1, high =1, size=(int(n/blocks),rank)) for i in range(blocks)]

	D = np.random.uniform(low=-1,high =1,size=(blocks*rank,blocks*rank))
	mat = compute_matrix_with_butterfly(As,D,Bs)
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




m = 24*16
n = 24*16

ranks = [24,6,3]
#rank =5
L=1
lc = 0
#T,originals = construct_butterfly_mat((m,n),ranks,L=4,lc=2)

#lst_A,lst_D,lst_B = gen_all_matrices(shape,ranks,L,lc)

inds_A,inds_B = figure_indices(solve_layer=3,solve_inds=[0,2],which=0,L=4,lc=2)
#print(inds_A)
#print(inds_B)
b = gen_solve_einstr(which=1,solve_layer=4,L=4,lc=2)

# T,originals = const_butterfly_mat((m,n), rank = rank)

# As = [np.random.uniform(low=-1,high = 1, size=(int(m/2),rank)) for i in range(2)]
# D = np.random.uniform(low=-1,high = 1, size=(2*rank,2*rank))
# Bs = [np.random.uniform(low=-1,high = 1, size=(int (n/2),rank)) for i in range(2)]


# Omega = create_omega(T.shape,sp_frac=0.4)

# T_sparse = np.einsum('ij,ij->ij',T,Omega,optimize=True)
# num_iter= 100
# regu = 0

# recon = compute_matrix_with_butterfly(As,D,Bs)
# error = la.norm(T - recon)
# print('Initial relative error is ',error/la.norm(T))





# for iters in range(num_iter):
# 	# Solve for D
# 	for i in range(2):
# 		for j in range(2):
# 			LHS = np.einsum('iz,jr,ij,il,jm->zrlm',As[i],Bs[j],Omega[i*int(m/2):(i+1)*int(m/2),
# 				j*int(n/2):(j+1)*int(n/2)],As[i],Bs[j],optimize=True).reshape((rank*rank,rank*rank))
# 			RHS = np.einsum('ij,iz,jr->zr',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
# 				As[i],Bs[j],optimize=True).reshape(-1)
# 			D[i*rank:(i+1)*rank,j*rank:(j+1)*rank] = la.solve(LHS + regu*np.eye(rank*rank),RHS).reshape((rank,rank)) #SPD Solve


# 	#Solve for A
# 	for i in range(2):
# 		LHS = np.zeros((int(T.shape[0]/2),rank,rank))
# 		RHS = np.zeros((int(T.shape[0]/2),rank))
# 		for j in range(2):
# 			LHS += np.einsum('jr,zr,ij,jp,lp->izl',Bs[j],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
# 				Bs[j],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
# 			RHS += np.einsum('ij,jr,lr->il',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],Bs[j],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
		
# 		for p in range(LHS.shape[0]):
# 			As[i][p,:] = la.solve(LHS[p] + regu*np.eye(rank),RHS[p,:]) #SPD Solve


# 	# Solve for B
# 	for j in range(2):
# 		LHS = np.zeros((int(T.shape[1]/2),rank,rank))
# 		RHS = np.zeros((int(T.shape[1]/2),rank))
# 		for i in range(2):
# 			LHS += np.einsum('ir,rz,ij,ip,pl->jzl',As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
# 				As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
# 			RHS += np.einsum('ij,ir,rl->jl',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],As[i],D[i*rank:(i+1)*rank,j*rank:(j+1)*rank],optimize=True)
		
# 		for p in range(LHS.shape[0]):
# 			Bs[j][p,:] = la.solve(LHS[p] + regu*np.eye(rank),RHS[p,:]) #SPD Solve
	
# 	recon = compute_matrix_with_butterfly(As,D,Bs)
# 	error = la.norm(T - recon)
# 	print('Relative error is ',error/la.norm(T))

# 	rel_err = error/la.norm(T)
# 	if rel_err < 1e-14:
# 		print('Converged in',iters)
# 		break


# Solve for L=1, lc =0. Test 

# for iters in range(num_iter):

# 	# Solve for all Ds first
# 	for d in range(len(lst_D)):
# 		# Each d gives a mapping back to i,j
# 		i = int(d/2)
# 		j = d%2
# 		LHS = np.einsum('al,ab,br,lm,mz, ap,pq,ls,st->rzqt',Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],lst_A[0][i],lst_A[1][d],lst_B[0][j],lst_B[1][d], 
# 			lst_A[0][i],lst_A[1][d],lst_B[0][j],lst_B[1][d],optimize=True).reshape((ranks[-1]*ranks[-1],ranks[-1]*ranks[-1]))
# 		RHS = np.einsum('al,ab,br,lm,mz->rz',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],lst_A[0][i],lst_A[1][d],lst_B[0][j],lst_B[1][d],
# 			optimize=True).reshape(-1)
# 		lst_D[d] = la.solve(LHS + regu*np.eye(ranks[-1]*ranks[-1]),RHS).reshape((ranks[-1],ranks[-1]))


# 	# Solve for level 1 As

# 	for d in range(len(lst_A[1])):
# 		i = int(d/2)
# 		j = d%2
# 		LHS = np.einsum('al,ar,lm,mn,zn,ap,lm,mn,qn->rzpq',Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
# 			lst_A[0][i],lst_B[0][j],lst_B[1][d],
# 			lst_D[d],lst_A[0][i],lst_B[0][j],lst_B[1][d],
# 			lst_D[d], optimize=True).reshape((ranks[-2]*ranks[-1],ranks[-2]*ranks[-1]))
# 		RHS =  np.einsum('al,ar,lm,mn,zn->rz',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],lst_A[0][i],lst_B[0][j],lst_B[1][d],
# 			lst_D[d],optimize=True).reshape(-1)


# 	# Solve for level 1 Bs

# 	for d in range(len(lst_B[1])):
# 		i = int(d/2)
# 		j = d%2
# 		LHS = np.einsum('la,ar,lm,mn,nz,ap,lm,mn,nq->rzpq',Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],lst_B[0][j],lst_A[0][i],lst_A[1][d],
# 			lst_D[d],lst_B[0][j],lst_A[0][i],lst_A[1][d],
# 			lst_D[d], optimize=True).reshape((ranks[-2]*ranks[-1],ranks[-2]*ranks[-1]))
# 		RHS =  np.einsum('la,ar,lm,mn,nz->rz',T_sparse[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],lst_B[0][j],lst_A[0][i],lst_A[1][d],
# 			lst_D[d],optimize=True).reshape(-1)


# 	# Solve for level 0 As

# 	for i in range(len(lst_A[0])):
# 		LHS = np.zeros((lst_A[0].shape[0],lst_A[0].shape[1],lst_A[0].shape[1]))
# 		for j in range(len(lst_B[0])):
# 			LHS += np.einsum('ij,jk,kl,ml,rm,jd,de,fe,zf->irz',Omega[i*int(m/2):(i+1)*int(m/2),j*int(n/2):(j+1)*int(n/2)],
# 				lst_B[0][j],lst_B[1][],lst_D[],lst_A[1][],lst_B[0][j],lst_B[1][],lst_D[],lst_A[1][] ,optimize=True)

# 			RHS += np.einsum()

# 		for p in range(lst_A[0].shape[0]):




	# Solve for level 0 Bs