import numpy as np
import numpy.linalg as la
import sys
import time
import random


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


def gen_all_matrices(shape,ranks,L,lc,seed = 123):
	m,n = shape
	assert len(ranks) == (L-lc+1), 'length of ranks should be the same as number of layers'
	np.random.seed(seed)
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
			lst_D = [ np.random.uniform(low=-1,high =1,size=(ranks[-1],ranks[-1])) for i in range((blocks)**2 * 4**(p+1))]
		p+=1
	return lst_A,lst_D,lst_B

def get_mats(inds_A,ind_D,inds_B,lst_A,lst_D,lst_B):
	mats_A = [lst_A[0][inds_A[0]]]
	mats_B = [lst_B[0][inds_B[0]]]
	for i in range(1,len(inds_A)):
		mats_A.append(lst_A[i][inds_A[i][0]][inds_A[i][1]])
		mats_B.append(lst_B[i][inds_B[i][0]][inds_B[i][1]])

	
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
	d_index = index *(2**(2*lc)) + i*(2**lc) + j
	return inds_A,d_index,inds_B

def get_solve_matrices(inds_A,inds_B,lst_A,lst_B,solve_layer,solve_index,which,L):
	mats_A = []
	mats_B = []

	inds_for_A = []
	inds_for_B = []

	if which == 0:
		total_pass = len(inds_B[0])
	else:
		total_pass = len(inds_A[0])

	s_l = L-solve_layer
	for pas in range(total_pass):
		if which ==0:
			if solve_layer == L and inds_A[0] == solve_index:
				mats_A.append([])
				inds_for_A.append(['solve index'])
				
			else:
				mats_A.append([lst_A[0][inds_A[0]]])
				inds_for_A.append([inds_A[0]])
				
			inds_for_B.append([inds_B[0][pas]])
			mats_B.append([lst_B[0][inds_B[0][pas]]])
			for l in range(1,len(inds_A)):
				if isinstance(inds_B[l][0],list):
					group = pas //( total_pass//len(inds_B[l][0]))
					if s_l == l and [inds_A[l][0][group],inds_A[l][1]] == solve_index:
						inds_for_A[pas].extend(['solve index'])
					else:
						mats_A[pas].extend([lst_A[l][inds_A[l][0][group]][inds_A[l][1]]])
						inds_for_A[pas].extend([[inds_A[l][0][group],inds_A[l][1] ]])
						
					inds_for_B[pas].extend([[inds_B[l][0][group], inds_B[l][1][pas % int(total_pass/len(inds_B[l][0]))]]])
					mats_B[pas].extend([lst_B[l][inds_B[l][0][group]][inds_B[l][1][pas % int(total_pass/len(inds_B[l][0]))]]])

				else:
					if s_l == l and [inds_A[l][0],inds_A[l][1]] == solve_index:
						inds_for_A[pas].extend(['solve index'])
					else:
						mats_A[pas].extend([lst_A[l][inds_A[l][0]][inds_A[l][1]]])
						inds_for_A[pas].extend([[inds_A[l][0],inds_A[l][1]]])
						

					inds_for_B[pas].extend([[inds_B[l][0],inds_B[l][1][pas]]])
					mats_B[pas].extend([lst_B[l][inds_B[l][0]][inds_B[l][1][pas]]])




		else:
			if solve_layer == L and inds_B[0] == solve_index:
				inds_for_B.append(['solve index'])
				mats_B.append([])
				
			else:
				inds_for_B.append([inds_B[0]])
				mats_B.append([lst_B[0][inds_B[0]]])
				
			mats_A.append([lst_A[0][inds_A[0][pas]]])
			inds_for_A.append([inds_A[0][pas]])
			for l in range(1,len(inds_B)):
				if isinstance(inds_A[l][0],list):
					group = pas //( total_pass//len(inds_A[l][0]))
					if s_l == l and [inds_B[l][0][group],inds_B[l][1]] == solve_index:
						inds_for_B[pas].extend(['solve index'])
					else:
						mats_B[pas].extend([lst_B[l][inds_B[l][0][group]][inds_B[l][1]]])
						inds_for_B[pas].extend([[inds_B[l][0][group],inds_B[l][1] ]])
						

					mats_A[pas].extend([lst_A[l][inds_A[l][0][group]][inds_A[l][1][pas % int(total_pass/len(inds_A[l][0]))  ]] ])
					inds_for_A[pas].extend([[inds_A[l][0][group], inds_A[l][1][pas % int(total_pass/len(inds_A[l][0]))]]])
				else:
					if s_l == l and [inds_B[l][0],inds_B[l][1]] == solve_index:
						inds_for_B[pas].extend(['solve index'])
					else:
						mats_B[pas].extend([lst_B[l][inds_B[l][0]][inds_B[l][1]]])
						inds_for_B[pas].extend([[inds_B[l][0],inds_B[l][1]]])
						
					mats_A[pas].extend([lst_A[l][inds_A[l][0]][inds_A[l][1][pas]]])
					inds_for_A[pas].extend([[inds_A[l][0],inds_A[l][1][pas]]])


	#print('A',inds_for_A)
	#print('B',inds_for_B)
	return mats_A,mats_B,inds_for_A,inds_for_B


def gen_einstr(length):
	A_str = ""
	B_str = ""
	einstr = ""
	for i in range(length):
		if i != length-1:
			A_str += chr(ord('a')+i) + chr(ord('a')+i+1) + ','
			B_str += chr(ord('a')+length+i) + chr(ord('a')+length+i+1) + ','
		else:
			A_str += chr(ord('a')+i) + 'z,'
			B_str += chr(ord('a')+length+i) + 'z'
	#einstr += A_str + "yz,"+ B_str +"->" + "a"+chr(ord('a')+length)
	einstr += A_str + B_str +"->" + "a"+chr(ord('a')+length)
	return einstr

def gen_indices(L,s_l):
	if s_l == L:
		inds = list(np.arange(2**s_l))
	else:
		p = L - s_l
		inds =[[g,i] for g in range(4**p) for i in range(2**s_l)]
	return inds

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
				A_str += chr(ord('a')+i) + 'z,'
				B_str += chr(ord('a')+length+i) + 'z'
			else:
				if which==0:
					B_str += chr(ord('a')+length+i) + 'z'
				else:
					B_str = B_str[:-1]
					A_str += chr(ord('a')+i) + 'z,'

	#einstr += "a"+chr(ord('a')+length) +','+ A_str + "yz,"+ B_str

	einstr += "a"+chr(ord('a')+length) +','+ A_str + B_str	 
	if which ==0:
		if s_l != length -1:
			RHS = einstr +"->"+ chr(ord('a')+s_l) + chr(ord('a')+s_l+1)
		else:
			RHS = einstr +"->"+ chr(ord('a')+s_l) + 'z'
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
				if i==0:
					if which==0:
						B_str += chr(ord('a')+length+i ) + chr(ord('a')+3*length) + ','
					elif which ==1:
						A_str += chr(ord('a')+i) + chr(ord('a')+2*length+i) + ','
				else:
					if which==0:
						B_str += chr(ord('a')+3*length+i -1) + chr(ord('a')+3*length+i) + ','
					elif which ==1:
						A_str += chr(ord('a')+2*length+i -1) + chr(ord('a')+2*length+i) + ','
		else:
			if i != s_l:
				A_str += chr(ord('a')+2*length+i-1) + 'x,'
				B_str += chr(ord('a')+3*length+i-1) + 'x'
			else:
				if which ==0:
					B_str += chr(ord('a')+3*length+i -1) + 'x'
				else:
					B_str = B_str[:-1]
					A_str += chr(ord('a')+ 2*length+i-1) + 'x,'


	#LHS = einstr + ',' + A_str + "wx,"+ B_str + '->'

	LHS = einstr + ',' + A_str +  B_str + '->'

	if which == 0:
		if s_l != length - 1 and s_l != 0:
			LHS += chr(ord('a')+s_l) + chr(ord('a')+s_l+1) + chr(ord('a')+2*length+ s_l -1) + chr(ord('a')+ 2*length+ s_l)
		elif s_l ==length -1:
			LHS += chr(ord('a')+s_l) + 'z' + chr(ord('a')+2*length+ s_l -1) + 'x'
		else:
			LHS += chr(ord('a')+0) + chr(ord('a')+s_l+1) +  chr(ord('a')+ 2*length+ s_l)
	else: 
		if s_l != length - 1 and s_l != 0: 
			LHS += chr(ord('a')+length+ s_l) + chr(ord('a')+ length+ s_l+1) + chr(ord('a')+3*length+ s_l-1) + chr(ord('a')+ 3*length+ s_l)
		elif s_l ==length -1:
			LHS += chr(ord('a')+length+s_l) + 'z' + chr(ord('a')+3*length+ s_l -1) + 'x'
		else:
			LHS += chr(ord('a')+length) + chr(ord('a')+length+s_l+1) +  chr(ord('a')+ 3*length+ s_l)
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
				num = inds_A[0]%4

			shift_A = int(num/2)*2**(l-1)  # 2**blocks shift for bottom blocks
			shift_B =  int(num%2)*2**(l-1)
			if l != L:
				inds_A.insert(0,[group,inds_A[0][1]+shift_A])  # insert at the first position always
				inds_B.insert(0,[group,[ids+shift_B for ids in inds_B[0][1]]])
			else:
				inds_A.insert(0,inds_A[0][1]+shift_A)  # insert at the first position always
				inds_B.insert(0,[ids+shift_B for ids in inds_B[0][1]])
			
		else:
			# This should give me the group number for both A and B
			if solve_layer != L :
				group = inds_B[0][0]//4 # Which quadrant?
				# Below now gives if first half or second half
				num = inds_B[0][0]%4 # 0,1,2,3 (Z ordering)--> 0,1 means top blocks, 2,3 means bottom blocks
			else:
				num = inds_B[0]%4

			shift_B = int(num%2)*2**(l-1)  # 2**blocks shift for blocks 1 and 3
			shift_A = int(num/2)*2**(l-1)
			if l != L:  
				inds_B.insert(0,[group,inds_B[0][1]+shift_B])	 # insert at the first position always
				inds_A.insert(0,[group,[ids+shift_A for ids in inds_A[0][1]]])
			else:
				inds_B.insert(0,inds_B[0][1]+shift_B)	 # insert at the first position always
				inds_A.insert(0,[ids+shift_A for ids in inds_A[0][1]])


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
					grps = [group_start, group_start + 2] # Left blocks
				else:
					grps = [group_start  +1, group_start + 3]# Right blocks
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
					groups = [group_start , group_start + 2]  # Left blocks
				else:
					groups = [group_start + 1, group_start + 3 ]	# Right blocks

				if l+1 != L:
					index = inds_B[-1][1]%2**l
				else:
					index = inds_B[-1]%2**l

				inds_B.append([groups,index])
				inds_A.append([groups,[ids for ids in range(2**l)]])

	return inds_A,inds_B

def compute_matrix_with_butterfly_gen(lst_A, lst_D, lst_B, L, lc):
	blocks = len(lst_A[0])
	m = lst_A[0][0].shape[0]*blocks
	n = lst_B[0][0].shape[0]*blocks
	big_mat = np.zeros((m,n))
	for i in range(blocks):
		for j in range(blocks):
			inds_A,ind_D,inds_B = get_indices(i,j,L,lc)
			# Get the corresponding matrices
			mats_A,mat_D,mats_B = get_mats(inds_A,ind_D,inds_B,lst_A,lst_D,lst_B)
			# Construct an einsum string (not really needed, all mat-mats)

			einstr = gen_einstr(len(mats_A))
			# Get the submatrix
			#print(einstr)
			# big_mat[i*int(m/blocks):(i+1)*int(m/blocks),
			# j*int(n/blocks):(j+1)*int(n/blocks)] = np.einsum(einstr,*mats_A,mat_D,*mats_B,optimize=True)

			big_mat[i*int(m/blocks):(i+1)*int(m/blocks),
			j*int(n/blocks):(j+1)*int(n/blocks)] = np.einsum(einstr,*mats_A,*mats_B,optimize=True)
			# Removing D for now.
	return big_mat


def construct_butterfly_mat(shape,ranks,L,lc,seed=123):
	lst_A,lst_D,lst_B = gen_all_matrices(shape,ranks,L,lc,seed)
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
			# big_mat[i*int(m/blocks):(i+1)*int(m/blocks),
			# j*int(n/blocks):(j+1)*int(n/blocks)] = np.einsum(einstr,*mats_A,mat_D,*mats_B,optimize=True)
			# The above will only be correct if lc= L/2, else we need to change
			# to sum the contributions of smaller blocks, would need more einsums
			# And add them up together. 


			big_mat[i*int(m/blocks):(i+1)*int(m/blocks),
			j*int(n/blocks):(j+1)*int(n/blocks)] = np.einsum(einstr,*mats_A,*mats_B,optimize=True)

			# Removing D
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


def generate_identity_matrix(n, num_entries,seed):
    np.random.seed(seed)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        row_indices = np.random.choice(n, num_entries, replace=False)
        matrix[i, row_indices] = 1
    return matrix

def create_low_rank(shape,rank):
	m,n = shape
	A = np.random.uniform(low=-1,high = 1, size=(m,rank))
	B =  np.random.uniform(low=-1,high = 1, size=(n,rank))
	T =  np.einsum('ir,jr->ij',A,B,optimize=True)
	return T

def create_omega(shape,sp_frac,L,seed=123):
	np.random.seed(seed)
	omega = np.zeros(np.prod(shape))
	omega[:int(sp_frac*np.prod(shape))] = 1
	np.random.shuffle(omega)
	omega = omega.reshape(shape)
	# omega = np.zeros(shape)
	# rows = np.arange(int(shape[0]/2**L)) # assuming square shape
	# for i in range(2**L):
	# 	for j in range(2**L):
	# 		#M = np.zeros((int(shape[0]/2**lc), int(shape[0]/2**lc)))
	# 		M = generate_identity_matrix(int(shape[0]/2**L),num_entries=3,seed=seed)
	# 		#inds = random.sample(list(rows), len(rows))
	# 		#print('shape of M',M.shape)
	# 		#for row in range(len(rows)):
	# 		#		M[row,inds[row]] = 1
	# 		omega[i*int(shape[0]/2**L):(i+1)*int(shape[0]/2**L),j*int(shape[0]/2**L):(j+1)*int(shape[0]/2**L)] = M
	# print('Percentage of nonzeros in Omega is',(np.sum(np.sum(omega))/ np.prod(shape))*100,'%')
	return omega



#m = 64*16
#n = 64*16

m = 20*(2**8)
n = 20*(2**8)
#L = 4
#lc= 2

#ranks = [64,8,2]

L = 8

lc = 4
#ranks = [10,8,6]
#ranks = [ 2,2,2,2]
ranks = [ 2,2,2,2,2]
#rank =3 

start = time.time()
T,originals = construct_butterfly_mat((m,n),ranks,L=L,lc=lc,seed=523)
end = time.time()
#print('time taken',end -start)
# 523
#123
# 31232

# lst_D = originals[1][:]

# lst_B = originals[2][:]

# lst_A= originals[0][:]



sparse = [0.1]

num_iter= 500
regu = 1e-14

errors = []

for sp in sparse:
	lst_A,lst_D,lst_B = gen_all_matrices((m,n),ranks,L=L,lc=lc,seed= 143)
	Omega = create_omega(T.shape,sp_frac=sp,L=L,seed = 122)
	T_sparse = np.einsum('ij,ij->ij',T,Omega,optimize=True)
	print('starting solve')
	print('L and lc',L,lc)
	print('shape of matrix is',m,n)
	print('ranks are',ranks)
	print('number of entries available',int(sp* (m*n)), 'or',sp*100,'%')
	recon = compute_matrix_with_butterfly_gen(lst_A, lst_D, lst_B,L = L ,lc=lc)
	error = la.norm(T - recon)/la.norm(T)
	print('relative error before starting is',error)
	for iters in range(num_iter):
		# Solve for each level
		#Omega = create_omega(T.shape,sp_frac=sp,seed = 31232)
		#T_sparse = np.einsum('ij,ij->ij',T,Omega,optimize=True)
		p= -1
		for s_l in range(L,lc-1,-1):
			inds =  gen_indices(L=L,s_l=s_l)
			s_l_lst = L-s_l
			for w in range(2):
				for ind in inds:
					inds_A,inds_B = figure_indices(solve_layer=s_l,solve_inds=ind,which=w,L= L,lc= lc)
					mats_A,mats_B,inds_for_A,inds_for_B = get_solve_matrices(inds_A,inds_B,lst_A,lst_B,solve_layer= s_l, solve_index = ind,which=w,L=L)
					if w==0:
						omega_T_indices = [ [inds_for_A[l][0], inds_for_B[l][0]] for l in range(len(inds_for_B))]
					else:
						omega_T_indices =  [ [inds_for_A[l][0], inds_for_B[l][0]] for l in range(len(inds_for_A))]

					if s_l == L:
						if w == 0:
							omega_T_indices = [ [ind, el[1]] for el in omega_T_indices ]
							LHS = np.zeros((lst_A[0][ind].shape[0],ranks[0],ranks[0]))
							RHS = np.zeros((lst_A[0][ind].shape[0], ranks[0]))
						else:
							omega_T_indices = [ [el[0], ind] for el in omega_T_indices ]
							LHS = np.zeros((lst_B[0][ind].shape[0],ranks[0],ranks[0]))
							RHS = np.zeros((lst_B[0][ind].shape[0],ranks[0]))

					else:
						LHS = np.zeros((ranks[p]*ranks[(p+1)], ranks[p]*ranks[(p+1)]))
						RHS = np.zeros((ranks[p]*ranks[(p+1)]))

					# loop through solve matrices

					LHS_e,RHS_e = gen_solve_einstr(which=w,solve_layer=s_l,L=L,lc=lc)
					#print('inds_A',inds_A)
					#print('inds_B',inds_B)
					#print('LHS',LHS_e)
					#print('RHS',RHS_e)
					#print('taotal legnth A',len(mats_A))
					#print('otatal lenght B',len(mats_B))
					#print('inds_for_A',inds_for_A)
					#print('inds_for_B',inds_for_B)
					#recon = compute_matrix_with_butterfly_gen(lst_A, lst_D, lst_B,L = L ,lc=lc)
					#recon_sp = np.einsum('ij,ij->ij',recon,Omega,optimize=True)
					#right_hand_side = T_sparse - recon_sp
					right_hand_side = T_sparse 

					for l in range(len(mats_A)):
						if s_l != lc:
							D_index =  inds_for_A[l][-1][0]*(2**(2*lc)) + inds_for_A[l][-1][1]*(2**lc) + inds_for_B[l][-1][1]
							
						else:
							if w==0:
								D_index =  ind[0]*(2**(2*lc)) + ind[1]*(2**lc) + inds_for_B[l][-1][1]
							else:
								D_index =  inds_for_A[l][-1][0]*(2**(2*lc)) + inds_for_A[l][-1][1]*(2**lc) + ind[1]
						
						#right_hand_side = T_sparse[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)] 
						if s_l != L:
							# LHS += np.einsum(LHS_e,Omega[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/ 2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)],*mats_A[l],lst_D[D_index],*mats_B[l],*mats_A[l],lst_D[D_index],*mats_B[l],
							# 	optimize=True).reshape((ranks[p]*ranks[(p+1)], ranks[p]*ranks[(p+1)]))
							# RHS += np.einsum(RHS_e,T_sparse[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)],*mats_A[l],lst_D[D_index],*mats_B[l],
							# 	optimize=True).reshape(-1)
							
							LHS += np.einsum(LHS_e,Omega[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/ 2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)],*mats_A[l],*mats_B[l],*mats_A[l],*mats_B[l],
								optimize=True).reshape((ranks[p]*ranks[(p+1)], ranks[p]*ranks[(p+1)]))
							RHS += np.einsum(RHS_e,right_hand_side[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)] ,*mats_A[l],*mats_B[l],
								optimize=True).reshape(-1)
						else:
							# LHS += np.einsum(LHS_e,Omega[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/ 2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)],*mats_A[l],lst_D[D_index],*mats_B[l],*mats_A[l],lst_D[D_index],*mats_B[l],
							# 	optimize=True)
							# RHS += np.einsum(RHS_e,T_sparse[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)],*mats_A[l],lst_D[D_index],*mats_B[l],
							# 	optimize=True)

							LHS += np.einsum(LHS_e,Omega[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/ 2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)],*mats_A[l],*mats_B[l],*mats_A[l],*mats_B[l],
								optimize=True)
							RHS += np.einsum(RHS_e,right_hand_side[omega_T_indices[l][0]*int(m/2**L):(omega_T_indices[l][0]+1)*int(m/2**L),omega_T_indices[l][1]*int(n/2**L):(omega_T_indices[l][1]+1)*int(n/2**L)] ,*mats_A[l],*mats_B[l],
								optimize=True)
					# Perform the solve
					if s_l ==L:
						for row in range(LHS.shape[0]):
							if not np.allclose(RHS[row,:],np.zeros_like(RHS[row,:])):
								if w ==0:
									lst_A[0][ind][row,:] = la.solve(LHS[row] + regu* np.eye(ranks[0]),RHS[row,:])
								else:
									lst_B[0][ind][row,:] = la.solve(LHS[row] + regu* np.eye(ranks[0]),RHS[row,:])
					else:
						if not np.allclose(RHS,np.zeros_like(RHS)):
							if w==0:
								lst_A[s_l_lst][ind[0]][ind[1]] = la.solve(LHS + regu* np.eye(ranks[p]*ranks[(p+1)]), RHS).reshape((ranks[p],ranks[(p+1)]))
							else:
								lst_B[s_l_lst][ind[0]][ind[1]] = la.solve(LHS + regu* np.eye(ranks[p]*ranks[(p+1)]), RHS).reshape((ranks[p],ranks[(p+1)]))

				#	print('finished ind',ind,'for w',w,'layer',s_l)
				#print('finished w',w,'for layer',s_l)
			# incrementing rank parameter as we get to next layer
			p+=1
		recon = compute_matrix_with_butterfly_gen(lst_A, lst_D, lst_B, L = L ,lc=lc)
		error = la.norm(T - recon)/la.norm(T)
		print('relative error after iter',iters+1,'is',error)
		if error< 1e-8:
			print('converged')
			break
					
		error2 = la.norm(T_sparse - np.einsum('ij,ij->ij',Omega,recon,optimize=True))
		print('sparse absolute error is',iters+1,'is',error2)
	errors.append(error)

