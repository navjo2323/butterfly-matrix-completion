import numpy as np
import numpy.linalg as la
import time
import copy
import itertools

# tensor formulation
def gen_einsum_string(L,lc):
	# The idea is that we will generate two set of strings one in small and one in caps to
	# utitlize max number of indices to scale upto large L. 
	# The small ones denote row blocks,rank modes for Gs, and physical index for rows of input
	# Large ones denote column blocks and rank modes for Hs, and physical index for columns of input

	left1 = "".join([chr(ord('a')+j) for j in range(L +1)])
	right1 = "".join([chr(ord('A')+j) for j in range(L +1)])

	tensor_inds = left1 + right1
	
	#left_side_ranks = "".join([chr(ord('a')+j for j in range(L-lc+ 1+1 , L-lc+ 1+1 + L-lc+2 ))])
	left_side_ranks = ""
	right_side_ranks = ""
	for j in range(L+1 , L+1 + L-lc+1 ):
		left_side_ranks += chr(ord('a') + j)
		right_side_ranks += chr(ord('A') + j)

	# Connect the last index of ranks
	left_side_ranks = left_side_ranks[:-1]+'z'
	right_side_ranks = right_side_ranks[:-1]+'z'
	
	left_tensor_inds = left1+ left_side_ranks[0]
	right_tensor_inds = right1+ right_side_ranks[0]
	
	left_lst_inds = []
	right_lst_inds = []
	for l in range(lc):
		left_lst_inds.append(left1[:-(l+1)]+right1[:l+1]+left_side_ranks[l:l+2])
		right_lst_inds.append(left1[:l+1]+right1[:-(l+1)]+right_side_ranks[l:l+2])

	full_string = left_tensor_inds+',' + ', '.join(left_lst_inds)+ ','+ ', '.join(right_lst_inds[::-1]) +','+ right_tensor_inds
	full_string += '->'+tensor_inds
	return full_string


def gen_tensor_inputs(m,n,L,lc,ranks):
	block_m = int(m/2**L)
	block_n = int(n/2**L)
	shape = [2 for l in range(2*L)]
	shape.append(block_m)
	shape.append(block_n)
	input_tensor = np.zeros(shape)
	twos = [2 for l in range(L)]
	sh1 = twos[:]
	sh1.append(block_m)
	sh1.append(ranks[0])
	sh2 = twos[:]
	sh2.append(block_n)
	sh2.append(ranks[0])

	left = np.random.uniform(-1,1,size=(sh1))
	right = np.random.uniform(-1,1,size=(sh2))

	g_lst = []
	h_lst = []
	sh1.pop(-2)
	sh1.append(ranks[1])
	sh2.pop(-2)
	sh2.append(ranks[1])

	p=2
	for l in range(lc):
		g_lst.append(np.random.uniform(-1,1,size=(sh2[:(l+1)] + sh1)))
		h_lst.append(np.random.uniform(-1,1,size=(sh1[:(l+1)] +sh2)))

		if l != lc-1:
			sh1.pop(0)
			sh1.pop(-2)
			sh1.append(ranks[p])
			sh2.pop(0)
			sh2.pop(-2)
			sh2.append(ranks[p])
			p+= 1

	return left,g_lst,h_lst,right

def const_butterfly_tensor(m,n,L,lc):
	left,g_lst,h_lst,right = gen_tensor_inputs(m,n,L,lc,ranks)
	strng = gen_einsum_string(L,lc)
	T = np.einsum(strng,left,*g_lst,*h_lst[::-1],right,optimize=True)
	return T,[left,g_lst,h_lst,right]

def create_omega(shape,sp_frac,L,seed=123):
	np.random.seed(seed)
	omega = np.zeros(np.prod(shape))
	omega[:int(sp_frac*np.prod(shape))] = 1
	np.random.shuffle(omega)
	omega = omega.reshape(shape)
	#omega = np.zeros(shape)
	#rows = np.arange(int(shape[0]/2**L)) # assuming square shape
	#omega = generate_identity_matrix(shape[0],num_entries=60,seed = seed)
	#for i in range(2**L):
	#	for j in range(2**L):
			#M = np.zeros((int(shape[0]/2**lc), int(shape[0]/2**lc)))
	#		M = generate_identity_matrix(int(shape[0]/2**L),num_entries=4,seed=seed)
			#inds = random.sample(list(rows), len(rows))
			#print('shape of M',M.shape)
			#for row in range(len(rows)):
			#		M[row,inds[row]] = 1
	#		omega[i*int(shape[0]/2**L):(i+1)*int(shape[0]/2**L),j*int(shape[0]/2**L):(j+1)*int(shape[0]/2**L)] = M
	# print('avg num entries per row',np.mean(np.sum(omega,axis=0)))
	# print('avg num entries per col',np.mean(np.sum(omega,axis=1)))
	# print('Percentage of nonzeros in Omega is',(np.sum(np.sum(omega))/ np.prod(shape))*100,'%')
	return omega

def recon_butterfly_tensor(left,g_lst,h_lst,right):
	strng = gen_einsum_string(L,lc)
	T = np.einsum(strng,left,*g_lst,*h_lst[::-1],right,optimize=True)
	return T

def gen_solve_einsum(l,w,L,lc):
	assert(l >= lc and l<= L)
	left = "".join([chr(ord('a')+j) for j in range(L +1)])
	right = "".join([chr(ord('A')+j) for j in range(L +1)])

	tensor_inds = left + right

	left_side_ranks = ""
	right_side_ranks = ""
	for j in range(L+1 , L+1 + L-lc+1 ):
		left_side_ranks += chr(ord('a') + j)
		right_side_ranks += chr(ord('A') + j)

	# Connect the last index of ranks
	left_side_ranks = left_side_ranks[:-1]+'z'
	right_side_ranks = right_side_ranks[:-1]+'z'

	left_tensor_inds = left+ left_side_ranks[0]
	right_tensor_inds = right+ right_side_ranks[0]

	left_lst_inds = []
	right_lst_inds = []


	left_side_ranks2 = ""
	right_side_ranks2 = ""
	for j in range(L+1 + L-lc+1, L+1 + L-lc+1 + L-lc+1 ):
		left_side_ranks2 += chr(ord('a') + j)
		right_side_ranks2 += chr(ord('A') + j)

	left_side_ranks2 = left_side_ranks2[:-1]+'Z'
	right_side_ranks2 = right_side_ranks2[:-1]+'Z'

	left_tensor_inds2 = left + left_side_ranks2[0]
	right_tensor_inds2 = right + right_side_ranks2[0]

	left_lst_inds2 = []
	right_lst_inds2 = []


	rhs_string = tensor_inds + ','
	lhs_string = ''

	if l==L:
		for layer in range(lc):
			left_lst_inds.append(left[:-(layer+1)]+right[:layer+1]+left_side_ranks[layer:layer+2])
			right_lst_inds.append(left[:layer+1]+right[:-(layer+1)]+right_side_ranks[layer:layer+2])

			left_lst_inds2.append(left[:-(layer+1)]+right[:layer+1]+left_side_ranks2[layer:layer+2])
			right_lst_inds2.append(left[:layer+1]+right[:-(layer+1)]+right_side_ranks2[layer:layer+2])

		if w==0:
			output = left + left_side_ranks[0]
			output2 = output + left_side_ranks2[0]
			rhs_string += ', '.join(left_lst_inds)+ ','+ ', '.join(right_lst_inds[::-1]) +','+ right_tensor_inds
			lhs_string += rhs_string + ',' + ', '.join(left_lst_inds2)+ ','+ ', '.join(right_lst_inds2[::-1]) +','+ right_tensor_inds2
		else:
			output = right + right_side_ranks[0]
			output2 = output + right_side_ranks2[0]
			rhs_string += left_tensor_inds+',' + ', '.join(left_lst_inds)+ ','+ ', '.join(right_lst_inds[::-1])		
			lhs_string += rhs_string + ',' + left_tensor_inds2 +',' + ', '.join(left_lst_inds2)+ ','+ ', '.join(right_lst_inds2[::-1])		
	
	else:
		l_trans = L-l -1
		for layer in range(lc):
			if layer !=l_trans:
				left_lst_inds.append(left[:-(layer+1)]+right[:layer+1]+left_side_ranks[layer:layer+2])
				right_lst_inds.append(left[:layer+1]+right[:-(layer+1)]+right_side_ranks[layer:layer+2])

				left_lst_inds2.append(left[:-(layer+1)]+right[:layer+1]+left_side_ranks2[layer:layer+2])
				right_lst_inds2.append(left[:layer+1]+right[:-(layer+1)]+right_side_ranks2[layer:layer+2])
			else:
				if w ==0:
					output = left[:-(layer +1)]+right[:layer+1]+left_side_ranks[layer:layer+2]
					right_lst_inds.append(left[:layer+1]+right[:-(layer+1)]+right_side_ranks[layer:layer+2])
					right_lst_inds2.append(left[:layer+1]+right[:-(layer+1)]+right_side_ranks2[layer:layer+2])
					output2 = output + left_side_ranks2[layer:layer+2]
				else:
					output = left[:layer+1]+right[:-(layer+1)]+right_side_ranks[layer:layer+2]
					output2 = output + right_side_ranks2[layer:layer+2]
					left_lst_inds.append(left[:-(layer+1)]+right[:layer+1]+left_side_ranks[layer:layer+2])
					left_lst_inds2.append(left[:-(layer+1)]+right[:layer+1]+left_side_ranks2[layer:layer+2])
					

		rhs_string += left_tensor_inds +',' + ', '.join(left_lst_inds) + ',' + ', '.join(right_lst_inds[::-1]) + ',' + right_tensor_inds
		lhs_string += rhs_string + ',' + left_tensor_inds2 + ',' + ', '.join(left_lst_inds2) + ',' + ', '.join(right_lst_inds2[::-1]) + ',' + right_tensor_inds2
	

	rhs_string += '->' + output
	lhs_string += '->' + output2
	
	return rhs_string,lhs_string

def solve_for_outer(w,L,T,Omega,left,g_lst,h_lst,right,regu=1e-14):

	rhs_einsum,lhs_einsum = gen_solve_einsum(l=L,w=w,L=L,lc=int(L/2))

	if w ==0:
		total_rows = left.shape[-2]
		rank = left.shape[-1]
	else:
		total_rows = right.shape[-2]
		rank = right.shape[-1]

	if w==0:
		LHS = np.einsum(lhs_einsum,Omega,*g_lst,*h_lst[::-1],right,*g_lst,*h_lst[::-1],right,optimize=True)
		RHS = np.einsum(rhs_einsum,T,*g_lst,*h_lst[::-1],right,optimize=True)
	else:
		LHS = np.einsum(lhs_einsum,Omega,left,*g_lst,*h_lst[::-1],left,*g_lst,*h_lst[::-1],optimize=True)
		RHS = np.einsum(rhs_einsum,T,left,*g_lst,*h_lst[::-1],optimize=True)

	for combination in itertools.product([0, 1], repeat=L):
		for row in range(total_rows):
			if la.norm(RHS[combination + (row,slice(None))]) > 1e-05:
				if w == 0:
					left[combination+ (row, slice(None))] = la.solve(LHS[combination + (row, slice(None), slice(None))] + regu*np.eye(rank), RHS[combination + (row,slice(None))])
				else:
					right[combination+ (row, slice(None))] = la.solve(LHS[combination + (row, slice(None), slice(None))] + regu*np.eye(rank), RHS[combination + (row,slice(None))])

	if w==0:
		return left
	else:
		return right

def solve_for_inner(w,L,l,T,Omega,left,g_lst,h_lst,right,regu=1e-14):
	rhs_einsum,lhs_einsum = gen_solve_einsum(l=l,w=w,L=L,lc=int(L/2))
	layer = L- l -1
	if w==0:
		new_lst = copy.deepcopy(g_lst)
		new_lst.pop(layer)
		LHS = np.einsum(lhs_einsum,Omega,left,*new_lst,*h_lst[::-1],right,left,*new_lst,*h_lst[::-1],right,optimize=True)
		RHS = np.einsum(rhs_einsum,T,left,*new_lst,*h_lst[::-1],right,optimize=True)
		ranks1 = g_lst[layer].shape[-2]
		ranks2 = g_lst[layer].shape[-1]

	else:
		new_lst = copy.deepcopy(h_lst)
		new_lst.pop(layer)
		LHS = np.einsum(lhs_einsum,Omega,left,*g_lst,*new_lst[::-1],right,left,*g_lst,*new_lst[::-1],right,optimize=True)
		RHS = np.einsum(rhs_einsum,T,left,*g_lst,*new_lst[::-1],right,optimize=True)
		ranks1 = h_lst[layer].shape[-2]
		ranks2 = h_lst[layer].shape[-1]


	for combination in itertools.product([0, 1], repeat=L+1):
		if w==0:
			g_lst[layer][combination + (slice(None), slice(None))] = la.solve(LHS[combination + (slice(None), slice(None), slice(None), slice(None) )].reshape(ranks[layer]*ranks[(layer+1)], ranks[layer]*ranks[(layer+1)]), 
				RHS[combination + (slice(None), slice(None))].reshape(-1)).reshape((ranks[layer],ranks[(layer+1)]))
		else:
			h_lst[layer][combination + (slice(None), slice(None))] = la.solve(LHS[combination + (slice(None), slice(None), slice(None), slice(None) )].reshape(ranks[layer]*ranks[(layer+1)], ranks[layer]*ranks[(layer+1)]), 
				RHS[combination + (slice(None), slice(None))].reshape(-1)).reshape((ranks[layer],ranks[(layer+1)]))
	if w==0:
		return g_lst
	else:
		return h_lst


L = 8

m = 20*(2**L)
n = 20*(2**L)

lc = int(L/2)
sp = 0.5
num_iters = 30

ranks = [2 for _ in range(L-lc+1)]


start = time.time()
T, originals = const_butterfly_tensor(m,n,L,lc)
end = time.time()
print('time taken is',end-start)

Omega = create_omega(T.shape,sp_frac=sp,L=L,seed=123)

T_sparse = T*Omega

left,g_lst,h_lst,right = gen_tensor_inputs(m,n,L,lc,ranks)

# left = originals[0]
# g_lst = originals[1]
# h_lst = originals[2]
# right = originals[3]

error = la.norm(T - recon_butterfly_tensor(left,g_lst,h_lst,right))/la.norm(T)
print('relative error',error)

for iters in range(num_iters):
	left = solve_for_outer(0,L,T_sparse,Omega,left,g_lst,h_lst,right)
	error = la.norm(T - recon_butterfly_tensor(left,g_lst,h_lst,right))/la.norm(T)

	right = solve_for_outer(1,L,T_sparse,Omega,left,g_lst,h_lst,right)
	error = la.norm(T - recon_butterfly_tensor(left,g_lst,h_lst,right))/la.norm(T)

	for l in range(int(L/2),L):
		g_lst =  solve_for_inner(0,L,l,T_sparse,Omega,left,g_lst,h_lst,right)
		error = la.norm(T - recon_butterfly_tensor(left,g_lst,h_lst,right))/la.norm(T)

		h_lst =  solve_for_inner(1,L,l,T_sparse,Omega,left,g_lst,h_lst,right)
		error = la.norm(T - recon_butterfly_tensor(left,g_lst,h_lst,right))/la.norm(T)

	print('relative error after',iters+1,'is',error)

