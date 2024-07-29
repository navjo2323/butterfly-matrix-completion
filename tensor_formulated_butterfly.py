import numpy as np
import numpy.linalg as la
import time
import copy
import itertools
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

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
    m = mat.shape[0]
    n = mat.shape[1]
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


def gen_tensor_inputs(m,n,L,lc,ranks,rng):
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

	left = rng.uniform(-5,5,size=(sh1))
	#left = rng.standard_normal(sh1)
	right = rng.uniform(-5,5,size=(sh2))
	#right = rng.standard_normal(sh2)
	g_lst = []
	h_lst = []
	sh1.pop(-2)
	sh1.append(ranks[1])
	sh2.pop(-2)
	sh2.append(ranks[1])

	p=2
	for l in range(lc):
		g_lst.append(rng.uniform(-5,5,size=(sh2[:(l+1)] + sh1)))
		#g_lst.append(rng.standard_normal(sh2[:(l+1)] + sh1))
		h_lst.append(rng.uniform(-5,5,size=(sh1[:(l+1)] +sh2)))
		#h_lst.append(rng.standard_normal(sh1[:(l+1)] +sh2))

		if l != lc-1:
			sh1.pop(0)
			sh1.pop(-2)
			sh1.append(ranks[p])
			sh2.pop(0)
			sh2.pop(-2)
			sh2.append(ranks[p])
			p+= 1

	return left,g_lst,h_lst,right

def const_butterfly_tensor(m,n,L,lc,ranks,rng):
	left,g_lst,h_lst,right = gen_tensor_inputs(m,n,L,lc,ranks,rng)
	strng = gen_einsum_string(L,lc)
	T = np.einsum(strng,left,*g_lst,*h_lst[::-1],right,optimize=True)
	return T,[left,g_lst,h_lst,right]

def create_omega(shape,nnz,L,rng):
	#np.random.seed(seed)
	omega = np.zeros(np.prod(shape))
	omega[:int(nnz)] = 1
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

def recon_butterfly_tensor(left,g_lst,h_lst,right,L,lc):
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
					
		if L != 2:
			rhs_string += left_tensor_inds +',' + ', '.join(left_lst_inds) + ',' + ', '.join(right_lst_inds[::-1]) + ',' + right_tensor_inds
			lhs_string += rhs_string + ',' + left_tensor_inds2 + ',' + ', '.join(left_lst_inds2) + ',' + ', '.join(right_lst_inds2[::-1]) + ',' + right_tensor_inds2
		
		else:
			if w ==0:
				rhs_string += left_tensor_inds +',' +  ', '.join(right_lst_inds[::-1]) + ',' + right_tensor_inds
				lhs_string += rhs_string + ',' + left_tensor_inds2 + ',' + ', '.join(right_lst_inds2[::-1]) + ',' + right_tensor_inds2
			else:
				rhs_string += left_tensor_inds +',' + ', '.join(left_lst_inds) + ',' +  right_tensor_inds
				lhs_string += rhs_string + ',' + left_tensor_inds2 + ',' + ', '.join(left_lst_inds2) + ','  + right_tensor_inds2

	rhs_string += '->' + output
	lhs_string += '->' + output2

	return rhs_string,lhs_string

def solve_for_outer(w,L,T,Omega,left,g_lst,h_lst,right,regu=1e-6):

	rhs_einsum,lhs_einsum = gen_solve_einsum(l=L,w=w,L=L,lc=int(L/2))
	trigger = 0


	if w ==0:
		total_rows = left.shape[-2]
		rank = left.shape[-1]
	else:
		total_rows = right.shape[-2]
		rank = right.shape[-1]

	s = time.time()
	if w==0:
		LHS = np.einsum(lhs_einsum,Omega,*g_lst,*h_lst[::-1],right,*g_lst,*h_lst[::-1],right,optimize=True)
		RHS = np.einsum(rhs_einsum,T,*g_lst,*h_lst[::-1],right,optimize=True)
	else:
		LHS = np.einsum(lhs_einsum,Omega,left,*g_lst,*h_lst[::-1],left,*g_lst,*h_lst[::-1],optimize=True)
		RHS = np.einsum(rhs_einsum,T,left,*g_lst,*h_lst[::-1],optimize=True)
	e = time.time()
	#print('time taken to compute LHS,RHS',e-s)
	s = time.time()
	for combination in itertools.product([0, 1], repeat=L):
		for row in range(total_rows):
			if la.norm(RHS[combination + (row,slice(None))]) > 1e-05:
				if w == 0:
					left[combination+ (row, slice(None))] = la.solve(LHS[combination + (row, slice(None), slice(None))] + regu*np.eye(rank), RHS[combination + (row,slice(None))])
				else:
					right[combination+ (row, slice(None))] = la.solve(LHS[combination + (row, slice(None), slice(None))] + regu*np.eye(rank), RHS[combination + (row,slice(None))])
			else:
				trigger = 1
	e = time.time()
	#print('time taken to solve',e-s)
	if w==0:
		return left,trigger
	else:
		return right,trigger

def solve_for_inner(w,L,l,T,Omega,left,g_lst,h_lst,right,regu=1e-6):
	rhs_einsum,lhs_einsum = gen_solve_einsum(l=l,w=w,L=L,lc=int(L/2))
	#print(lhs_einsum)
	layer = L- l -1

	if w==0:
		if L != 2:
			new_lst = copy.deepcopy(g_lst)
			new_lst.pop(layer)
			LHS = np.einsum(lhs_einsum,Omega,left,*new_lst,*h_lst[::-1],right,left,*new_lst,*h_lst[::-1],right,optimize=True)
			RHS = np.einsum(rhs_einsum,T,left,*new_lst,*h_lst[::-1],right,optimize=True)
			ranks1 = g_lst[layer].shape[-2]
			ranks2 = g_lst[layer].shape[-1]
		else:
			LHS = np.einsum(lhs_einsum,Omega,left,*h_lst[::-1],right,left,*h_lst[::-1],right,optimize=True)
			RHS = np.einsum(rhs_einsum,T,left,*h_lst[::-1],right,optimize=True)
			ranks1 = g_lst[layer].shape[-2]
			ranks2 = g_lst[layer].shape[-1]

	else:
		if L !=2:
			new_lst = copy.deepcopy(h_lst)
			new_lst.pop(layer)
			LHS = np.einsum(lhs_einsum,Omega,left,*g_lst,*new_lst[::-1],right,left,*g_lst,*new_lst[::-1],right,optimize=True)
			RHS = np.einsum(rhs_einsum,T,left,*g_lst,*new_lst[::-1],right,optimize=True)
			ranks1 = h_lst[layer].shape[-2]
			ranks2 = h_lst[layer].shape[-1]
		else:
			LHS = np.einsum(lhs_einsum,Omega,left,*g_lst,right,left,*g_lst,right,optimize=True)
			RHS = np.einsum(rhs_einsum,T,left,*g_lst,right,optimize=True)
			ranks1 = h_lst[layer].shape[-2]
			ranks2 = h_lst[layer].shape[-1]


	for combination in itertools.product([0, 1], repeat=L+1):
		if not np.allclose(RHS[combination + (slice(None), slice(None))],np.zeros_like(RHS[combination + (slice(None), slice(None))])):
			if w==0:
				g_lst[layer][combination + (slice(None), slice(None))] = la.solve(LHS[combination + (slice(None), slice(None), slice(None), slice(None) )].reshape((ranks1*ranks2, ranks1*ranks2))
					+regu*np.eye(ranks1*ranks2), 
					RHS[combination + (slice(None), slice(None))].reshape(-1)).reshape((ranks1,ranks2))
			else:
				h_lst[layer][combination + (slice(None), slice(None))] = la.solve(LHS[combination + (slice(None), slice(None), slice(None), slice(None) )].reshape((ranks1*ranks2, ranks1*ranks2))
					+regu*np.eye(ranks1*ranks2), 
					RHS[combination + (slice(None), slice(None))].reshape(-1)).reshape((ranks1,ranks2))
	if w==0:
		return g_lst
	else:
		return h_lst


def check_omega(omega,L):
	for combination in itertools.product([0, 1], repeat=2*L):
		if la.norm(omega[combination + (slice(None), slice(None))])< 1e-06:
			print('problem at block',combination)

            


            
            
def butterfly_completer(T_sparse, T, Omega, L, left, g_lst, h_lst, right, num_iters, tol):
    logging.debug('---------------Butterfly Completion------------------')
    
    nnz = np.sum(Omega)
    logging.debug(f"Number of observed entries: {nnz}")
    
    errors = []
    
    recon = recon_butterfly_tensor(left, g_lst, h_lst, right, L, int(L/2))
    error = la.norm(T - recon) / la.norm(T)
    errors.append(error)
    sparse_error = la.norm(T_sparse - Omega * recon) / la.norm(T_sparse)
    logging.debug(f'Initial relative error in observed entries: {sparse_error}')
    logging.debug(f'Initial relative error in all of the tensor: {error}')
    
    for iters in range(num_iters):
        s = time.time()
        logging.debug(f"Iteration {iters+1}/{num_iters}")
        left, trig = solve_for_outer(0, L, T_sparse, Omega, left, g_lst, h_lst, right)
        if trig:
            logging.debug('trig: no rows to solve')

        right, trig = solve_for_outer(1, L, T_sparse, Omega, left, g_lst, h_lst, right)
        if trig:
            logging.debug('trig: no rows to solve')

        for l in range(L-1, int(L/2)-1, -1):
            g_lst = solve_for_inner(0, L, l, T_sparse, Omega, left, g_lst, h_lst, right)

        for l in range(int(L/2), L, 1):
            h_lst = solve_for_inner(1, L, l, T_sparse, Omega, left, g_lst, h_lst, right)
        e = time.time()
        logging.debug(f'Time in iteration {iters+1}: {e-s}')
        recon = recon_butterfly_tensor(left, g_lst, h_lst, right, L, int(L/2))
        error = la.norm(T - recon) / la.norm(T)
        errors.append(error)
        sparse_error = la.norm(T_sparse - Omega * recon) / la.norm(T_sparse)
        logging.debug(f'Relative error in observed entries: {sparse_error}')
        logging.debug(f'Relative error in all of the tensor after {iters + 1} iterations: {error}')
        logging.debug('-----------------')
        if iters + 1 >= 5 and error >= 3:
            logging.debug('Overfitting or error not reducing, stopping iterations')
            break
        if error < tol:
            logging.debug('converged')
            break
    
    return left, g_lst, h_lst, right