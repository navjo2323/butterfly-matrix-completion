import numpy as np 
import numpy.linalg as la 
import itertools
import logging
import time

# convention to use l=L for outer ones, and l= lc for middle
# r_c= 0 means g_lst side and r_c = 1 h_lst side
# We are looking for T[...] = G_lst @ H_lst^H   {^H represents conjugate transpose}
# H_lst conjugate transposes 
# index convention is i_1 to i_L, j_1, j_L, i_0,j_0
# convention: all "i"s come first, 0 is the last index.


#When doing normal equations, anything with H i

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

def gen_tensor_inputs(m,n,L,lc,ranks,rng):
	block_m = int(m/2**L)
	block_n = int(n/2**L)
	twos = [2 for l in range(L)]
	sh1 = twos[:]
	sh1.append(block_m)
	sh1.append(ranks[0])
	sh2 = twos[:]
	sh2.append(block_n)
	sh2.append(ranks[0])

	g_lst = [rng.uniform(-1,1,size=(sh1))]
	h_lst = [rng.uniform(-1,1,size=(sh2))]


	sh1 = [2] + sh1
	sh2 = [2] + sh2
	for l in range(lc):
		sh1.pop(-2)
		sh1.append(ranks[l+1])
		sh1.pop(0)

		sh2.pop(-2)
		sh2.append(ranks[l+1])
		sh2.pop(0)

		g_lst.append(rng.uniform(-1,1,size=(sh2[:(l+1)] + sh1)))
		h_lst.append(rng.uniform(-1,1,size=(sh1[:(l+1)] +sh2)))

	return g_lst,h_lst


def figure_output_index(index,level,L,lc,r_c):
	first = index[:L]
	second = index[L:2*L]
	if level ==L:
		if r_c == 0:
			output = first + (index[-2],) + (slice(None),)
		else:
			output = second + (index[-1],) + (slice(None),)

	else:
		if r_c ==0:
			output = first[: L -(L-level) +1] + second[:(L-level)] + (slice(None),slice(None))
		else:
			output = first[:(L-level)] + second[: L - (L-level ) +1] + (slice(None),slice(None))

	return output

def multiply_mats(g_lst,h_lst,level,L,lc,r_c,index):
	ind_skip = L- level
	first_index = index[:L]
	second_index = index[L:2*L]


	if r_c == 0:
		output_index = second_index + (index[-1],) + (slice(None),)
		first = h_lst[0][output_index]
		
		for i in range(1,len(h_lst)):
			output_index = first_index[:i] + second_index[:L - i + 1] + (slice(None),slice(None))
			first = first.T@h_lst[i][output_index]

		for i in range(len(g_lst)-1,ind_skip,-1):
			output_index = first_index[:L - i + 1] + second_index[:i] + (slice(None),slice(None))
			first = g_lst[i][output_index]@first

		if ind_skip ==0:
			return first
		
		else:
			output_index = first_index + (index[-2],) + (slice(None),)
			second = g_lst[0][output_index]

			for i in range(1,ind_skip):
				output_index = first_index[:L - i + 1] + second_index[:i] + (slice(None),slice(None))
				second = second.T@g_lst[i][output_index]

			return np.outer(second,first)

	else:
		output_index = first_index + (index[-2],) + (slice(None),)
		first = g_lst[0][output_index]
		
		for i in range(1,len(g_lst)):
			output_index =  first_index[:L - i + 1] + second_index[:i] + (slice(None),slice(None))
			first = first.T@g_lst[i][output_index]

		for i in range(len(h_lst)-1,ind_skip,-1):
			output_index = first_index[:i] + second_index[:L - i + 1] + (slice(None),slice(None))
			first = h_lst[i][output_index]@first

		if ind_skip ==0:
			return first

		else:
			output_index = second_index + (index[-1],) + (slice(None),)
			second = h_lst[0][output_index]

			for i in range(1,ind_skip):
				output_index = first_index[:i] + second_index[:L - i + 1] + (slice(None),slice(None))
				second = second.T@h_lst[i][output_index]

			return np.outer(second,first)



def multiply_mats_all(g_lst,h_lst,index,L):
	frst_index = index[:L]
	scnd_index = index[L:2*L]

	output_index1 = frst_index + (index[-2],) + (slice(None),) 
	output_index2 = scnd_index + (index[-1],) + (slice(None),)
	first = g_lst[0][output_index1]
	second = h_lst[0][output_index2]

	for i in range(1,len(g_lst)):

		output_index1 = frst_index[:L - i + 1] + scnd_index[:i] + (slice(None),slice(None))
		output_index2 = frst_index[:i] + scnd_index[:L - i + 1 ] + (slice(None),slice(None))

		first = first.T@g_lst[i][output_index1]
		second = second.T@h_lst[i][output_index2]


	return np.inner(first,second)



def contract_RHS_T(T,inds,g_lst,h_lst,level,L,lc,r_c):
	# inds are given in tensor format instead of (m,n)
	assert ( level <= L and level >= lc)

	lst_ind = L- level 
	if r_c ==0:
		output = np.zeros_like(g_lst[lst_ind])
	else:
		output = np.zeros_like(h_lst[lst_ind])
	k=0
	for index in inds:
		output_index = figure_output_index(index,level,L,lc,r_c)
		output[output_index] += T[k]* multiply_mats(g_lst,h_lst,level,L,lc,r_c,index)
		k+=1
	return output

def ALS_solve(T,inds,g_lst,h_lst,level,L,lc,r_c,regu):
	assert ( level <= L and level >= lc)

	lst_ind = L- level 
	if r_c ==0:
		output = np.zeros_like(g_lst[lst_ind])
		if lst_ind ==0:
			LHS = np.zeros(g_lst[lst_ind].shape + (g_lst[lst_ind].shape[-1],))
		else:
			LHS = np.zeros(g_lst[lst_ind].shape[:-2] + (g_lst[lst_ind].shape[-1]*g_lst[lst_ind].shape[-2],g_lst[lst_ind].shape[-1]*g_lst[lst_ind].shape[-2]))
	else:
		output = np.zeros_like(h_lst[lst_ind])
		if lst_ind ==0:
			LHS = np.zeros(h_lst[lst_ind].shape + (h_lst[lst_ind].shape[-1],))
		else:
			LHS = np.zeros(h_lst[lst_ind].shape[:-2] + (h_lst[lst_ind].shape[-1]*h_lst[lst_ind].shape[-2],h_lst[lst_ind].shape[-1]*h_lst[lst_ind].shape[-2]))
	
	k=0
	#sort indices with respect to the solve index
	#then loop over the inds, accumulate and solve
	#do this later
	for index in inds:
		output_index1 = figure_output_index(index,level,L,lc,r_c)
		mats = multiply_mats(g_lst,h_lst,level,L,lc,r_c,index)
		output[output_index1] += T[k]*mats
		if lst_ind ==0:
			output_index2 = output_index1 + (slice(None),)
		else:
			output_index2 = output_index1
		LHS[output_index2] += np.outer(mats.reshape(-1),mats.reshape(-1)) 
		k+=1
	if lst_ind ==0:
		if r_c ==0:
			rows = g_lst[0].shape[-2]
		else:
			rows = h_lst[0].shape[-2]
		for combination in itertools.product([0, 1], repeat=L):
			for row in range(rows):
				if la.norm(output[combination+ (row,slice(None))]) > 1e-05:
					if r_c ==0:
						g_lst[lst_ind][combination+ (row,slice(None))] = la.solve(LHS[combination + (row,slice(None),slice(None))]+ regu*np.eye(output.shape[-1]),
							output[combination + (row,slice(None))])
					else:
						h_lst[lst_ind][combination+ (row,slice(None))] = la.solve(LHS[combination + (row,slice(None),slice(None))]+ regu*np.eye(output.shape[-1]),
							output[combination + (row,slice(None))])
	else:
		for combination in itertools.product([0, 1], repeat=L+1):
			if not np.allclose(output[combination + (slice(None), slice(None))],np.zeros_like(output[combination + (slice(None), slice(None))])):
				if r_c ==0:
					g_lst[lst_ind][combination + (slice(None),slice(None))] = la.solve(LHS[combination + (slice(None),slice(None))] + regu*np.eye(output.shape[-1]*output.shape[-2]),
						output[combination+ (slice(None),slice(None))].reshape(-1)).reshape((output.shape[-2],output.shape[-1]))
				else:
					h_lst[lst_ind][combination + (slice(None),slice(None))] = la.solve(LHS[combination + (slice(None),slice(None))] + regu*np.eye(output.shape[-1]*output.shape[-2]),
						output[combination+ (slice(None),slice(None))].reshape(-1)).reshape((output.shape[-2],output.shape[-1]))
	return g_lst,h_lst



def index_convert(indices, I, J, L):
    # Get tuples of (i,j) and convert to i_1 ... i_L j_1 ... j_L, i_0, j_0
    inds = []
    assert(I % 2**L == 0 and J % 2**L == 0)

    for ind in indices:
        assert(ind[0] < I and ind[1] < J)
        ind_i = ind[0]
        ind_j = ind[1]
        left = np.zeros(L, dtype=int)
        right = np.zeros(L, dtype=int)
        num1 = I
        num2 = J
        for m in range(L):
            val1 = int(ind_i >= num1//2)
            val2 = int(ind_j >= num2//2)
            left[m] = val1
            right[m] = val2
            num1 = num1//2
            num2 = num2//2
            if val1:
                ind_i -= num1
            if val2:
                ind_j -= num2
        inds.append(tuple(np.concatenate((left, right, [ind_i], [ind_j]))))
    return tuple(inds)



def contract_all(inds,g_lst,h_lst,L):
	# inds are given in tensor format instead of (m,n)
	i=0
	output = np.zeros((len(inds)))
	for index in inds:
		output[i] =  multiply_mats_all(g_lst,h_lst,index,L)
		i+=1

	return output


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

	if L != 0:
		full_string = left_tensor_inds+',' + ', '.join(left_lst_inds)+ ','+ ', '.join(right_lst_inds[::-1]) +','+ right_tensor_inds
	else:
		full_string = left_tensor_inds+',' + right_tensor_inds

	full_string += '->'+tensor_inds
	return full_string

def create_inds(I, J, nnz,rng):
    unique_tuples = set()
    while len(unique_tuples) < nnz:
        Is = rng.randint(low=0, high=I)
        Js = rng.randint(low=0, high=J)
        unique_tuples.add((Is, Js))
    return tuple(unique_tuples)

def const_butterfly_tensor(m,n,L,lc,ranks,rng):
	g_lst,h_lst = gen_tensor_inputs(m,n,L,lc,ranks,rng)
	strng = gen_einsum_string(L,lc)
	T = np.einsum(strng,g_lst[0],*g_lst[1:],*h_lst[1:][::-1],h_lst[0],optimize=True)
	return T,g_lst,h_lst

def recon_butterfly_tensor(g_lst,h_lst,L,lc):
	strng = gen_einsum_string(L,lc)
	T = np.einsum(strng,g_lst[0],*g_lst[1:],*h_lst[1:][::-1],h_lst[0],optimize=True)
	return T

def get_T_sparse(T,inds,L):
	T_sparse = np.zeros(len(inds))
	i=0
	for ind in inds:
		correct_index = ind[:L] + (ind[-2],) + ind[L:2*L] + (ind[-1],)
		T_sparse[i] = T[correct_index]
		i+=1
	return T_sparse

def butterfly_completer2(T,inds, L, g_lst, h_lst, num_iters, tol):
    #inds = index_convert(indices, I, J, L)
    T_sparse = get_T_sparse(T,inds,L)
    logging.debug('---------------Butterfly Completion------------------')

    nnz = len(inds)
    print("Number of observed entries:",nnz)
    
    errors = []
    
    recon = recon_butterfly_tensor(g_lst,h_lst, L, int(L/2))
    error = la.norm(T - recon) / la.norm(T)
    errors.append(error)
    recon_sparse = contract_all(inds,g_lst,h_lst,L)
    sparse_error = la.norm(T_sparse - recon_sparse) / la.norm(T_sparse)
    print('Initial relative error in observed entries:',sparse_error)
    print('Initial relative error in all of the tensor:',error)
    
    for iters in range(num_iters):
        s = time.time()
        print("Iteration", iters+1,"/",num_iters)

        for level in range(L,L//2-1,-1):
            print('At level: ',level)
            for r_c in range(2):
                g_lst,h_lst = ALS_solve(T_sparse,inds,g_lst,h_lst,level,L,L//2,r_c,regu=1e-8)
        
        e = time.time()
        print('Time in iteration', iters+1 ,':', e-s)
        
        recon = recon_butterfly_tensor(g_lst,h_lst, L, int(L/2))
        error = la.norm(T - recon) / la.norm(T)
        errors.append(error)
        recon_sparse = contract_all(inds,g_lst,h_lst,L)
        sparse_error = la.norm(T_sparse - recon_sparse) / la.norm(T_sparse)
        print('Relative error in observed entries: ',sparse_error)
        print('Relative error in all of the tensor after', iters + 1,' iterations: ',error)
        print('-----------------')
        if iters + 1 >= 5 and error >= 3:
        	print('Overfitting or error not reducing, stopping iterations')
        	break
        if error < tol:
            print('converged')
            break
    
    return g_lst, h_lst
