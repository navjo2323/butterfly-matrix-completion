import numpy as np
import numpy.linalg as la
import time

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


def gen_rhs_einsum(l,w,L,lc):
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



L = 4

m = 20*(2**L)
n = 20*(2**L)

lc = int(L/2)

ranks = [2 for _ in range(L-lc+1)]

#left,g_lst,h_lst,right = const_input_via_tensor(m,n,L,lc,ranks)
einstr = gen_einsum_string(L=L,lc=lc)


# start = time.time()
# T, originals = const_butterfly_tensor(m,n,L,lc)
# end = time.time()
# print('time taken is',end-start)

einstr1,einstr2 = gen_rhs_einsum(l=4,w=1,L=L,lc= lc)
print(einstr1)
print(einstr2)
