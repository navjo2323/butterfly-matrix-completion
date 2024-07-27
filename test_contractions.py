import numpy as np
import numpy.linalg  as la
import time
from tensor_formulated_butterfly import *
from butterfly_helper_functions import *



def Omega_T_for_test(I,J,inds,shape,L):
	T_arr = np.random.randn(len(inds))

	# rows, cols = zip(*inds)
	# Omega = np.zeros((I,J))
	# T_tensor = np.zeros((I,J))
	# Omega[rows,cols] = 1.0
	# k = 0
	# for i in range(len(rows)):
	# 	T_tensor[rows[i],cols[i]] = T_arr[k]
	# 	k+= 1
	Omega = np.zeros((shape))
	T_tensor = np.zeros((shape))
	i=0

	for ind in inds:
		correct_index = ind[:L] + (ind[-2],) + ind[L:2*L] + (ind[-1],)
		T_tensor[correct_index] = T_arr[i]
		Omega[correct_index] = 1.0
		i+=1

	return T_arr,Omega,T_tensor

nnz = 5000000
c = 4
L= 12
lc = int(L/2)
I= c*2**L
J = c*2**L
r_c = 1
num = 2
level = L- num

print('ratio of nnz',nnz/(I*J))

regu = 1e-6

shape = tuple([2 for l in range(L)]) + (int(I/2**L),) + tuple([2 for l in range(L)]) + (int(J/2**L),)


rng = np.random.RandomState(np.random.randint(10000))
ranks = [2 for i in range(L-lc+1)]
indices = create_inds(I,J,nnz,rng)
print('inds created')


inds = index_convert(indices,I,J,L)
print('index converted')

T_arr, Omega, T_tensor = Omega_T_for_test(I,J,inds,shape,L)
print('array is ready')
#print(np.sum(Omega))
#print(la.norm(T_arr) - la.norm(T_tensor))

g_lst,h_lst = gen_tensor_inputs(I,J,L,lc,ranks,rng)

left = g_lst[0].copy()
g_lst1 = copy.deepcopy(g_lst[1:])
h_lst1 = copy.deepcopy(h_lst[1:])
right = h_lst[0].copy()

#error = la.norm(T_tensor - Omega*recon_butterfly_tensor(g_lst[0],g_lst[1:],h_lst[1:],h_lst[0],L,lc))/la.norm(T_tensor)
#error2 = la.norm(T_arr- contract_all(inds,g_lst,h_lst,L))/la.norm(T_arr)
#print('check errors are same',error-error2)

print('starting solve now')
s = time.time()
#output1 = contract_RHS_T(T_arr,inds,g_lst,h_lst,level=level,L=L,lc=lc,r_c=r_c)
lst1,lst2 = ALS_solve(T_arr,inds,g_lst,h_lst,level=level,L=L,lc=lc,r_c=r_c,regu=1e-06)
#answer1 = lst1[L - level].copy()
answer1 = lst2[L - level].copy()
e = time.time()
print('time taken for new',e-s)


# s = time.time()
# rhs_einsum,lhs_einsum = gen_solve_einsum(l=level,w=r_c,L=L,lc=int(L/2))

#output2 = np.einsum(rhs_einsum,T_tensor,*g_lst[1:],*h_lst[1:][::-1],h_lst[0],optimize=True)
#output2 = np.einsum(rhs_einsum,T_tensor,g_lst[0],*g_lst[1:],*h_lst[1:][::-1],optimize=True)

# new_lst = copy.deepcopy(g_lst1)
# layer = L- level -1
# new_lst.pop(layer)
# output2 = np.einsum(rhs_einsum,T_tensor,left,*new_lst,*h_lst1[::-1],right,optimize=True)
#output2 = np.einsum(rhs_einsum,T_tensor,left,*g_lst1,*new_lst[::-1],right,optimize=True)
#output2,trigger = solve_for_outer(r_c,L,T_tensor,Omega,left,g_lst1,h_lst1,right,regu=1e-06)
# lst = solve_for_inner(r_c,L,level,T_tensor,Omega,left,g_lst1,h_lst1,right,regu=1e-06)
# answer2 = lst[num-1]
#output2 = lst[0]

# e = time.time()

# print('time taken for old',e-s)


#print('error norm is',la.norm(output1-output2))
#print('norm is',la.norm( answer2 - answer1))
#print('norm diff lhs',la.norm(LHS1 - LHS2))
#.reshape((ranks1*ranks2, ranks1*ranks2))
