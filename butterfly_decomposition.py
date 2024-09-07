import numpy as np
import time
import copy
import itertools
import numpy.linalg as la
import scipy.linalg as sla

from butterfly_helper_functions import *
from tensor_formulated_butterfly import get_butterfly_tens_from_mat,const_butterfly_tensor,gen_solve_einsum, solve_for_inner

# import signal

# def timeout_handler(signum, frame):
#     raise TimeoutError("Timed out!")

# # Set the signal handler
# signal.signal(signal.SIGALRM, timeout_handler)
# signal.alarm(5)  # Set a 5-second timeout



def get_index(i,L,c):
    ind_i = i
    left = []
    num1 = 2**L
    for m in range(L):
        val1 = int(ind_i >= num1//2)
        left.append(val1)
        num1 = num1//2
        if val1:
            ind_i -= num1
    return left

def get_butterfly_tens_from_factor(mat,L,lc,c):
    m = mat.shape[0]
    r = mat.shape[1]
    block_m = int(m/2**L)
    shape = [2 for l in range(L)]
    shape.append(block_m)
    shape.append(r)
    T = np.zeros(shape)
    for i in range(2**L):
        left = get_index(i,L,c)
        T[tuple(left +[slice(None)] )] = mat[c*i:c*(i+1),: ]
    return T

def butterfly_rhs_full(T,g_lst,h_lst,w,level,L,lc,extra=1):
    left_str = "".join([chr(ord('a')+j) for j in range(L +1)])
    right_str = "".join([chr(ord('A')+j) for j in range(L +1)])
    
    left_side_ranks = ""
    right_side_ranks = ""
    
    for j in range(L+1 , L+1 + L-lc+1 ):
        left_side_ranks += chr(ord('a') + j)
        right_side_ranks += chr(ord('A') + j)
        
    left_tensor_inds = left_str + left_side_ranks[0]
    right_tensor_inds = right_str +  right_side_ranks[0]
    
    #disjoint_str = "".join([chr(ord('s')+j) for j in range(lc)])
    
    
    if w==0:
        shape = list(h_lst[0].shape[(L-level):-1]) + [g_lst[L-level].shape[-1] +extra]
        #print(shape)
        einstr = left_str + right_str + ',' + right_str[(L-level):] + left_side_ranks[L-level] +'->'
        result_str = left_str + right_str[: (L-level)] + left_side_ranks[L-level]
        
        einstr += result_str
        random_mat = np.random.randn(*shape)
        #print('multiplying random matrix',einstr)
        mat = np.einsum(einstr,T,random_mat,optimize=True)
        
        result_str = left_str + right_tensor_inds[: (L-level)] + left_side_ranks[L-level]
        if level !=L:
            full_einstr =  left_str+ left_side_ranks[0]
            for i in range(L-level-1):
                full_einstr += ',' + left_str[:-( i +1)]+right_str[:i+1]+left_side_ranks[i:i+2] 
            
            full_einstr += ',' + result_str + '->'
            full_einstr += left_str[:-( L-level)]+right_str[:(L-level)]+left_side_ranks[(L-level-1):(L-level)+1]
            #print('einstring to get the factor',full_einstr)
            mat = np.einsum(full_einstr,*g_lst[:L-level],mat,optimize=True)
            
    else:
        shape = list(g_lst[0].shape[(L-level):-1]) + [h_lst[L-level].shape[-1] +extra]
        #print(shape)
        einstr = left_str + right_str + ',' + left_str[(L-level):]  +right_side_ranks[L-level] +'->'
        result_str = left_str[: (L-level)] + right_str   + right_side_ranks[L-level]
        
        einstr += result_str
        #print('multiplying random matrix',einstr)
        random_mat = np.random.randn(*shape)
        mat = np.einsum(einstr,T,random_mat,optimize=True)
        
        result_str = left_tensor_inds[: (L-level)] + right_str  + right_side_ranks[L-level]
        if level !=L:
            full_einstr =  right_str+ right_side_ranks[0]
            for i in range(L-level-1):
                full_einstr += ',' + left_str[:i+1]+ right_str[:-( i +1)]+ right_side_ranks[i:i+2] 
            
            full_einstr += ',' + result_str + '->'
            full_einstr +=  left_str[:(L-level)] + right_str[:-( L-level)]+ right_side_ranks[(L-level-1):(L-level)+1]
            #print('einstring to get the factor',full_einstr)
            mat = np.einsum(full_einstr,*h_lst[:L-level],mat,optimize=True)
        
            
    
    return mat,random_mat



def last_solve_full(T,g_lst,h_lst,L):
    lc = int(L/2)
    level = L - lc
    
    
    left_str = "".join([chr(ord('a')+j) for j in range(L +1)])
    right_str = "".join([chr(ord('A')+j) for j in range(L +1)])
    
    left_side_ranks = ""
    right_side_ranks = ""
    
    if w==0:
        rank1 = g_lst[-1].shape[-2]
        rank2 = g_lst[-1].shape[-1]
    else:
        rank1 = h_lst[-1].shape[-2]
        rank2 = h_lst[-1].shape[-1]
    
    for j in range(L+1 , L+1 + L-lc+1 ):
        left_side_ranks += chr(ord('a') + j)
        right_side_ranks += chr(ord('A') + j)
        
        # Note last ranks for lhs and rhs are the same, I dont have same indices here
        
    
    
    result_str = left_str[:(L-level)] +right_str[:(L-level)] + left_side_ranks[L-level] + right_side_ranks[L-level]
    full_einstr = left_str + right_str
    
    full_einstr += ',' + left_str + left_side_ranks[0]
    
    for i in range(L-level):
        full_einstr += ',' + left_str[:-(i+1)]+ right_str[ : (i+1)]+ left_side_ranks[i:i+2] 
    
    full_einstr += ',' + right_str + right_side_ranks[0]
    for i in range(L-level):
        full_einstr += ',' + left_str[:(i+1)]+ right_str[:-( i +1)]+ right_side_ranks[i:i+2] 
    
    full_einstr += '->' + result_str
    
    print(full_einstr)
    result = np.einsum(full_einstr, T, *g_lst,*h_lst,optimize=True)
    
    return result
    
        
        




def butterfly_rhs(left,right,g_lst,h_lst,w,level,L,lc,extra=1):
    left_str = "".join([chr(ord('a')+j) for j in range(L +1)])
    right_str = "".join([chr(ord('A')+j) for j in range(L +1)])
    
    left_side_ranks = ""
    right_side_ranks = ""
    
    for j in range(L+1 , L+1 + L-lc+1 ):
        left_side_ranks += chr(ord('a') + j)
        right_side_ranks += chr(ord('A') + j)
        
    left_tensor_inds = left_str + left_side_ranks[0]
    right_tensor_inds = right_str +  right_side_ranks[0]
    
    #disjoint_str = "".join([chr(ord('s')+j) for j in range(lc)])
    
    
    if w==0:
        shape = list(h_lst[0].shape[(L-level):-1]) + [g_lst[L-level].shape[-1] +extra]
        #print(shape)
        einstr = left_str +'z' + ',' + right_str + 'z' + ',' + right_str[(L-level):] + left_side_ranks[L-level] +'->'
        result_str = left_str + right_str[: (L-level)] + left_side_ranks[L-level]
        
        einstr += result_str
        random_mat = np.random.randn(*shape)
        #print('multiplying random matrix',einstr)
        mat = np.einsum(einstr,left,right,random_mat,optimize=True)
        
        result_str = left_str + right_tensor_inds[: (L-level)] + left_side_ranks[L-level]
        if level !=L:
            full_einstr =  left_str+ left_side_ranks[0]
            for i in range(L-level-1):
                full_einstr += ',' + left_str[:-( i +1)]+right_str[:i+1]+left_side_ranks[i:i+2] 
            
            full_einstr += ',' + result_str + '->'
            full_einstr += left_str[:-( L-level)]+right_str[:(L-level)]+left_side_ranks[(L-level-1):(L-level)+1]
            #print('einstring to get the factor',full_einstr)
            mat = np.einsum(full_einstr,*g_lst[:L-level],mat,optimize=True)
            
    else:
        shape = list(g_lst[0].shape[(L-level):-1]) + [h_lst[L-level].shape[-1] +extra]
        #print(shape)
        einstr = left_str + 'z' + ','+ right_str + 'z'+ ',' + left_str[(L-level):]  +right_side_ranks[L-level] +'->'
        result_str = left_str[: (L-level)] + right_str   + right_side_ranks[L-level]
        
        einstr += result_str
        #print('multiplying random matrix',einstr)
        random_mat = np.random.randn(*shape)
        mat = np.einsum(einstr,left,right,random_mat,optimize=True)
        
        result_str = left_tensor_inds[: (L-level)] + right_str  + right_side_ranks[L-level]
        if level !=L:
            full_einstr =  right_str+ right_side_ranks[0]
            for i in range(L-level-1):
                full_einstr += ',' + left_str[:i+1]+ right_str[:-( i +1)]+ right_side_ranks[i:i+2] 
            
            full_einstr += ',' + result_str + '->'
            full_einstr +=  left_str[:(L-level)] + right_str[:-( L-level)]+ right_side_ranks[(L-level-1):(L-level)+1]
            #print('einstring to get the factor',full_einstr)
            mat = np.einsum(full_einstr,*h_lst[:L-level],mat,optimize=True)
    return mat,random_mat

def last_solve(left,right,g_lst,h_lst,L):
    lc = int(L/2)
    level = L - lc
    

    
    left_str = "".join([chr(ord('a')+j) for j in range(L +1)])
    right_str = "".join([chr(ord('A')+j) for j in range(L +1)])
    
    left_side_ranks = ""
    right_side_ranks = ""
    
    
    for j in range(L+1 , L+1 + L-lc+1 ):
        left_side_ranks += chr(ord('a') + j)
        right_side_ranks += chr(ord('A') + j)
        
        # Note last ranks for lhs and rhs are the same, I dont have same indices here
        
    
    # Numpy cannot do it in one go
    # Breaking einsum down for it

    final_result_str = left_str[:(L-level)] +right_str[:(L-level)] + left_side_ranks[L-level] + right_side_ranks[L-level]
    


    full_einstr = left_str + 'z'  
    
    full_einstr += ',' + left_str + left_side_ranks[0]
    
    for i in range(L-level):
        full_einstr += ',' + left_str[:-(i+1)]+ right_str[ : (i+1)]+ left_side_ranks[i:i+2] 
    
    
    result1 = 'z' + left_str[:(L-level)] +right_str[:(L-level)] + left_side_ranks[-1]
    full_einstr +=  '->' + result1

    #print(full_einstr)
    inter = np.einsum(full_einstr,left,*g_lst,optimize=True)


    full_einstr = right_str +'z'
    full_einstr += ','+ right_str + right_side_ranks[0]
    for i in range(L-level):
        full_einstr += ',' + left_str[:(i+1)]+ right_str[:-( i +1)]+ right_side_ranks[i:i+2] 
    
    result2 = 'z' + left_str[:(L-level)] +right_str[:(L-level)] + right_side_ranks[-1]
    full_einstr += '->' + result2

    
    #print(full_einstr)
    inter2 = np.einsum(full_einstr,right,*h_lst,optimize=True)


    full_einstr = result1 + ',' + result2 + '->' + final_result_str
    #print(full_einstr)
    result = np.einsum(full_einstr, inter,inter2,optimize=True)
    
    return result
        
def qr_factor(factor,L,level,w,extra=1):
    shape = list(factor.shape)
    shape[-1] -= extra
    rank = shape[-1]
    output = np.zeros(shape)
    if level==L:
        for comb in itertools.product([0,1],repeat=level):
            tup = comb + (slice(None),slice(None))
            Q,_,_ = sla.qr(factor[tup],pivoting=True,mode='economic')
            c = Q.shape[0]
            nums = min(rank,c)      #already handled in user input tho
            output[tup] = Q[:,:nums]
    else:
        for comb in itertools.product([0,1],repeat=level):
            for comb2 in itertools.product([0,1],repeat= L- level ):
                if w==0:
                    tup1 = comb + (0,) + comb2 + (slice(None),slice(None))
                    tup2 = comb + (1,) + comb2 + (slice(None),slice(None))
                else:
                    tup1 = comb2 + comb + (0,) + (slice(None),slice(None))
                    tup2 = comb2 + comb + (1,) + (slice(None),slice(None))

                Q,_,_ = sla.qr(np.concatenate((factor[tup1],factor[tup2]), axis=0),pivoting=True,mode='economic')
                output[tup1] = Q[:factor[tup1].shape[0],:rank]
                output[tup2] = Q[factor[tup1].shape[0]:2*factor[tup1].shape[0],: rank]
    return output


    
    


def butterfly_decompose_low_rank(left_mat,right_mat,L,ranks,g_lst,h_lst):
    lc = int(L/2)
    c = g_lst[0].shape[-2]
    
    
    input_mat = left_mat@right_mat.T
    # create tensors from matrices
    left_tensor = get_butterfly_tens_from_factor(left_mat,L,lc,c)
    right_tensor = get_butterfly_tens_from_factor(right_mat,L,lc,c)
    
    extra = 2
    for level in range(L,lc-1,-1):
        print('at level',level)
        for w in range(2):
            factor,mat = butterfly_rhs(left_tensor,right_tensor,g_lst,h_lst,w,level,L,lc,extra=extra)
            Q = qr_factor(factor,L,level,w,extra=extra)
            if w ==0:
                g_lst[L-level] = Q
            else:
                h_lst[L-level] = Q
            #print('------------')
       
    mats = last_solve(left_tensor,right_tensor,g_lst,h_lst,L)
    g_lst = absorb_factor(mats,g_lst,L)
            
    
    recon = recon_butterfly_tensor( g_lst, h_lst, L, int(L/2))
    
    
    T = get_butterfly_tens_from_mat(input_mat,L,lc,c)
    error = la.norm(T - recon) / la.norm(T)
    
    print('error is',error)

    return g_lst,h_lst
    
    
def butterfly_decompose(T,L,ranks,g_lst,h_lst):
    lc = int(L/2)
    c = g_lst[0].shape[-2]
    
    recon = recon_butterfly_tensor(g_lst, h_lst, L, int(L/2))
    error = la.norm(T - recon) / la.norm(T)

    print('error is',error)
    
    extra = 2
    for level in range(L,lc-1,-1):
        print('at level',level)
        for w in range(2):
            factor,mat = butterfly_rhs_full(T,g_lst,h_lst,w,level,L,lc,extra=extra)
            Q = qr_factor(factor,L,level,w,extra=extra)
            if w ==0:
                g_lst[L-level] = Q
            else:
                h_lst[L-level] = Q
                
            #recon = recon_butterfly_tensor(g_lst, h_lst, L, int(L/2))
            #error = la.norm(T - recon) / la.norm(T)
    
            #print('error is',error)
    
    
    #T = get_butterfly_tens_from_mat(input_mat,L,lc,c)
    
    
    mats = last_solve_full(T,g_lst,h_lst,L)
    g_lst = absorb_factor(mats,g_lst,L)

    recon = recon_butterfly_tensor(g_lst, h_lst, L, int(L/2))
    error = la.norm(T - recon) / la.norm(T)

    print('error is',error)
    
    
    return g_lst,h_lst
    
    
def absorb_factor(mats,g_lst,L):
    

    lc = int(L/2)
    level = L - lc
    
    
    left_str = "".join([chr(ord('a')+j) for j in range(L +1)])
    right_str = "".join([chr(ord('A')+j) for j in range(L +1)])
    
    left_side_ranks = ""
    right_side_ranks = ""
    

    
    for j in range(L+1 , L+1 + L-lc+1 ):
        left_side_ranks += chr(ord('a') + j)
        right_side_ranks += chr(ord('A') + j)
        
        # Note last ranks for lhs and rhs are the same, I dont have same indices here
    if L != 0:
        mat_einstr = left_str[:(L-level)] +right_str[:(L-level )] + left_side_ranks[-1] + right_side_ranks[-1]
        

        fac_str =  left_str[: (L-level+1)]+ right_str[ : (L-level)]+ left_side_ranks[-2] + left_side_ranks[-1] 
        
        full_einstr =fac_str + ',' +   mat_einstr  
        
        fac_str = left_str[:(L-level+1)]+ right_str[ : (L-level )]+ left_side_ranks[-2] + right_side_ranks[-1]
        full_einstr += '->' + fac_str
    else:

        mat_einstr = left_side_ranks[-1] + right_side_ranks[-1]
        

        fac_str =  left_str + left_side_ranks[-1] 
        
        full_einstr =fac_str + ',' +   mat_einstr  
        
        fac_str = left_str + right_side_ranks[-1]
        full_einstr += '->' + fac_str

    #print(full_einstr)
    g_lst[-1] = np.einsum(full_einstr,g_lst[-1],mats,optimize=True)
    
    
    
    
    return g_lst
    
    
    
# L= 6
# c = 7
# lc = int(L/2)
# w = 1
# num = 3
# l = L- num
# I= c*2**L
# J = c*2**L
# r_BF= 6
# ranks = [r_BF for _ in range(L-lc+1 )]

# for i in range(len(ranks)):
#     if i==0:
#         ranks[0] = min(ranks[0],c)
#     else:
#         ranks[i] = min(2*ranks[i-1],ranks[i])
# print('ranks for butterfly decomposition are ', ranks)


# # # # print('size ',I)
# rng = np.random.RandomState(np.random.randint(1000))



# # left = np.random.randn(I,60)
# # right = np.random.randn(J,60)

# T,lst = const_butterfly_tensor(I,J,L,lc,ranks,rng)



# factor = qr_factor(lst[0],L=L,level=L,w=0,extra=0)

# recon = recon_butterfly_tensor(left, g_lst1, h_lst1, right, L, int(L/2))
    
    
# #T = get_butterfly_tens_from_mat(input_mat,L,lc,c)
# error = la.norm(T - recon) / la.norm(T)
# print('I:',I,'L:',L,'c:',c)

# print('shape before',left.shape)


# left_tensor = get_butterfly_tens_from_factor(left,L,lc,c)
# right_tensor = get_butterfly_tens_from_factor(right,L,lc,c)

# print('shape now',left_tensor.shape)


# print('starting einsum')

g_lst,h_lst = gen_tensor_inputs(I,J,L,lc,ranks,rng)


# level = L
# for i in range(len(g_lst)-1):
#     g_lst[i] = qr_factor(g_lst[i],L=L,level=level,w=0,extra=0)
#     h_lst[i] = qr_factor(h_lst[i],L=L,level=level,w=0,extra=0)
#     level-=1
    
#T = recon_butterfly_tensor(g_lst[0], g_lst[1:], h_lst[1:], h_lst[0], L, int(L/2))

# # print('--')
# mats = last_solve(left_tensor,right_tensor,g_lst,h_lst,L)
# g_lst = absorb_factor(mats,g_lst,L)

# print('ok done')


# g_einsum = copy.deepcopy(g_lst)
# h_einsum = copy.deepcopy(h_lst)

# if w==0:
#     g_einsum.pop(num)
# else:
#     h_einsum.pop(num)
# rhs_einsum,lhs_einsum = gen_solve_einsum(l=L-int(L/2),w=w,L=L,lc=int(L/2))

# print(rhs_einsum)
# print('--')
# print(lhs_einsum)    
    

# m = last_solve(left_tensor,right_tensor,g_lst,h_lst,L)
# s = time.time()
# # path = np.einsum_path(rhs_einsum,left_tensor,right_tensor,*g_einsum,*h_einsum[::-1],optimize=True)
# # print(path)
# # print('--------------')
# # lst = [(1,10),(0,2),(1,8),(0,2),(1,6),(0,2),(1,4),(0,2),(0,1)]
# # for i in range(1,len(path))
# # path[i] = lst[i-1]


# # try:
# #     rhs = np.einsum(rhs_einsum,left_tensor,right_tensor,*g_einsum,*h_einsum[::-1],optimize=path)
# # except TimeoutError:
# #     print("einsum operation took too long and was terminated")
# # finally:
# #     # Disable the alarm
# #     signal.alarm(0)
    
    
# #mat = last_solve(left_tensor,right_tensor,g_lst,h_lst,w,L,extra=1)



# g_lst, h_lst = butterfly_decompose(T,L,ranks,g_lst,h_lst)
# Q = qr_factor(rhs,L,l,w)
# print(Q.shape)


# e = time.time()

# print('how much time??',e-s)

# print('done')
