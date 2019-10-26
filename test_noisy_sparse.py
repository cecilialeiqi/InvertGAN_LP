import numpy as np
import math
import copy
from scipy.optimize import linprog
from decimal import *
import torch
import cvxopt
import argparse
import time
import matplotlib.pyplot as plt

from cvxopt import info
print info.version

getcontext().prec = 28

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('activation', type=str, default='ReLU', help='activation type, ReLU or LeakyReLU')
    parser.add_argument('l', type= int, default=3, help='number of layers')
    parser.add_argument('use_true_nets',type=int, default=1, help='whether to use true nets')
    parser.add_argument('net_name',type=str,default='./network/trained_vae_20_60_784_nobias.pth')
    parser.add_argument('norm_type',type=int,default=0,help='0 or 1, random noise type')
    parser.add_argument('sizes',type=str,default='20,60,784')
    parser.add_argument('task',type=str,default='invert', help='task from sensing, invert, and choose_k')
    parser.add_argument('method', type=str,default='linf', help='linf, l1, relax')
    return parser.parse_args()

opt = parse_args()

# set width and length
theta = 1.2
activation = opt.activation # 'LeakyReLU'
task = opt.task
method = opt.method
l = int(opt.l) # number of layers
norm_type = opt.norm_type #1
use_true_nets = opt.use_true_nets #False

sizes = [int(i) for i in opt.sizes.split(',')]

weights={}
bias={}
layers={}

def infer_per_layer(weight, bias, z, act = 'ReLU'):
    z = weight.dot(z)
    zpos = np.maximum(z, np.zeros(len(z)))
    if act == 'ReLU':
        return zpos
    elif act == 'LeakyReLU':
        zneg = np.minimum(z, np.zeros(len(z)))
        return zpos+0.1*zneg
    else:
        exit('Error!! Invalid activation')

def inference(weights, bias, z, error=0, act = 'ReLU'):
    for i in range(l-1):
        layers[i] = z
        z = infer_per_layer(weights[i], bias[i], z, act)
    layers[l-1] = z + error
    # error is guaranteed to be small
    return layers[l-1]

def get_weights(sizes, random=not use_true_nets):
    for i in range(l-1):
        # this part is with the real matrices
        if not random:
            # this part is to load real matrix
            state = torch.load(opt.net_name, map_location = 'cpu')
            wt = state['fc'+str(i+3)+'.weight']
            weights[i] = np.array(wt)# """
        else:
            # this part is with the random matrices
            weights[i] = np.random.normal(0,1,(sizes[i+1],sizes[i]))
            weights[i] /= np.sqrt(len(weights[i]))
        bias[i] = np.zeros(sizes[i+1])
        print(weights[i].shape)
    return weights

def check_available(A_ub, x, b_ub):
    result = np.dot(A_ub, x)
    b_ub = np.array(b_ub)
    #print('result: {0}, x: {1}, b: {2}'.format(result, x, b_ub))
    if np.max(result.reshape(-1) - b_ub.reshape(-1))>0:
        #for i in range(len(result)):
        #    print(result[i], b_ub[i], result[i]-b_ub[i])
        return False
    else:
        return True

def method_lp_relax(weight, bias, current, thres, first_layer, truth, last_layer=False, act = 'ReLU'):
    c = np.zeros(len(weight[0,:]))
    for i in range(len(weight)):
        coeff = max(current[i],0)
        if coeff < 2 * thres:
            coeff = 0
        if act == 'ReLU':
            c += weight[i,:] * coeff
        else:
            c += weight[i,:] * current[i]
    c_origin = c
    c = cvxopt.matrix(c[:])
    A_ub = weight[:]
    b_ub = np.maximum(0, current + thres)
    if act == 'LeakyReLU':
        A_ub = np.vstack([A_ub, -A_ub])
        b_ub = np.hstack([b_ub, -10 * np.minimum(0, current-thres)])
    if act == 'ReLU' and not first_layer:
        A_ub = np.vstack([A_ub, -np.identity(weight.shape[1])])
        b_ub = np.hstack([b_ub, np.zeros(weight.shape[1])])
    A_ub = A_ub.astype(np.double)
    A_ub = cvxopt.matrix(A_ub)
    b_ub = cvxopt.matrix(b_ub)
    sol = cvxopt.solvers.lp(-c, A_ub, b_ub, solver='cvxopt')
    if sol['status'] =='primal infeasible':
        return False, [], 0
    #print(sol)
    res = np.array(sol['x']).reshape((len(c),))
    flag = check_available(A_ub, truth, b_ub)
    print('whether available:', flag)
    max_true = c_origin.dot(truth)
    max_derived = c_origin.dot(res)
    pb = 0
    for i in range(len(current)):
        pb += max(0, current[i]) * (current[i] + thres)
    print('primal bound is:', pb)
    if flag and max_true>max_derived:
        print('weird!!, real value is clearly better and also feasible')
    if sol['status'] =='primal infeasible':
        return False, 0, 0
    return True, res, max_derived


def method_linprog(weight, bias, current, thres, first_layer, truth, last_layer=False, act = 'ReLU'):
    if act == 'ReLU':
        ind_ub = current <= thres
        # in this part, A[ind] z <= 2*thres
        ind_ul = current > thres
        n_eq = np.sum([1 if i else 0 for i in ind_ul])
        n_in = len(current) - n_eq
        # in this part, current[ind]-thres <= A[ind] z <= current[ind]+thres
        # also, we require z >= 0 if not first_layer
        print('{0} equality constraints vs {1} inequalitys, and {2} variables'.format(n_eq, n_in, len(truth)))
        #A_ub = weight[ind_ub,:]

        A_ub = np.hstack([weight, -np.ones((weight.shape[0],1))])
        #print(len(weight[ind_ul]), n_eq)
        A_lb = np.hstack([-weight[ind_ul,:],-np.ones((n_eq,1))])
        A = np.vstack([A_ub, A_lb])
        tmp = np.zeros(weight.shape[1] +1 )
        tmp[-1] = 1
        A = np.vstack([A, tmp])
        A = np.vstack([A, -tmp])
        if act == 'LeakyReLU':
            obs = current[:]
            obs[current<=thres] = 0
            b_ub = np.hstack([-bias + obs, -current[ind_ul]+bias[ind_ul]])
        else:
            print(weight.shape)
            print(len(bias),len(current), len(current[ind_ul]), len(bias[ind_ul]))
            b_ub = np.hstack([-bias+current, -current[ind_ul]+bias[ind_ul]])
        b = np.hstack([b_ub, thres])
        b = np.hstack([b, 0])
    else:
        ind_ge=current > thres
        ind_le=current < -thres
        ind_lg = np.logical_and(current<=thres, current>=-thres)
        A_ub1 = np.hstack([weight[ind_ge], -np.ones((weight[ind_ge].shape[0],1))])
        A_ub2 = np.hstack([weight[ind_lg], -np.ones((weight[ind_lg].shape[0],1))])
        A_ub3 = np.hstack([weight[ind_le], -10*np.ones((weight[ind_le].shape[0],1))])
        A_lb1 = np.hstack([-weight[ind_ge], -np.ones((weight[ind_ge].shape[0],1))])
        A_lb2 = np.hstack([-weight[ind_lg], -10*np.ones((weight[ind_lg].shape[0],1))])
        A_lb3 = np.hstack([-weight[ind_le], -10*np.ones((weight[ind_le].shape[0],1))])
        A = np.vstack([A_ub1,A_ub2,A_ub3,A_lb1,A_lb2,A_lb3])
        tmp = np.zeros(weight.shape[1] + 1)
        tmp[-1] = 1
        A = np.vstack([A, tmp])
        A = np.vstack([A, -tmp])
        b_ub = np.hstack([current[ind_ge], current[ind_lg], 10*current[ind_le],
            -current[ind_ge], -10*current[ind_lg], -10*current[ind_le]])
        b = np.hstack([b_ub, thres])
        b = np.hstack([b, 0])

    if act == 'ReLU' and not first_layer:
        tmpm = np.zeros((weight.shape[1], weight.shape[1]+1))
        tmpm[:,:-1] = -np.identity(weight.shape[1])
        A = np.vstack([A, tmpm])
        b = np.hstack([b, np.zeros(weight.shape[1])])
    c = cvxopt.matrix(tmp)
    print(A.shape, b.shape)
    A = A.astype(np.double)
    b = b.astype(np.double)
    A_ub = cvxopt.matrix(A)
    b_ub = cvxopt.matrix(b)
    sol = cvxopt.solvers.lp(c, A_ub, b_ub, solver='cvxopt')
    if sol['status'] =='primal infeasible':
        return False, []
    print(check_available(A_ub, np.hstack([truth,thres]), b_ub))
    res = np.array(sol['x']).reshape((len(c),))
    print(check_available(A_ub, res, b_ub), res[-1])
    return True, res[:-1]

def loss(A, z, obs, act = 'ReLU'):
    return np.linalg.norm(obs - A.dot(inference(weights, np.zeros(sizes[-1]), z, np.zeros(sizes[-1]), act)))

def test_sensing(ind, output, e, method = 'linprog', act='ReLU'):
    weights = get_weights(sizes)
    weight = weights[len(sizes)-2]
    weights[len(sizes)-2] = weight[ind,:]
    z = invert_GAN(output, e, method, act)
    return z

def test(k, e, method='linprog', act = 'ReLU'):
    flag = True
    sizes[0] = k
    weights = get_weights(sizes)
    while flag:
        z0 = np.random.normal(0, 1, sizes[0])
        if norm_type == 1:
            error = np.random.normal(0,e,sizes[-1])
        else:
            error = np.random.uniform(-e,e,sizes[-1])
        print(max(error), np.linalg.norm(error))
        output = inference(weights, bias, z0, error, act)
        nonzero = np.sum([i>0 for i in output])
        flag = False
    z = invert_GAN(output, e, method, act)
    init_error = np.linalg.norm(z0-z)
    return init_error/np.linalg.norm(z0),\
           np.linalg.norm(error),\
           np.linalg.norm(output) #inference(weights, np.zeros(sizes[-1]), current))

def invert_GAN(output, e=1e-6, method = 'linprog', act = 'ReLU'):
    # now we want to find out z0 from output
    thres = e
    current = copy.copy(output)
    for j in range(l-1):
        i = l-1-j
        if method == 'invert':
            current = method_invert(weights[i-1], bias[i-1], current, thres)
        else:
            if j==0:
                last_layer = True
            else:
                last_layer = False
            if method == 'linprog':
                method_ = method_linprog
            else:
                if method == 'relax':
                    method_ = method_lp_relax
                else:
                    exit('Error!! Invalid method')
            results = method_(weights[i-1], bias[i-1], current, thres, i==1, layers[i-1], last_layer, act)
            if method == 'linprog':
                flag, new_v = results[0], results[1]
            else:
                feasible, new_v, product = results[0], results[1], results[2]
                print('=============running', method_, '===============')
                flag = False
                nextvalue = infer_per_layer(weights[i-1], bias[i-1], new_v, act)
                olderror = np.linalg.norm(layers[i]-nextvalue,1)
            while not flag:
                thres *= theta
                if method == 'linprog':
                    flag, new_v = method_(weights[i-1], bias[i-1], current, thres, i==1, layers[i-1], last_layer, act)
                else:
                    old_v = new_v[:]
                    feasible, new_v, new_product = method_(weights[i-1], bias[i-1], current, thres, i==1, layers[i-1], last_layer, act)
                    #if feasible and new_product < product + thres * (1-1/theta) *len(current):
                    #flag = True
                    print(thres, 'old optimal:', product, 'new optimal:', new_product)
                    product = new_product
                    nextvalue = infer_per_layer(weights[i-1], bias[i-1], new_v, act)
                    nexterror = np.linalg.norm(layers[i]-nextvalue,1)
                    if nexterror >= olderror:
                        new_v = old_v
                        flag = True
                    olderror = nexterror
            current = new_v
            print('iter@ {0}, threshold:{1}'.format(j,thres))
        nextvalue = weights[i-1].dot(current) #res.x)
        nextvalue = np.maximum(nextvalue, np.zeros(len(nextvalue)))
        init_error = np.linalg.norm(layers[i-1]-current)#, np.inf)#,
        #print("previous error:", init_error, np.linalg.norm(layers[i-1]-current, np.inf))
        #print("next error:", np.linalg.norm(layers[i]-nextvalue, np.inf))
    return current



error_set = [1e-6] #[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
n_test = 1  #200
error= np.zeros([len(error_set), n_test])
noise = np.zeros([len(error_set), n_test])
outputnorm = np.zeros([len(error_set), n_test])


filename = './l'+str(norm_type) + '_error_'
if use_true_nets:
    filename+='real'
else:
    filename+='random'
for i in range(l):
    filename+='_'+str(sizes[i])
filename+='_'+activation

if activation == 'LeakyReLU':
    p = 0.5
else:
    p = 0.9

if task == 'sensing':
    m = int(784*p)
    # recover from observations of p portion of pixels
    ind = np.random.permutation(sizes[-1])[0:m]
    ind.sort()
    weights = get_weights(sizes)
    old_sizes=sizes[:]
    sizes[-1] = m
    z0 = np.random.normal(0, 1, sizes[0])
    e=1e-6
    error = 0#np.zeros(old_sizes[-1])
    output = inference(weights, bias, z0, error, activation)
    z = test_sensing(ind, output[ind], e)
    weights = get_weights(old_sizes)
    recovery = inference(weights, bias, z, np.zeros(old_sizes[-1]), activation)
    print(z0,z)
    plt.imshow(recovery.reshape(28,28))
    plt.imshow(output.reshape(28,28))

if task == 'invert' and method == 'relax':
    for j, err in enumerate(error_set):
        for i in range(n_test):
            error[j, i], noise[j,i], outputnorm[j,i] = test(sizes[0], err, 'relax', activation)
        np.savetxt('error_results_big_'+str(j), error[j,:], delimiter=',')
    np.savetxt(filename+'_relax_error', error,delimiter=',')
    np.savetxt(filename+'_relax_noise', noise,delimiter=',')
    np.savetxt(filename+'_relax_output', outputnorm,delimiter=',')

if task == 'invert' and method == 'linf':
    for j, err in enumerate(error_set):
        for i in range(n_test):
            error[j, i], noise[j,i], outputnorm[j,i] = test(sizes[0], err, 'linprog', activation)
    np.savetxt(filename+'_linf_error', error,delimiter=',')
    np.savetxt(filename+'_linf_noise', noise,delimiter=',')
    np.savetxt(filename+'_linf_output', outputnorm,delimiter=',')


if task == 'choose_k':
    error = np.zeros([25, n_test])
    times = np.zeros([25, n_test])
    import os
    directory = './success_results_'+activation
    if not os.path.exists(directory):
            os.makedirs(directory)
    for k in np.linspace(10,250,25):
        k = int(k)
        for j, err in enumerate([1e-9]):
            for i in range(n_test):
                st = time.time()
                error[k/10-1, i], _, _ = test(k, err, 'linprog',activation)
                times[k/10-1, i] = time.time() - st
        np.savetxt(directory+'/'+str(k), error[k/10-1,:], delimiter=',')
        np.savetxt(directory+'/'+str(k)+'_time', times[k/10-1,:], delimiter=',')



