# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
import os
import numpy as np
from model import GaussianProcess
from singlemes import MaxvalueEntropySearch
from scipy.optimize import minimize as scipyminimize
from platypus import NSGAII, Problem, Real
import sobol_seq


######################Algorithm input##############################
paths='.'
from benchmark_functions import branin,Currin
functions=[branin,Currin]
#this is just a toy example of how to define constraints, this means we want branin>=0 and Currin>=0
constraints=[branin,Currin] 
def evaluation(xx,d):
    global functions
    x=[item for item in xx]
    y=[functions[i](x,d) for i in range(len(functions))]
    c=[constraints[i](x,d) for i in range(len(constraints))]
    all_=y+c
    print(all_)
    return all_
functions_and_contraints=evaluation


M=len(functions)
C=len(constraints)
d=2
seed=1
np.random.seed(seed)
total_iterations=100
intial_number=5
sample_number=1
bound=[0,1]
Fun_bounds = [bound]*d
grid = sobol_seq.i4_sobol_generate(d,1000,np.random.randint(0,1000))

###################GP Initialisation##########################

GPs=[]
Multiplemes=[]
GPs_C=[]
Multiplemes_C=[]


for i in range(M):
    GPs.append(GaussianProcess(d))
for i in range(C):
    GPs_C.append(GaussianProcess(d))
#
for k in range(intial_number):
    exist=True
    while exist:
        design_index = np.random.randint(0, grid.shape[0])
        x_rand=list(grid[design_index : (design_index + 1), :][0])
        if (any((x_rand == x).all() for x in GPs[0].xValues))==False:
            exist=False
    functions_contraints_values=functions_and_contraints(x_rand,d)
    for i in range(M):
        GPs[i].addSample(np.asarray(x_rand),functions_contraints_values[i])
    for i in range(C):
        GPs_C[i].addSample(np.asarray(x_rand),functions_contraints_values[i+M])
    with  open(os.path.join(paths,'Inputs.txt'), "a") as filehandle:  
        for item in x_rand:
            filehandle.write('%f ' % item)
        filehandle.write('\n')
    filehandle.close()
    with  open(os.path.join(paths,'Outputs.txt'), "a") as filehandle:  
        for listitem in functions_contraints_values:
            filehandle.write('%f ' % listitem)
        filehandle.write('\n')
    filehandle.close()
        
    
for i in range(M):   
    GPs[i].fitModel()
    Multiplemes.append(MaxvalueEntropySearch(GPs[i]))
for i in range(C):   
    GPs_C[i].fitModel()
    Multiplemes_C.append(MaxvalueEntropySearch(GPs_C[i]))
    

for l in range(total_iterations):
    for i in range(M):
        Multiplemes[i]=MaxvalueEntropySearch(GPs[i])
        Multiplemes[i].Sampling_RFM()
    for i in range(C):
        Multiplemes_C[i]=MaxvalueEntropySearch(GPs_C[i])
        Multiplemes_C[i].Sampling_RFM()
    max_samples=[]
    max_samples_constraints=[]
    for j in range(sample_number):
        for i in range(M):
            Multiplemes[i].weigh_sampling()
        for i in range(C):
            Multiplemes_C[i].weigh_sampling()
        cheap_pareto_front=[]
        def CMO(xi):
            xi=np.asarray(xi)
            y=[Multiplemes[i].f_regression(xi)[0][0] for i in range(len(GPs))]
            y_c=[Multiplemes_C[i].f_regression(xi)[0][0] for i in range(len(GPs_C))]
            return y,y_c
        
        problem = Problem(d, M,C)
        problem.types[:] = Real(bound[0], bound[1])

        problem.constraints[:]=[">=0" for i in range(C)]
        problem.function = CMO
        algorithm = NSGAII(problem)
        algorithm.run(1500)

        cheap_pareto_front=[list(solution.objectives) for solution in algorithm.result]
        cheap_constraints_values=[list(solution.constraints) for solution in algorithm.result]
        
        maxoffunctions=[-1*min(f) for f in list(zip(*cheap_pareto_front))]#this is picking the max over the pareto: best case
        maxofconstraints=[-1*min(f) for f in list(zip(*cheap_constraints_values))]
        max_samples.append(maxoffunctions)
        max_samples_constraints.append(maxofconstraints)
    def mesmo_acq(x):
        
        if np.prod([GPs_C[i].getmeanPrediction(x)>=0 for i in range(len(GPs_C))]):
            multi_obj_acq_total=0
            for j in range(sample_number):
                multi_obj_acq_sample=0
                for i in range(M):
                    multi_obj_acq_sample=multi_obj_acq_sample+Multiplemes[i].single_acq(np.asarray(x),max_samples[j][i])
                for i in range(C):
                    multi_obj_acq_sample=multi_obj_acq_sample+Multiplemes_C[i].single_acq(np.asarray(x),max_samples_constraints[j][i])
                multi_obj_acq_total=multi_obj_acq_total+multi_obj_acq_sample
            return (multi_obj_acq_total/sample_number)
        else:
            return 10e10


        
    # l-bfgs-b acquisation optimization
    x_tries = np.random.uniform(bound[0], bound[1],size=(1000, d))
    y_tries=[mesmo_acq(x) for x in x_tries]
    sorted_indecies=np.argsort(y_tries)
    i=0
    x_best=x_tries[sorted_indecies[i]]
    while (any((x_best == x).all() for x in GPs[0].xValues)):
        print(x_best)
        print(GPs[0].xValues)
        i=i+1
        x_best=x_tries[sorted_indecies[i]]
    y_best=y_tries[sorted_indecies[i]]
    x_seed=list(np.random.uniform(low=bound[0], high=bound[1], size=(100,d)))    
    for x_try in x_seed:
        result = scipyminimize(mesmo_acq, x0=np.asarray(x_try).reshape(1, -1), method='L-BFGS-B', bounds=Fun_bounds)
        if not result.success:
            continue
        if ((result.fun<=y_best) and (not (result.x in np.asarray(GPs[0].xValues)))):
            x_best=result.x
            y_best=result.fun
            
        

#---------------Updating and fitting the GPs-----------------  
    functions_contraints_values=functions_and_contraints(x_best,d)
    for i in range(M): 
        GPs[i].addSample(np.asarray(x_best),functions_contraints_values[i])
        GPs[i].fitModel()
    for i in range(C): 

        GPs_C[i].addSample(x_best,functions_contraints_values[M+i])
        GPs_C[i].fitModel()
    with  open(os.path.join(paths,'Inputs.txt'), "a") as filehandle:  
        for item in x_best:
            filehandle.write('%f ' % item)
        filehandle.write('\n')
    filehandle.close()
    with  open(os.path.join(paths,'Outputs.txt'), "a") as filehandle:  
        for listitem in functions_contraints_values:
            filehandle.write('%f ' % listitem)
        filehandle.write('\n')
    filehandle.close()

