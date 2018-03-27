# -*- coding: utf-8 -*-
''''
Code based on tim evan's example ersimpledegreedistribution.py code provided 
for the C&N 2018 course at imperial college london.
''''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from logbin2018_edit import logbin
from scipy import special
from scipy import stats
import os

class BA_Model:
    
    '''
    BA_Model: class for creating a BA model simulation, using preferential, random
    or mixed attachment. 
    Important methods: 
        grow_preferential: grows the model for one time step using preferential 
        attachment. 
        grow_random: grows the model for one time step using random attachment. 
        grow_mixed: grows the model for one time step using mixed attachment. 
        iterate: iterate the model over a number of time steps. 
        degree_distribution: calculates degree distibution of the network.
        plot_degree_distibution: plots degree distibution. 
        visualise: plot network using NetworkX.
        
    Args: 
        n: size of initial network. 
        M: number of vertices added each iteration. 
        mode: preferential, random or mixed. 
        q: mixed attachment parameter (default 0.5).
    '''
    
    def __init__(self, n, M, mode, q=0.5):
        
        self.n = n # size of initial network
        self.M = M # number of vertices to add each iteration
        self.mode = mode
        self.q = q
        
        self.vertex_list = []
        self.vertex_rand_list = []
        self.G = nx.Graph()
        self.printon = False
        
        # adds required number of nodes for initial graph with no edges
        for i in range(self.n):
            self.G.add_node(i)
            self.vertex_rand_list.append(i)
        
        # adds edges creating a complete graph
        for s in range(self.n):
            for t in range(s + 1, self.n):
                self.G.add_edge(s, t)    
                self.vertex_list.append(s)
                self.vertex_list.append(t)
                if self.printon:
                    print('--- new edge from', s,' to ', t) # useful for debugging
                        
    def grow_preferential(self): 
                
        new_vertex = self.G.number_of_nodes() # use this vertex for the rest of method
        self.G.add_node(self.G.number_of_nodes()) # adds additional vertex
        self.vertex_rand_list.append(new_vertex)

        target_vertex = set() # set so only unique elements
        
        while len(target_vertex) < self.M: # creates M unique elements from the existing vertices
            x = random.choice(self.vertex_list)
            target_vertex.add(x)
                
        for t in target_vertex:
            self.G.add_edge(new_vertex, t)
            self.vertex_list.append(t)
            self.vertex_list.append(new_vertex)
            if self.printon:
                print('--- new edge from', new_vertex, 'to', t) 
    
    def grow_random(self): 
        
        new_vertex = self.G.number_of_nodes() # use this vertex for the rest of method
        self.G.add_node(self.G.number_of_nodes()) # adds additional vertex
        self.vertex_rand_list.append(new_vertex)
        
        target_vertex = set() # set so only unique elements
        
        while len(target_vertex) < self.M: # creates M unique elements from the existing vertices
            x = random.choice(self.vertex_rand_list)
            target_vertex.add(x)
                
        for t in target_vertex:
            self.G.add_edge(new_vertex, t)
            self.vertex_list.append(t)
            self.vertex_list.append(new_vertex)
            if self.printon:
                print('--- new edge from', new_vertex, 'to', t) 
                
    def grow_mixed(self):
        
        new_vertex = self.G.number_of_nodes() # use this vertex for the rest of method
        self.G.add_node(self.G.number_of_nodes()) # adds additional vertex
        self.vertex_rand_list.append(new_vertex)
        
        target_vertex = set() # set so only unique elements
        
        while len(target_vertex) < self.M: # creates M unique elements from the existing vertices
            
            if np.random.choice([0, 1], p=[self.q, 1-self.q]):
                x = random.choice(self.vertex_list)
                target_vertex.add(x)
            else:
                x = random.choice(self.vertex_rand_list)
                target_vertex.add(x)
        
        for t in target_vertex:
            self.G.add_edge(new_vertex, t)
            self.vertex_list.append(t)
            self.vertex_list.append(new_vertex)
            if self.printon:
                print('--- new edge from', new_vertex, 'to', t) 
            
    def iterate(self, N):
        
        self.N = N

        while (self.G.number_of_nodes() < self.N):        
            if self.mode=='preferential':
                self.grow_preferential()
    
            if self.mode=='random':
                self.grow_random()
             
            if self.mode=='mixed':
                self.grow_mixed()
                
    def degree_distribution(self):
        
        degree_distribution_list = [self.G.degree(a) for a in self.G.nodes()]
        degree_distribution_list = np.array(degree_distribution_list)
        degree, counts = np.unique(degree_distribution_list, return_counts=True)
        prob = counts/len(degree_distribution_list)

        return degree_distribution_list, degree, counts, prob
    
    def plot_degree_distribution(self):
        
        self.degree_distribution_list, self.degree, self.counts, self.prob = self.degree_distribution()
        
        self.log_binned = logbin(self.degree_distribution_list, 1.4, zeros=True)
        
        fig, ax = plt.subplots()
        ax.loglog(self.degree, self.prob, marker='.', linestyle='', label='Unbinned')
        ax.loglog(self.log_binned[0], self.log_binned[1], marker='+', linestyle='', label='Log binned')
        ax.loglog(self.degree, 2*self.M**2*self.degree**(-3.), linestyle='--', color='red', label='2m^2k^-3')
        ax.grid()
        ax.set_xlabel('Degree of Node, k')
        ax.set_ylabel('Probability')
        ax.legend(loc='best')
    
    def visualise(self):
        
        fig, ax = plt.subplots()
        nx.draw_networkx(self.G, withlabels=True, ax=ax)
        ax.axis('off')
            
class BA_Model_Multi:
    
    '''
    BA_Model_Multi: class for creating a BA model simulation, using preferential, random
    or mixed attachment, averaged over a number of repeats.
    Important methods: 
        run: iterate the model over a number of time steps, and repeat a number of times. 
        degree_distribution: calculates degree distibution of the network.
        theoretical_degree_distribution_x: analytic solution for a given attachment mode.
        
    Args:
        a: number of repeats
        scale: log binning scale
        M: number of vertices added each iteration. 
        mode: preferential, random or mixed. 
        q: mixed attachment parameter (default 0.5).
        
    Returns: 
        log_binned: log binned degree distribution.
        pdf_theory_log_binned: analytic solution also log binned.
        pdf_theory: unbinned analytic solution. 
        k: tuple, mean, max and s.d. on maximum degree k1.
        degree_distribution: raw degree distribution data.
        pdf_theory_exact: analytic degree distribution evaluated only for degrees in the numerical data.
        
    '''
    
    
    def __init__(self, a, scale, N, M, mode, q=0.5):
        
        self.a = a
        self.scale = scale
        self.N = N
        self.M = M
        self.mode = mode
        self.q = q
    
    def theoretical_degree_distribution_pref(self, degree_list, m):
        
        p_inf = np.zeros(len(degree_list), dtype=np.float64)
        
        for i, k in enumerate(degree_list):
            p_inf[i] = (2. * m * (m + 1.)) / (np.float64(k) * (np.float64(k) + 1.) * (np.float64(k) + 2.))
        
        return p_inf
    
    def theoretical_degree_distribution_rand(self, degree_list, m):
        
        p_inf = np.zeros(len(degree_list), dtype=np.float64)
        m1 = np.float64(m)
        for i, k in enumerate(degree_list):
            k1 = np.float64(k)
            #p_inf[i] = (m1**(k1 - m1)) / ((1. + m1)**(k1 - m1 + 1.))
            p_inf[i] =  (1./(m1+1)) * (m1/(1.+m1))**(k1-m1)
        return p_inf
    
    def theoretical_degree_distribution_mixed(self, degree_list, m, q):
        
        p_inf = np.zeros(len(degree_list), dtype=np.float64)
        
        for i, k in enumerate(degree_list):
#            p1 = special.gamma(m + 1. + ((2. * m * (1. - q)) / q) + (2. / q))
#            p2 = special.gamma(np.float64(k) + ((2. * m * (1. - q)) / q))
#            p3 = special.gamma(m + ((2. * m * (1. - q)) / q)) * (1. + ((q * m) / 2.) + m * (1. - q))
#            p4 = special.gamma(np.float64(k) + 1. + ((2. * m * (1. - q)) / q) + (2. / q))
#            p_inf[i] = (p1 * p2) / (p3 * p4)
            #commented out code is gneeral, below code is only for q=0.5
            p1 = 12 * m * (3*m+3) * (3*m+2) * (3*m+1)
            p2 = (k+2*m+4) * (k+2*m+3) * (k+2*m+2) * (k+2*m+1) * (k+2*m)
            p_inf[i] = p1/p2
        
        return p_inf
    
    def run(self):
        
        degree_distribution_list_multi = np.zeros(self.a*self.N, dtype=np.int64)
        k1_list = [] # list to hold numerical largest k values
        
        for i in range(self.a): # run ba model a times
            ba = BA_Model(n=self.M + 1, M=self.M, mode=self.mode, q=self.q)
            ba.iterate(self.N)
            degree_distribution_list, degree, counts, prob = ba.degree_distribution()
            k1 = max(degree) # find largest numerical k value each run
            k1_list.append(k1)
            for j in range(self.N):
                #store the degrees found in each run
                degree_distribution_list_multi[i*self.N+j] = degree_distribution_list[j]
        
        print("Average degree for m: {} is k={}".format(self.M, np.mean(degree_distribution_list_multi)))
        print("Mode degree for m: {} is k={}".format(self.M, stats.mode(degree_distribution_list_multi)))
        # return the unique degree values and counts for the a runs
        degree_distribution = np.unique(degree_distribution_list_multi, return_counts=True)
        
        degree_distribution_list_multi.sort()
        # find frequencies for every degree up to max degree
        degree_distribution_counts = np.bincount(degree_distribution_list_multi)
        k1_max = max(k1_list) # find the largest degree found over a runs
        # log bin the counts for every degree up to max degree
        log_binned = logbin(degree_distribution_counts, k1_max, self.scale, zeros=True)
        log_binned_x = log_binned[0] # geometric mean of the log binned x scale
        # average the log binned y values over the number of runs
        log_binned_y = np.array([i / self.a for i in log_binned[1]])
        log_binned_y_sum = np.array([i / self.a for i in log_binned[2]])
        # store log binned data in a tuple
        log_binned_data = (log_binned_x, log_binned_y, log_binned_y_sum)
        # average and sd on the largest k value
        k1_mean = np.mean(k1_list)
        k1_std = np.std(k1_list)
        # remove the values < M from the log binned data 
        firstindex = np.where(log_binned_data[0]>self.M)[0][0]
        log_binned = [log_binned_data[i][firstindex:] for i in range(0,3)]
        # theoretical distributions
        # create, convert to frequencies and then log bin for chi squared comparison
        if self.mode=='preferential':    
            pdf_theory = self.N*self.theoretical_degree_distribution_pref(range(k1_max+1), self.M)
            pdf_theory_exact = self.N*self.theoretical_degree_distribution_pref(degree_distribution[0], self.M)
            pdf_theory_log_binned = logbin(pdf_theory, k1_max, self.scale, zeros=True) 
        if self.mode == 'random': 
            pdf_theory = self.N*self.theoretical_degree_distribution_rand(range(k1_max+1), self.M)
            pdf_theory_exact = self.N*self.theoretical_degree_distribution_rand(degree_distribution[0], self.M)
            pdf_theory_log_binned = logbin(pdf_theory, k1_max, self.scale, zeros=True) 
            
        if self.mode=='mixed':
            pdf_theory = self.N*self.theoretical_degree_distribution_mixed(range(k1_max+1), self.M, self.q)
            pdf_theory_exact = self.N*self.theoretical_degree_distribution_mixed(degree_distribution[0], self.M, self.q)
            pdf_theory_log_binned = logbin(pdf_theory, k1_max, self.scale, zeros=True) 
        # remove the same values from the theoretical log binned distributions
        pdf_theory_log_binned = [pdf_theory_log_binned[i][firstindex:] for i in range(0,3)]
        
        return log_binned, pdf_theory_log_binned, pdf_theory, (k1_mean, k1_std, k1_max), degree_distribution, pdf_theory_exact

#ba = BA_Model(n=4, M=3, mode='random')
#ba.iterate(10**5)
#ba.plot_degree_distribution()
#print(sum(ba.prob))    

#ba_multi1 = BA_Model_Multi(a=1, scale=1.3, N=10**4, M=16, mode='preferential')
#ba_multi1.run()
#ba_multi2 = BA_Model_Multi(a=20, N=10**5, M=4)
#ba_multi3 = BA_Model_Multi(a=20, N=10**5, M=5)
#ba_multi1.plot_degree_distribution()
#ba_multi2.plot_degree_distribution()
#ba_multi3.plot_degree_distribution()
#print(sum(ba_multi.prob_list))

#plt.show()
