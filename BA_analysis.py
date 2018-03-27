# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from BA_model_edit import BA_Model_Multi
from helper import chisquared
from scipy import stats
from scipy import special
from logbin2018 import logbin

class BA_Analysis:
    
    '''
    BA_Analysis: class for analysing a BA_Model(_Multi) class output.
    Important methods:
        plot_degree_distribution: plots the degree distribution for a single network
        configuration. 
        theoretical_k1_x: theoretical k1 values. 
        theoretical_x: theoretical degree distributions.
        plot_degree_distribution_M: plots the degree distribution for several M
        values, fixed N. 
        plot_degree_distribution_N: plots the degree distribution for several N
        values, fixed M. 
        plot_p_value_M: plots the p value variation for amount of distribution 
        included, for several M.
        
    Args: 
        a: number of repeats.
        scale: log-binning scale.
        mode: preferential, random or mixed.
        q: mixed attachment parameter (default 0.5).
    '''
    
    
    def __init__(self, a, scale, mode, q=0.5): 
        
        self.a = a
        self.scale = scale
        self.mode = mode
        self.q = q
        
    def plot_degree_distribution(self, N, M):
        
        ba_multi = BA_Model_Multi(a=self.a, scale=self.scale, N=N, M=M, mode=self.mode)
    
        self.log_binned, self.log_binned_theory, self.pdf_theory, self.k1, self.degree_distribution, self.pdf_theory_exact = ba_multi.run()
        self.k1_max = self.k1[2]
        self.prob = self.log_binned[1] / ba_multi.N # log binned pdf from data
        self.prob_theory = self.pdf_theory / ba_multi.N # pdf from theory
        self.prob_theory_binned = self.log_binned_theory[1] / ba_multi.N # log binned pdf (averages) from theory
        cutoff_index = np.where(self.log_binned[2]<=3*ba_multi.M)[0][0] # remove tail of distribution
        
        # perform chi squared test on the summed log bins from data and theory
#        cs, pval = chisquared(self.log_binned[2][:], self.log_binned_theory[2][:])
#        cs_check, p_check = stats.chisquare(self.log_binned[2][:], self.log_binned_theory[2][:])
#        
#        print('Chi-Squared Test Statistic: {}, P-Value: {}'.format(cs, pval))
#        print('Chi-Squared Test Statistic (Check): {}, P-Value: {}'.format(cs_check, p_check))
#   
        
        fig, ax = plt.subplots()
        ax.loglog(self.log_binned[0], self.log_binned[2], label='Log binned')
        ax.loglog(self.log_binned_theory[0], self.log_binned_theory[2], color='black', linestyle='--', label='Theoretical')
        ax.axvline(self.log_binned[0][cutoff_index], color='red')
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$n(k)$')
        ax.legend(loc='best')
        
        fig1, ax1 = plt.subplots()
        ax1.loglog(self.log_binned[0], self.prob, label='Log binned')
        ax1.loglog(range(ba_multi.M, self.k1_max+1), self.prob_theory[ba_multi.M:], color='black', linestyle='--', label='Theoretical')
        #ax1.loglog(self.degree_distribution[0], self.prob_theory, color='black', linestyle='--', label='Theoretical')
        ax1.set_xlabel(r'$k$')
        ax1.set_ylabel(r'$p(k)$')
        ax1.legend(loc='best')
        
        fig2, ax2 = plt.subplots()  
        ax2.scatter(self.degree_distribution[0], self.degree_distribution[1], s=5, label='Unbinned data')
        ax2.plot(range(ba_multi.M, self.k1_max+1), self.a*self.pdf_theory[ba_multi.M:], color='black', linestyle='--', label='Theoretical')
        print(len(self.degree_distribution[0]))
        print(len(self.pdf_theory_exact))
        ks, pval_ks = stats.ks_2samp(self.degree_distribution[1]/(ba_multi.N*self.a), self.pdf_theory_exact/ba_multi.N)
        print('K-S Test Statistic: {}, P-Value: {}'.format(ks, pval_ks))
        ax2.set_xlabel(r'$k$')
        ax2.set_ylabel(r'$n(k)$')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend(loc='best')
        
        fig3, ax3 = plt.subplots()  
        ax3.scatter(self.degree_distribution[0], self.degree_distribution[1]/(ba_multi.N*self.a), s=5, label='Unbinned data')
        ax3.plot(self.degree_distribution[0], self.pdf_theory_exact/ba_multi.N, color='black', linestyle='--', label='Theoretical')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
    def theoretical_k1(self, m, N): 
        
        return 0.5 * (-1. + np.sqrt(1 + 4 * m * (m + 1) * N))
    
    def theoretical_k1_rand(self, m, N):
        
        return m - (np.log(N))/(np.log(m) - np.log(1 + m))
    
    def theoretical_k1_mixed(self, m, N):
        
        return 0.5 * (- 3. - 4. * m + np.sqrt(5. + 4. * np.sqrt(1. + 9. * m * (1. + m ) * (1. + 3. * m) * (2. + 3. * m) * N)))
    
    def theoretical_pref(self, k, m):
        
        return (2. * m * (m + 1.)) / (np.float64(k) * (np.float64(k) + 1.) * (np.float64(k) + 2.))
    
    def theoretical_rand(self, k, m):
        
        return (m**(np.float64(k) - m)) / ((1. + m)**(np.float64(k) - m + 1.))
    
    def theoretical_mixed(self, k, m, q=0.5):
#            
#        p1 = special.gamma(m + 1. + ((2. * m * (1. - q)) / q) + (2. / q))
#        p2 = special.gamma(k + ((2. * m * (1. - q)) / q))
#        p3 = special.gamma(m + ((2. * m * (1. - q)) / q)) * (1. + ((q * m) / 2.) + m * (1. - q))
#        p4 = special.gamma(k + 1. + ((2. * m * (1. - q)) / q) + (2. / q))
#        
        p1 = 12 * m * (3*m+3) * (3*m+2) * (3*m+1)
        p2 = (k+2*m+4) * (k+2*m+3) * (k+2*m+2) * (k+2*m+1) * (k+2*m)
        
        return p1 / p2
    
    def plot_degree_distribution_M(self, N, M, data_mode): 
        
        # data mode: load = load saved data, save: run model and save runs
        
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()

        for m in M: 
            ba_multi = BA_Model_Multi(a=self.a, scale=self.scale, N=N, M=m, mode=self.mode)
            if data_mode=='save':
                log_binned, log_binned_theory, pdf_theory, k, degree_distribution, degree_distribution_exact = ba_multi.run()
                k_max = k[2]
                np.save('data/log_binned_M_{}_N_{}_R_{}_{}'.format(m, N, self.a, self.mode), log_binned)
                np.save('data/log_binned_theory_M_{}_N_{}_R_{}_{}'.format(m, N, self.a, self.mode), log_binned_theory)
                np.save('data/pdf_theory_M_{}_N_{}_R_{}_{}'.format(m, N, self.a, self.mode), pdf_theory)
                np.save('data/k_M_{}_N_{}_R_{}_{}'.format(m, N, self.a, self.mode), k)
                np.save('data/degree_distribution_M_{}_N_{}_R_{}_{}'.format(m, N, self.a, self.mode), degree_distribution)
            
            if data_mode=='read':
                log_binned = np.load('data/log_binned_M_{}_N_{}_R_{}_{}.npy'.format(m, N, self.a, self.mode))
                log_binned_theory = np.load('data/log_binned_theory_M_{}_N_{}_R_{}_{}.npy'.format(m, N, self.a, self.mode))
                pdf_theory = np.load('data/pdf_theory_M_{}_N_{}_R_{}_{}.npy'.format(m, N, self.a, self.mode))
                k = np.load('data/k_M_{}_N_{}_R_{}_{}.npy'.format(m, N, self.a, self.mode))
                k_max = (k[2]).astype(int)
                degree_distribution = np.load('data/degree_distribution_M_{}_N_{}_R_{}_{}.npy'.format(m, N, self.a, self.mode))
            
           # print("Average largest k value: {} +- {}, for m: {}".format(k[0], k[1], m))
            prob = log_binned[1]/ba_multi.N
            prob_theory = pdf_theory/ba_multi.N

            cutoff_index = np.where(log_binned[2]<=3*ba_multi.M)[0][0]
            cs, pval = chisquared(log_binned[2][:], log_binned_theory[2][:])
            cs_check, p_check = stats.chisquare(log_binned[2][:cutoff_index], log_binned_theory[2][:cutoff_index])

            print('Chi-Squared Test Statistic for m: {}: {}, P-Value: {}'.format(m, cs, pval))
            print('Chi-Squared Test Statistic for m: {} (Check): {}, P-Value: {}'.format(m, cs_check, p_check))
            ks, pval_ks = stats.ks_2samp(log_binned[1][:], log_binned_theory[1][:])
            print('K-S Test Statistic for m: {}, {}, P-Value: {}'.format(m, ks, pval_ks))
            
            # plotting theoretical pdfs and data pdfs for different m values
            ax.loglog(log_binned[0], prob, label='m={}'.format(m), marker='.', linestyle='')
            ax.loglog(range(ba_multi.M, k_max+1), prob_theory[ba_multi.M:], color='black', linestyle='--')
            
            # plotting average counts & theoretical counts for different m values
            ax1.plot(degree_distribution[0], degree_distribution[1] / self.a, marker='.', markersize=5, label='m={}'.format(m), linestyle='')
            ax1.plot(range(m, k_max+1), pdf_theory[m:], color='black', linestyle='--')
        
        ax.legend(loc='best')
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$p(k)$')
        
        ax1.legend(loc='best')
        ax1.set_xlabel(r'$k$')
        ax1.set_ylabel(r'$n(k)$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        if self.mode=='preferential':
            fig.savefig('degree_dist_pref_M_R_{}.pdf'.format(self.a))
        if self.mode=='random':
            fig.savefig('degree_dist_rand_M_R_{}.pdf'.format(self.a))
        if self.mode=='mixed':
            fig.savefig('degree_dist_mixed_M_R_{}.pdf'.format(self.a))
            
    def plot_degree_distribution_N(self, N, M, data_mode):
        
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        # calculate theoretical largest expected degree and plot
        if self.mode=='preferential':
            k1_th = [self.theoretical_k1(M, i) for i in range(min(N), max(N))]
            ax.plot(range(min(N), max(N)), k1_th, color='red', linestyle='--', label='Theoretical')
        if self.mode=='random':
            k1_th = [self.theoretical_k1_rand(M, i) for i in range(min(N), max(N))]
            ax.plot(range(min(N), max(N)), k1_th, color='red', linestyle='--', label='Theoretical')
        if self.mode=='mixed':
            k1_th = [self.theoretical_k1_mixed(M, i) for i in range(min(N), max(N))]
            ax.plot(range(min(N), max(N)), k1_th, color='red', linestyle='--', label='Theoretical')
            
        k1_values = []
        for n in N:
            ba_multi = BA_Model_Multi(a=self.a, scale=self.scale, N=n, M=M, mode=self.mode)
            if data_mode=='save':
                log_binned, log_binned_theory, pdf_theory, k, degree_distribution, degree_distribution_exact = ba_multi.run()
                k_max = k[2]
                np.save('data/log_binned_M_{}_N_{}_R_{}_{}'.format(M, n, self.a, self.mode), log_binned)
                np.save('data/log_binned_theory_M_{}_N_{}_R_{}_{}'.format(M, n, self.a, self.mode), log_binned_theory)
                np.save('data/pdf_theory_M_{}_N_{}_R_{}_{}'.format(M, n, self.a, self.mode), pdf_theory)
                np.save('data/k_M_{}_N_{}_R_{}_{}'.format(M, n, self.a, self.mode), k)
                np.save('data/degree_distribution_M_{}_N_{}_R_{}_{}'.format(M, n, self.a, self.mode), degree_distribution)
            
            if data_mode=='read':
                log_binned = np.load('data/log_binned_M_{}_N_{}_R_{}_{}.npy'.format(M, n, self.a, self.mode))
                log_binned_theory = np.load('data/log_binned_theory_M_{}_N_{}_R_{}_{}.npy'.format(M, n, self.a, self.mode))
                pdf_theory = np.load('data/pdf_theory_M_{}_N_{}_R_{}_{}.npy'.format(M, n, self.a, self.mode))
                k = np.load('data/k_M_{}_N_{}_R_{}_{}.npy'.format(M, n, self.a, self.mode))
                k_max = (k[2]).astype(int)
                degree_distribution = np.load('data/degree_distribution_M_{}_N_{}_R_{}_{}.npy'.format(M, n, self.a, self.mode))
                
            prob = log_binned[1]/n
            prob_theory = pdf_theory/n
            if self.mode=='preferential':
                print("Theoretical k_1: {}".format(self.theoretical_k1(M, n)))
            if self.mode=='random':
                print("Theoretical k_1: {}".format(self.theoretical_k1_rand(M, n)))

            prob_y_scaled = []
            prob_x_scaled = []
            # data collapse of degree distribution for different N values
            for i, x_i in enumerate(log_binned[0]):
                prob_x_scaled.append(x_i/k[2]) # scale x by largest numerical k
            for i, y_i in enumerate(prob):
                # scale y by dividing by the theoretical value
                if self.mode=='preferential':
                    prob_y_scaled.append(y_i/self.theoretical_pref(log_binned[0][i], M))
                if self.mode=='random':
                    prob_y_scaled.append(y_i/self.theoretical_rand(log_binned[0][i], M))   
                if self.mode=='mixed':
                    prob_y_scaled.append(y_i/self.theoretical_mixed(log_binned[0][i], M))                       
                ax.errorbar(n, k[0], yerr=k[1], marker='.', ms=3, capsize=3, elinewidth=1, color='black')
            k1_values.append(k[0])
            # plot the numerical distributions for different N values
            ax1.loglog(log_binned[0], prob, label='N={}'.format(n), marker='.', linestyle='', markersize=5)
            # plot the data collapse of these distributions
            ax2.loglog(prob_x_scaled, prob_y_scaled, label='N={}'.format(n), marker='.', linestyle='', markersize=5)
            
        
        ax1.loglog(range(M, k_max+1), prob_theory[M:], color='black', linewidth=1)    
        ax1.legend(loc='best')
        ax1.set_ylabel(r'$p(k)$')
        ax1.set_xlabel(r'$k$')
    
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$k_1$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if self.mode == 'preferential' or 'mixed':
            print(np.log(N))
            print(np.log(k1_values))
            linregsoln = stats.linregress(np.log(N[:]), np.log(k1_values[:]))

            slope, intercept, r_value, p_value, std_err = linregsoln
            print('Linear regression exponent value: m={} +- {}'.format(slope, std_err))
            ax.plot(N, [np.exp(slope*np.log(i)+intercept) for i in N], color='black')
            
        if self.mode=='preferential':
            fig1.savefig('degree_dist_pref_N.pdf')
        if self.mode=='random':
            fig1.savefig('degree_dist_rand_N.pdf')
        if self.mode=='mixed':
            fig1.savefig('degree_dist_mixed_N.pdf')
        
        ax2.legend(loc='best')
        ax2.set_xlabel(r'$k/k_1$')
        ax2.set_ylabel(r'$p_{data}(k)/p_{theory}(k)$')
        
    def plot_p_value_M(self, N, M):
        
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        
        for m in M:
            ba_multi = BA_Model_Multi(a=self.a, scale=self.scale, N=N, M=m, mode=self.mode)
            log_binned, log_binned_theory, pdf_theory, k, degree_distribution, pdf_theory_exact = ba_multi.run()
            #prob = log_binned[1]/ba_multi.N
            #prob_theory = pdf_theory/ba_multi.N
            
            D_values = []
            ks_p_values = []
            
            for i in range(1,len(degree_distribution[0])):
                #cutoff_index = np.where(log_binned[2]<=i)[0][0]
                #cs_check, p_check = stats.chisquare(log_binned[2][:i], log_binned_theory[2][:i])
                #cs_p_values.append(p_check)
                #print('Chi-Squared Test Statistic for m: {} (Check): {}, P-Value: {}'.format(m, cs_check, p_check))
                D, pval_ks = stats.ks_2samp(degree_distribution[1][:i]/(N*self.a), pdf_theory_exact[:i]/N)
                ks_p_values.append(pval_ks)
                D_values.append(D)
                #print('K-S Test Statistic for m: {}, {}, P-Value: {}'.format(m, ks, pval_ks))
                
            ax.plot(degree_distribution[0][1:], D_values, label='m={}'.format(m))
            ax1.plot(degree_distribution[0][1:], ks_p_values, label='m={}'.format(m))
        ax.legend(loc='best')
        ax1.legend(loc='best')
        
if __name__ == '__main__':
    ba = BA_Analysis(100, 1.15, 'preferential')
    #ba.plot_degree_distribution(10**4, 3)
    #ba = BA_Analysis(50, 1.3, 'random')
    #ba.plot_degree_distribution(10**4, 4)
    #ba = BA_Analysis(50, 1.3, 'mixed')
    #ba.plot_degree_distribution(10**4, 4)
    M = [3**i for i in range(0,3)]
    #N = [int(10**i) for i in np.linspace(2,5,7)]
    #N = [5**i for i in range(3,8)]
    #N = [10**i for i in range(2, 6)]
    ba.plot_degree_distribution_M(10**4, M, 'save')
    #ba.plot_degree_distribution_N(N, 3)
    #ba.plot_p_value_M(10**4, M)
    plt.show()