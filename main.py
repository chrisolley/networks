# -*- coding: utf-8 -*-
from helper import latexfigure
from BA_analysis import BA_Analysis
import matplotlib.pyplot as plt
import numpy as np

#latexfigure(0.8)

ba_pref = BA_Analysis(100, 1.1, 'preferential')
ba_rand = BA_Analysis(100, 1.1, 'random')
ba_mixed = BA_Analysis(100, 1.1, 'mixed')

M = [3**i for i in range(0,5)]
#N = [int(10**i) for i in np.linspace(2,6,8)]
N = [10**i for i in range(2,6)]
#N = [5**i for i in range(3,6)]

ba_pref.plot_degree_distribution_M(10**4, M, 'read')
ba_pref.plot_degree_distribution_N(N, 3, 'read')


ba_rand.plot_degree_distribution_M(10**4, M, 'read')
ba_rand.plot_degree_distribution_N(N, 3, 'read')

ba_pref.plot_p_value_M(10**4, M)
ba_rand.plot_p_value_M(10**4, M)
ba_mixed.plot_p_value_M(10**4, M)

M = [3**i for i in range(0,5)]

ba_mixed.plot_degree_distribution_M(10**4, M, 'read')
ba_mixed.plot_degree_distribution_N(N, 3, 'read')

#ba = BA_Analysis(1000, 1.1, 'random')
#ba.plot_degree_distribution_M(10**4, M)
#
#ba = BA_Analysis(1000, 1.1, 'random')
#ba.plot_degree_distribution_N(N, 3)
#
#ba = BA_Analysis(1000, 1.1, 'mixed')
#ba.plot_degree_distribution_M(10**4, M)

plt.show()