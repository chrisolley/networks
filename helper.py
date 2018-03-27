# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import rc
from pylab import rcParams
from scipy import stats 

def figsize(scale):
    '''
    figsize: Scales figure output for use in latex typesetting. 
    Args: 
        scale: scale used for figure in latex document.
    Return: 
        figwidth, figheight: width and hight in inches required for this scale.
    '''
    
    fig_width_pt = 469.755 # assuming a4 page  
    inches_per_pt = 1.0/72.27 
    figwidth = fig_width_pt*inches_per_pt*scale
    figheight = figwidth # creates square figures
    
    return figwidth, figheight

def latexfigure(scale):
    '''
    latexfigure: Sets up matplotlib params for plotting for latex typesetting. 
    Args: 
        scale: scale used for figure in latex document.
    Returns: 
        None
    '''
    
    rc('font', **{'family':'sans-serif','sans-serif':['Computer Modern Sans serif'], 'size':9}) #
    rc('text', usetex=True)
    rcParams['figure.figsize'] = figsize(scale)[0], figsize(scale)[1]
    
def chisquared(O, E):
    
    cs = 0
    
    for i in range(len(O)):
        
        cs += (O[i]-E[i])**2/E[i]
    
    pval = 1 - stats.chi2.cdf(cs, len(O)-1)
    return cs, pval