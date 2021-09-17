# -*- coding: utf-8 -*-
"""
Title:
Concept:
Created on Thu Sep  9 08:32:04 2021

@author: Andrew Nelson
"""
#% Import Libraries
import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, 'C:\\Users\\AJN Lab\\Documents\\GitHub\\Python-Master\\Hurricane\\')
sys.path.insert(1, 'C:\\Users\\andre\\Documents\\GitHub\\Python-Master\\Hurricane\\')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#% Functions

def correlate(x,y):
    c = x*0
    spacing = y.size
    if spacing % 2 is 0:
        shift = np.int16(spacing / 2)
    else:
        shift = np.int16((spacing  +1)/2)
    # Add padding
    a = x
    a = np.append(np.zeros(spacing - 1), x)
    a = np.append(a,np.zeros(spacing - 1))
    y = y - y.mean()
    for i in range(x.size):
        sub_a = a[i + shift+1:i+spacing + shift+1]
        logos = sub_a != 0
        sub_a = sub_a - sub_a.mean()
        sub_a = sub_a/sub_a.max()
        sub_a = (sub_a*logos)
        
        c[i] = np.sum(np.dot(sub_a,y))
    
    return c

#%
if __name__ == '__main__':
    for file_name in os.listdir():
        if file_name.endswith('xlsx'):
            xlsx_file = file_name

    db = pd.read_excel(xlsx_file)

    frames = np.array(db['Unnamed: 0'][1:])
    silent = np.array(db['ROIs'][1:])
    ambiguous = np.array(db['Unnamed: 2'][1:])
    definite = np.array(db['Unnamed: 3'][1:])
    ensemble = np.array(db['Ensemble'][1:])
    kernel = (ensemble - ensemble[:40].mean())/(max(ensemble) - ensemble[:40].mean())
    dsilent = silent - silent[:40].mean()
    dambiguous = ambiguous - ambiguous[:40].mean()
    ddefinite = definite - definite[:40].mean()
    # Going under the assumption that [t-1 t0 t+1] w/ t0 = stim frame
    # Are the important data points, we can define a kernel to be [1/3 1/3 1/3]
    # The convolution that results would then have excesses of 2 on either end
    # So we would want to start from [2:t-3] to avoid averages from any stimulation
    kernel = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    #kernel = np.array([1/3, 1/3, 1/3])
    #kernel = np.array([1/7,1/7,1/7,1/7,1/7,1/7,1/7])
    # Simulation
    y = ensemble - ensemble[:40].mean()
    x = ensemble - ensemble[:40].mean()
    x = kernel
    
    y = definite
    c = np.convolve(y,x)
    early_average = c[4:51-4].mean()
    early_std = c[4:51-4].std()
    plt.plot(c[4:-4])
    
    
    
    noise = ensemble[:40].std()
    # measure ensemble's signal to noise
    snr_have = (y.max()/noise) # SNR of ensemble is around 75625
    snrs = np.logspace(0, 1., 21)
    trials = 1000
    result = np.zeros([trials, snrs.size])
    thresh = result*0
    false_pos = result*0
    false_thresh = result*0
    average_trace = np.convolve(y,x)*0
#    average_trace = y*0
    conv_offset = x.size - 1
    center = 50 + x.size - 2
    for j in range(snrs.size):
        average_trace = average_trace *0
        for k in range(trials):
            noise =  np.random.normal(0,scale = y.max()/np.sqrt(snrs[j]), size = y.size)
            y_prime = y + noise
            bkn = y_prime[:48].std()
            c = np.convolve(noise,x)
            mu = c[conv_offset:center-conv_offset].mean()
            sig = c[conv_offset:center-conv_offset].std()
            if c[center] > mu + sig:
                false_pos[k,j] = 1
            false_thresh[k,j] = noise[49:52].mean()> noise[:48].mean() + bkn
            c = np.convolve(y_prime,x)
            bkn = y_prime[:48].std()
            thresh[k,j] = y_prime[49:52].mean() > y_prime[:48].mean() + bkn
            mu = c[conv_offset:center-conv_offset].mean()
            sig = c[conv_offset:center-conv_offset].std()
            if c[center] > mu + sig:
                result[k,j] =  1
            plt.plot(c[conv_offset:-conv_offset], alpha = 0.5)
            average_trace = average_trace + c
        plt.plot(average_trace[conv_offset:-conv_offset]/(k+1),'k')
        plt.plot([center-conv_offset, center-conv_offset],[-1000, 1000],'r')
        plt.title('SNR : {0}'.format(snrs[j]))
        plt.xlabel('Max height occurs at {0}'.format(np.argmax(average_trace)))
        plt.show()
        # plt.plot(correlate(average_trace,x))
        # plt.show()
    sums = result.sum(0)/trials
    sums2 = thresh.sum(0)/trials
    sums3 = false_pos.sum(0)/trials
    sums4 = false_thresh.sum(0)/trials
    print("Success readout")
    print("    SNR     Conv    Thresh  False+  False T")
    conv_err = result.std(0)/np.sqrt(trials)
    thresh_err = thresh.std(0)/np.sqrt(trials)
    fpos_err = false_pos.std(0)/np.sqrt(trials)
    ft_err = false_thresh.std(0)/np.sqrt(trials)
    plt.errorbar(snrs,sums,conv_err, label = 'Convolution')
    plt.errorbar(snrs,sums2,thresh_err, label = 'Threshold')
    plt.errorbar(snrs,sums3,fpos_err, label = 'False Positive Conv')
    plt.errorbar(snrs,sums4,ft_err, label = 'False Positive Thresh')
    plt.xlabel('Peak Signal to Noise')
    plt.ylabel('Percentage')
    plt.xscale('log')
    plt.legend()
    plt.show()
    plt.plot(snrs, sums3/(sums3 + sums), label = 'Conv ratio')
    plt.plot(snrs, sums4/(sums2 + sums4), label = 'Old ratio')
    plt.ylabel('False + / All')
    plt.xlabel('Peak Signal to Noise')
    plt.legend()
    plt.show()
    
    for i in range(snrs.size):
        print('{0:6} : {1:6} : {2:6} : {3:6}: {4:6}'.format(snrs[i].round(1), sums[i], sums2[i], sums3[i], sums4[i]))
        
    