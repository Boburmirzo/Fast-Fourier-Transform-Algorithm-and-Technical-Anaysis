# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:15:25 2018

@author: bumurzokov
"""

import numpy as np
import pylab as pl
from numpy import fft
import pandas as pd

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n) 
    indexes = list(range(n))             # frequencies
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

#Open csv file and parse Adj close value 
def main():
    data = pd.read_csv('MSFTnew.csv')
    price = data['Adj Close']
    print(price)
    print(data.dtypes);
    x = np.array(price)
    n_predict = 100
    extrapolation = fourierExtrapolation(x, n_predict)
    pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r')
    pl.plot(np.arange(0, x.size), x, 'b', linewidth = 3)
    pl.legend()
    pl.show()
    
if __name__ == "__main__":
    main()