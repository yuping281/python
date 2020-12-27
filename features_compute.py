# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:45:59 2019

@author: mayn
"""
import numpy as np

def compute_density(features):
    t_1 = features['Temp']
    M = 18.129
    p = features['P']
    p_1 = p
    x_n = 0.00702
    x_c = 0.00338
    Z_a = 0.99963
    M_a = 28.9626
    R = 0.00831451
    G_i = M/M_a
    
    K_t = (x_c+1.681*x_n)*100
    K_p = (x_c-0.392*x_n)*100
    F_t = 226.29/(99.15+211.9*G_i-K_t)
    F_p = 156.47/(160.8-7.22*G_i+K_p)
    t_j = (1.8*t_1+492)*F_t-460
    p_j = 145.04*p_1*F_p
    tou = (t_j+460)/500
    
    m = 0.033037*tou**(-2)-0.0221323*tou**(-3)+0.0161353*tou**(-5)
    n = (-0.133185*tou**(-1)+0.265827*tou**(-2)+0.0457697*tou**(-4))/m
    H = (p_j+14.7)/1000
    B = (3-m*n**2)/(9*m*H**2);
    E = 1-0.00075*H**2.3*(2-np.exp(-20*(1.09-tou)))-1.317*(1.09-tou)**4*H*(1.69-H**2)
 
    b = (9*n-2*m*n**3)/(54*m*H**3)-E/(2*m*H**2)
    D = (b+(b**2+B**3)**0.5)**(1/3)
    Den_G = M_a**G_i**(p_1+0.101325)/R/Z_a/(t_1+273.15)*(B/D-D+n/(3*H))/(1+0.00132/tou**3.25)**2
    return Den_G



# def compute_freq_power(y):
#     Fs = 10.0; # sampling rate采样率
#     Ts = 1.0/Fs; # sampling interval 采样区间
#     # t = np.arange(0,1,Ts) # time vector,这里Ts也是步长
#
#     # ff = 25; # frequency of the signal信号频率
#
#     n = len(y) # length of the signal
#     k = np.arange(n)
#     T = n/Fs
#     frq = k/T # two sides frequency range
#     frq1 = frq[range(int(n/2))] # one side frequency range
#
#     YY = np.fft.fft(y) # 未归一化
#     Y1 = YY[range(int(n/2))]
#     power = abs(Y1)**2
#     return power,frq1
#
#
# def compute_f1(p,f):
#     return (p*f).sum()/p.sum()
#
# def compute_f2(p,f):
#     Ai = 0
#     Aix = 0
#     for i in range(len(p)-1):
#         Ai += 0.5*(p[i+1]+p[i])*(f[i+1]-f[i])
#         Aix += 1/6*(p[i+1]-p[i])*(f[i+1]**2+f[i+1]*f[i]+f[i])
#     return Aix/Ai
#
# def compute_Entropy(p):
#     di = p/p.sum()
#     return -(di*np.log2(di)).sum()
#
# def compute_SF(p):
#     p_mean = p.mean()
#     n = len(p)
#     return (((p-p_mean)**2).sum()/(n-1))**0.5/p_mean
#
# def compute_skew(y):
#     return (((y-y.mean())/y.std())**3).sum()/len(y)
#
# def compute_kurt(y):
#     return (((y-y.mean())/y.std())**4).sum()/len(y)
#
# def compute_zrcs(y):
#     zrcs = 0
#     for i in range(len(y)-1):
#         if y.iloc[i]*y.iloc[i+1]<0:
#             zrcs += 1
#     return zrcs
#
# def compute_avgcs(y):
#     avgcs = 0
#     avg = y.mean()
#     for i in range(len(y)-1):
#         if (y.iloc[i]-avg)*(y.iloc[i+1]-avg)<0:
#             avgcs += 1
#     return avgcs