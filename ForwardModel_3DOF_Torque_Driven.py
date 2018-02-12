#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:25:37 2017

@author: huawei
"""

# Geneerate Ode Function
import numpy as np
from numpy import cos, sin
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from annimation_generator import animate_pendulum
#from scipy.stats.stats import pearsonr

def ITP(Y, x):
    
    
    if x >= duration:
        a = func_acc(duration)
        Tor_a = func_tor_a(duration)
        Tor_k = func_tor_k(duration)
        Tor_h = func_tor_h(duration)
    else:
        a = func_acc(x)
        Tor_a = func_tor_a(x)
        Tor_k = func_tor_k(x)
        Tor_h = func_tor_h(x)
    
    
    l_L = parameter[0]
    l_U = parameter[1]
    d_L = parameter[2]
    d_U = parameter[3]
    d_T = parameter[4]
    m_L = parameter[5]/scaling
    m_U = parameter[6]/scaling
    m_T = parameter[7]/scaling
    
    I_Lz = parameter[8]/scaling
    I_Uz = parameter[9]/scaling
    I_Tz = parameter[10]/scaling

    g = 0.981*10
    
    theta_a = Y[0]
    theta_k = Y[1]
    theta_h = Y[2]
    omega_a = Y[3]
    omega_k = Y[4]
    omega_h = Y[5]
#           
    
    M = np.matrix([
                   [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                   
                   [0, 0, 0, (I_Lz+I_Tz+I_Uz+d_L**2*m_L+m_T*(d_T**2+2*d_T*l_L*
                    cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h) + l_L**2+2*l_L*
                    l_U*cos(theta_k)+l_U**2) + m_U*(d_U**2+2*d_U*l_L*cos(theta_k)+l_L**2)),
    
                    (I_Tz+I_Uz+d_U*m_U*(d_U+l_L*cos(theta_k))+m_T*(d_T**2+d_T*
                    l_L*cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h)+l_L*l_U*cos(theta_k)+l_U**2)),
                     
                    (I_Tz+d_T*m_T*(d_T+l_L*cos(theta_k+theta_h)+l_U*cos(theta_h)))],
                                                             
                   [0, 0, 0, (I_Tz+I_Uz+d_U*m_U*(d_U+l_L*cos(theta_k))+m_T*(d_T**2+d_T*l_L*
                     cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h)+l_L*l_U*cos(theta_k)+l_U**2)),
                                                             
                    (I_Tz+I_Uz+d_U**2*m_U+m_T*(d_T**2+2*d_T*l_U*cos(theta_h)+l_U**2)),
                    
                    (I_Tz+d_T*m_T*(d_T+l_U*cos(theta_h)))],
                   
                   [0, 0, 0, (I_Tz+d_T*m_T*(d_T+l_L*cos(theta_k+theta_h) + l_U*cos(theta_h))),
                    (I_Tz+d_T*m_T*(d_T+l_U*cos(theta_h))), (I_Tz+d_T**2*m_T)]
                   
                   ])
    
    ForceVec = np.matrix([
                            [omega_a], [omega_k], [omega_h], 
                          [(-a*d_L*m_L*cos(theta_a) -a*d_T*m_T*cos(theta_a+theta_k+theta_h) 
                            - a*d_U*m_U*cos(theta_a+theta_k) - a*l_L*m_T*cos(theta_a) - a*l_L*m_U*cos(theta_a) 
                            - a*l_U*m_T*cos(theta_a+theta_k) + d_L*g*m_L*sin(theta_a) + d_T*g*m_T*sin(theta_a+theta_k+theta_h) 
                            + d_T*l_L*m_T*(omega_a+omega_k+omega_h)**2*sin(theta_k+theta_h) - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h) 
                            - d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h) + d_T*l_U*m_T*(omega_a+omega_k+omega_h)**2*sin(theta_h)
                            + d_U*g*m_U*sin(theta_a+theta_k) + d_U*l_L*m_U*(omega_a+omega_k)**2*sin(theta_k) - d_U*l_L*m_U*omega_a**2*sin(theta_k)
                            + g*l_L*m_T*sin(theta_a) + g*l_L*m_U*sin(theta_a) + g*l_U*m_T*sin(theta_a+theta_k) + l_L*l_U*m_T*(omega_a+omega_k)**2*sin(theta_k)
                            - l_L*l_U*m_T*omega_a**2*sin(theta_k)- Tor_a)],
                    
                           [(-a*d_T*m_T*cos(theta_a+theta_k+theta_h) - a*d_U*m_U*cos(theta_a+theta_k) - a*l_U*m_T*cos(theta_a+theta_k) + d_T*g*m_T*sin(theta_a+theta_k+theta_h)
                            - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h) - d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h) + d_T*l_U*m_T*(omega_a+omega_k+omega_h)**2*sin(theta_h) + d_U*g*m_U*sin(theta_a+theta_k)
                            - d_U*l_L*m_U*omega_a**2*sin(theta_k)+g*l_U*m_T*sin(theta_a+theta_k) - l_L*l_U*m_T*omega_a**2*sin(theta_k)
                            - Tor_k)],
            
                           [(-a*d_T*m_T*cos(theta_a+theta_k+theta_h) + d_T*g*m_T*sin(theta_a+theta_k+theta_h) - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h)-d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h)
                            - Tor_h)]
            
                            ])
                    
    
    M_rev = M.I
    
    DY =np.dot(M_rev, ForceVec)
    DY = np.asarray(DY).reshape(-1)
    
    return DY

# Initial conditions on y, y' at x=0
#init = 0.1, -0.1, 0, 0

num_nodes = 1001
duration = 20.0
#startpoint = 1000

#num_par = 21

Accel_String = 'Experiment Data/Subj03/SwayL0002.txt'
X_String = 'x_meas.txt'
Torque_String = 'torque.txt'

accel_meas_full = np.loadtxt(Accel_String)
x_meas = np.loadtxt(X_String)
torque_meas = np.loadtxt(Torque_String)

accel_meas = (accel_meas_full[8000:15501-1, 3] + accel_meas_full[8001:15501, 3])/2.0

init = x_meas[0, :]

# First integrate from 0 to 20
time = np.linspace(0,duration,num_nodes)

Para_string = 'Model Parameter/Para_Winter_Subj03.txt' 
parameter = np.loadtxt(Para_string)

func_tor_a = interp1d(time, -torque_meas[:num_nodes, 0])
func_tor_k = interp1d(time, -torque_meas[:num_nodes, 1])
func_tor_h = interp1d(time, -torque_meas[:num_nodes, 2])

func_acc = interp1d(time, accel_meas[:num_nodes])

scaling = 100

sol = odeint(ITP, init, time)

#accel = np.zeros(num_nodes)
#
#states = np.zeros((num_nodes, 4))
#
##states[:, 0] = -base_motion
#states[:, 0] = -accel_meas_full[8000:8000+num_nodes, 1]
#states[:, 1:4] = sol[:, 0:3]
#
#length = np.array([0.45, 0.45, 0.85])
#
#animate_pendulum(time, states, length, filename='Analysis_150_250/Subj03/Pert1_SFPDRLPD50/Best_fit.mp4')
#                

fig1 = plt.figure(figsize=(12, 6))
fig1.add_subplot(3,1,1)
plt.plot(time, sol[:, 0]*180.0/3.1416, label = 'Controller Fit')
#plt.plot(time, BestTraj[:, 0]*180.0/3.1416, label = 'Best Identified')
plt.plot(time, x_meas[:num_nodes, 0]*180.0/3.1416, label = 'Original Data')
plt.xlabel('Time (second)')
plt.ylabel('Ankle Motion (degree)')
plt.legend()

fig1.add_subplot(3,1,2)
plt.plot(time, sol[:, 1]*180.0/3.1416, label = 'Controller Fit')
#plt.plot(time, BestTraj[:, 1]*180.0/3.1416, label = 'Best Identified')
plt.plot(time, x_meas[:num_nodes, 1]*180.0/3.1416, label = 'Original Data')
plt.xlabel('Time (second)')
plt.ylabel('Knee Motion (degree)')

fig1.add_subplot(3,1,3)
plt.plot(time, sol[:, 2]*180.0/3.1416, label = 'Controller Fit')
#plt.plot(time, BestTraj[:, 2]*180.0/3.1416, label = 'Best Identified')
plt.plot(time, x_meas[:num_nodes, 2]*180.0/3.1416, label = 'Original Data')
plt.xlabel('Time (second)')
plt.ylabel('Hip Motion (degree)')





#with open('NormalDistributionNoise'+str(m+1) +'.txt','w') as Outfile:
#    StringP = ""
#    for n in range(num_nodes):
#        StringP += str(W1d[n])
#        StringP += " "
#        StringP += str(W2d[n])
#        StringP += " "
#        StringP += str(W3[n])
#        StringP += " "
#        StringP += str(W4[n])
#        StringP += "\n"
#    Outfile.write(StringP)
#
#with open('TrajectoryWithNoise'+str(m+1) +'.txt','w') as Outfile:
#    StringP = ""
#    for k in range(num_nodes):
#        for l in range(4):
#            StringP += str(sol[k, l])
#            StringP += " "
#        StringP += "\n"
#    Outfile.write(StringP)