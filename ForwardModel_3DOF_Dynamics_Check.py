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
from annimation_generator import animate_pendulum
#from scipy.stats.stats import pearsonr

def ITP(Y, x):
    
    
    if x >= duration:
        a = func_acc(duration)
    else:
        a = func_acc(x)
    
    
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
    k00 = p[0]
    k01 = p[1]
    k02 = p[2]
    k03 = p[3]
    k04 = p[4]
    k05 = p[5]
    k10 = p[6]
    k11 = p[7]
    k12 = p[8]
    k13 = p[9]
    k14 = p[10]
    k15 = p[11]
    k20 = p[12]
    k21 = p[13]
    k22 = p[14]
    k23 = p[15]
    k24 = p[16]
    k25 = p[17]
    
    ref_a = p[18]
    ref_k = p[19]
    ref_h = p[20]
    
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
                            - l_L*l_U*m_T*omega_a**2*sin(theta_k)- (k00*(theta_a-ref_a) + k01*(theta_k-ref_k) + k02*(theta_h-ref_h) + k03*omega_a + k04*omega_k + k05*omega_h))],
                    
                           [(-a*d_T*m_T*cos(theta_a+theta_k+theta_h) - a*d_U*m_U*cos(theta_a+theta_k) - a*l_U*m_T*cos(theta_a+theta_k) + d_T*g*m_T*sin(theta_a+theta_k+theta_h)
                            - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h) - d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h) + d_T*l_U*m_T*(omega_a+omega_k+omega_h)**2*sin(theta_h) + d_U*g*m_U*sin(theta_a+theta_k)
                            - d_U*l_L*m_U*omega_a**2*sin(theta_k)+g*l_U*m_T*sin(theta_a+theta_k) - l_L*l_U*m_T*omega_a**2*sin(theta_k)
                            - (k10*(theta_a-ref_a) + k11*(theta_k-ref_k) + k12*(theta_h-ref_h) + k13*omega_a + k14*omega_k + k15*omega_h))],
            
                           [(-a*d_T*m_T*cos(theta_a+theta_k+theta_h) + d_T*g*m_T*sin(theta_a+theta_k+theta_h) - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h)-d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h)
                            - (k20*(theta_a-ref_a) + k21*(theta_k-ref_k) + k22*(theta_h-ref_h) + k23*omega_a + k24*omega_k + k25*omega_h))]
            
                            ])
                    
    
    M_rev = M.I
    
    DY =np.dot(M_rev, ForceVec)
    DY = np.asarray(DY).reshape(-1)
    
    return DY

# Initial conditions on y, y' at x=0
#init = 0.1, -0.1, 0, 0

num_nodes = 5001
duration = 100.0
#startpoint = 1000

num_par = 21

Accel_String = 'Experiment Data/Subj03/SwayL0002.txt'
X_String = 'Experiment Data/Subj03/JointsGFL0002.txt'

accel_meas_full = np.loadtxt(Accel_String)
x_traj = np.loadtxt(X_String)
# First integrate from 0 to 20
time = np.linspace(0,duration,num_nodes)

#init = np.zeros(6)
#
#init[0] = -0.1
#init[1] = 0.2
#init[2] = -0.13



#Ta = 0
#Tk = 0
#Th = 0


#x_traj[8000, 1:]*3.1416/180.0

p = np.loadtxt('Results_150_250/Subj03/Pert1_SFPDRL/Controller_038.txt')
BestTraj = np.loadtxt('Results_150_250/Subj03/Pert1_SFPDRL/TrajectoryResult_038.txt')
init = BestTraj[0, :]
#p = np.array([132, 0, 0, 80, 0, 0,
#              0, 93.1, 0, 0, 40, 0,
#              0, 0, 59.2, 0, 0, 13.5,
#              -0.08, 0.2, -0.13])

#p[:num_par-3] = p[:num_par-3]*100

Para_string = 'Model Parameter/Para_Winter_Subj03.txt' 

parameter = np.loadtxt(Para_string)

#base_accel = np.zeros(num_nodes)
##base_accel[10:30] = -1
#
#base_motion = np.zeros(num_nodes)
#base_vel = np.zeros(num_nodes)
#
#for k in range(1, num_nodes):
#    base_vel[k] = base_vel[k-1] + base_accel[k-1]*0.02
#    base_motion[k] = base_motion[k-1] + base_vel[k-1]*0.02
#        
#func_acc = interp1d(time, base_accel)
func_acc = interp1d(time, accel_meas_full[8000:8000+num_nodes, 3])

scaling = 100

sol = odeint(ITP, init, time)

accel = np.zeros(num_nodes)

states = np.zeros((num_nodes, 4))

#states[:, 0] = -base_motion
states[:, 0] = -accel_meas_full[8000:8000+num_nodes, 1]
states[:, 1:4] = sol[:, 0:3]

length = np.array([0.45, 0.45, 0.85])

animate_pendulum(time, states, length, filename='Analysis_150_250/Subj03/Pert1_SFPDRLPD50/Best_fit.mp4')
                

fig1 = plt.figure(figsize=(12, 6))
fig1.add_subplot(3,1,1)
plt.plot(time, sol[:, 0]*180.0/3.1416, label = 'Controller Fit')
#plt.plot(time, BestTraj[:, 0]*180.0/3.1416, label = 'Best Identified')
plt.plot(time, x_traj[8000:8000+num_nodes, 1], label = 'Original Data')
plt.xlabel('Time (second)')
plt.ylabel('Ankle Motion (degree)')
plt.legend()

fig1.add_subplot(3,1,2)
plt.plot(time, sol[:, 1]*180.0/3.1416, label = 'Controller Fit')
#plt.plot(time, BestTraj[:, 1]*180.0/3.1416, label = 'Best Identified')
plt.plot(time, x_traj[8000:8000+num_nodes, 2], label = 'Original Data')
plt.xlabel('Time (second)')
plt.ylabel('Knee Motion (degree)')

fig1.add_subplot(3,1,3)
plt.plot(time, sol[:, 2]*180.0/3.1416, label = 'Controller Fit')
#plt.plot(time, BestTraj[:, 2]*180.0/3.1416, label = 'Best Identified')
plt.plot(time, x_traj[8000:8000+num_nodes, 3], label = 'Original Data')
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