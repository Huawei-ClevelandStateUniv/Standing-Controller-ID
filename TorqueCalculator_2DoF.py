#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:23:02 2018

@author: huawei
"""

import numpy as np
from StandingModel_2DoF import Model
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

#import plotly
#plotly.tools.set_credentials_file(username='HuaweiWang',
#                                  api_key='u3JPHaMgbggGVFml1yUd')
#
#import plotly.plotly as py
#import plotly.graph_objs as go

num_nodes = 7500
duration = 100.0

interval = duration/(num_nodes-1)

start_nodes = 8000

num_states = 4
num_par = 8

Accel_String = 'Experiment Data/Subj03/SwayL0002.txt'
X_String = 'Experiment Data/Subj03/JointsGFL0002.txt'

Para_string = 'Model Parameter/Para_Winter_Subj03.txt' 
parameter = np.loadtxt(Para_string)

accel_meas_full = np.loadtxt(Accel_String)
x_traj = np.loadtxt(X_String)
time = np.linspace(0,duration,num_nodes)

x = x_traj[start_nodes:start_nodes+num_nodes, [1, 3, 4, 6]]*3.1416/180.0
accel = accel_meas_full[start_nodes:start_nodes+num_nodes, 3]

b, a = butter(2, 6.0/(50/2))

x_f = filtfilt(b, a, x.T)
accel_f = filtfilt(b, a, accel.T)

x_f = x_f.T
accel_f = accel_f.T

Con = np.zeros(num_par)

Question = Model(x_f, accel_f, num_nodes, interval, parameter,
                 scaling = 100.0, integration_method ='midpoint')


x_meas = np.zeros((num_nodes-1, num_states))
xdot_meas = np.zeros((num_nodes-1, num_states))
accel_meas = np.zeros((num_nodes-1, 1))
torque = np.zeros((num_nodes-1, num_states))

for k in range(1, num_nodes):
    x_meas[k-1, :] = (x_f[k,:]+x_f[k-1,:])/2
    xdot_meas[k-1, :] = (x_f[k,:]-x_f[k-1,:])/interval
    accel_meas[k-1] = (accel_f[k] - accel_f[k-1])/interval

    torque[k-1, :], dfdx, dfdxdot, dfdp = Question.dynamic_fun(x_meas[k-1, :], xdot_meas[k-1, :], Con, accel_meas[k-1])

fig = plt.figure(figsize=(12, 6))

plt.plot(time[:1000], torque[:1000, 2])
plt.plot(time[:1000], torque[:1000, 3])


#with open('x_meas.txt','w') as Outfile:
#    StringP = ""
#    for n in range(num_nodes-1):
#        for m in range (num_states):
#            StringP += str(x_meas[n, m])
#            StringP += " "
#        StringP += "\n"
#    Outfile.write(StringP)
#
#with open('torque.txt','w') as Outfile:
#    StringP = ""
#    for k in range(num_nodes-1):
#        for l in range(num_states/2):
#            StringP += str(torque[k, l+3])
#            StringP += " "
#        StringP += "\n"
#    Outfile.write(StringP)



#trace1 = go.Scatter(
#    x = x_meas[:, 0],
#    y = torque[:, 5],
#    mode = 'markers'
#)
#
#trace2 = go.Scatter(
#    x = x_meas[:, 1],
#    y = torque[:, 5],
#    mode = 'markers'
#)
#
#trace3 = go.Scatter(
#    x = x_meas[:, 2],
#    y = torque[:, 5],
#    mode = 'markers'
#)
#
#trace4 = go.Scatter(
#    x = x_meas[:, 3],
#    y = torque[:, 5],
#    mode = 'markers'
#)
#
#trace5 = go.Scatter(
#    x = x_meas[:, 4],
#    y = torque[:, 5],
#    mode = 'markers'
#)
#
#trace6 = go.Scatter(
#    x = x_meas[:, 5],
#    y = torque[:, 5],
#    mode = 'markers'
#)
#
#data = [trace1, trace2, trace3, trace4, trace5, trace6]
#fig = go.Figure(data=data)
#
#py.iplot(fig, filename='hip_torque_experiment')