#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:02:38 2017

@author: huawei
"""
import numpy as np
#import sympy as sy
from numpy import sin, cos
from scipy.sparse import find

class Model(object):
    def __init__(self, x_meas, accel_meas, num_nodes, num_states, num_par,
                 interval, parameter, scaling = 1.0, integration_method ='backward euler'):
        
        self.x_meas = x_meas
        self.accel_meas = accel_meas
        self.num_nodes = num_nodes
        self.interval = interval
        self.parameter = parameter
        self.scaling = scaling
        self.intergration_method = integration_method
        
        self.num_states = num_states
        self.num_cons = num_states
        self.num_par = num_par
        self.num_conspernode = num_states
        
        self.Jac_Ind()
        
        self.itheta_a = np.linspace(0, self.num_states*self.num_nodes, self.num_nodes, endpoint=False, dtype=int)
        self.itheta_k = np.linspace(0, self.num_states*self.num_nodes, self.num_nodes, endpoint=False, dtype=int) + 1
        self.itheta_h = np.linspace(0, self.num_states*self.num_nodes, self.num_nodes, endpoint=False, dtype=int) + 2
                                 
    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        N = self.num_nodes
        
        f_theta_a = np.sum((x[self.itheta_a] - self.x_meas[self.itheta_a])**2)
        f_theta_k = np.sum((x[self.itheta_k] - self.x_meas[self.itheta_k])**2)
        f_theta_h = np.sum((x[self.itheta_h] - self.x_meas[self.itheta_h])**2)
        
        fs = (f_theta_a + f_theta_k + f_theta_h)/N
        
        
        obj = fs
        
        
        return  obj
    
    def gradient(self, x):
        #
        # The callback for calculating the gradient
        N = self.num_nodes
        
        grad = np.zeros_like(x)
        
#        grad[:self.num_states*self.num_nodes] = 2.0*self.interval*(x[:self.num_states*self.num_nodes] - self.x_meas)
        grad[self.itheta_a] = 2.0*(x[self.itheta_a] - self.x_meas[self.itheta_a])/N
        grad[self.itheta_k] = 2.0*(x[self.itheta_k] - self.x_meas[self.itheta_k])/N
        grad[self.itheta_h] = 2.0*(x[self.itheta_h] - self.x_meas[self.itheta_h])/N
        
        return grad

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        N = self.num_nodes
        S = self.num_states
        C = self.num_cons
        h = self.interval
        
        a = self.accel_meas
        cons = np.zeros((C*(N - 1)))
        par = x[-self.num_par:]
        
        for p in range(N-1):
            
            xp = x[(p)*S:(p+1)*S]
            xn = x[(p+1)*S:(p+2)*S]

            ap = a[p]
            an = a[p+1]
            
            if self.intergration_method == 'backward euler':
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun(xn, (xn-xp)/h, par, an)            
            elif self.intergration_method == 'midpoint':
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun((xn + xp)/2, (xn - xp)/h, par, (ap + an)/2)
            else:
                print 'Do not have the Intergration Method code'
                
            cons[S*p: S*(p+1)] = f
                 
        return cons
    
    def jacobianstructure(self):
        # Sepcify the structure of Jacobian, in case the size of Jacobian is too large. 
        
        N = self.num_nodes
        S = self.num_states
        C = self.num_cons
        
#        self.Row = np.zeros(((N-1)*(S*2 + S*(S/2) + (P-S/2) + S/2*S/2)))
#        self.Col = np.zeros(((N-1)*(S*2 + S*(S/2) + (P-S/2) + S/2*S/2)))

        self.Row = np.array([])
        self.Col = np.array([])
        
        for j in range(0, N-1):
                          
            x = np.array([j*C, j*C, j*C, j*C, j*C+1, j*C+1, j*C+1, j*C+1, j*C+2,
                          j*C+2, j*C+2, j*C+2, j*C+3, j*C+3, j*C+3, j*C+3, j*C+3,
                          j*C+3, j*C+3, j*C+3, j*C+3, j*C+3, j*C+3, j*C+3, j*C+3,
                          j*C+3, j*C+3, j*C+3, j*C+3, j*C+3, j*C+3, j*C+3, j*C+3,
                          j*C+4, j*C+4, j*C+4, j*C+4, j*C+4, j*C+4, j*C+4, j*C+4,
                          j*C+4, j*C+4, j*C+4, j*C+4, j*C+4, j*C+4, j*C+4, j*C+4,
                          j*C+4, j*C+4, j*C+4, j*C+4, j*C+4, j*C+5, j*C+5, j*C+5,
                          j*C+5, j*C+5, j*C+5, j*C+5, j*C+5, j*C+5, j*C+5, j*C+5,
                          j*C+5, j*C+5, j*C+5, j*C+5, j*C+5, j*C+5, j*C+5, j*C+5,
                          j*C+5, j*C+5])
    
            y = np.array([j*S, j*S+3, j*S+6, j*S+9, j*S+1, j*S+4, j*S+7, j*S+10,
                          j*S+2, j*S+5, j*S+8, j*S+11, j*S, j*S+1, j*S+2, j*S+3,
                          j*S+4, j*S+5, j*S+6, j*S+7, j*S+8, j*S+9, j*S+10, j*S+11,
                          N*S, N*S+1, N*S+2, N*S+3, N*S+4, N*S+5, N*S+18, N*S+19, N*S+20, 
                          j*S, j*S+1, j*S+2, j*S+3, j*S+4, j*S+5, j*S+6, j*S+7,
                          j*S+8, j*S+9, j*S+10, j*S+11, N*S+6, N*S+7, N*S+8, N*S+9, N*S+10, 
                          N*S+11, N*S+18, N*S+19, N*S+20, j*S, j*S+1, j*S+2, j*S+3, j*S+4, 
                          j*S+5, j*S+6, j*S+7, j*S+8, j*S+9, j*S+10, j*S+11,
                          N*S+12, N*S+13, N*S+14, N*S+15, N*S+16, N*S+17, N*S+18, N*S+19, N*S+20])
    
            x.astype(int)
            y.astype(int)
            
            self.Row = np.hstack((self.Row, x))
            self.Col = np.hstack((self.Col, y))
        
        return (self.Row, self.Col)
    
    def Jac_Ind(self):
        
        self.Row_part = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2,
                          2, 2, 2, 3, 3, 3, 3, 3,
                          3, 3, 3, 3, 3, 3, 3, 3,
                          3, 3, 3, 3, 3, 3, 3, 3,
                          4, 4, 4, 4, 4, 4, 4, 4,
                          4, 4, 4, 4, 4, 4, 4, 4,
                          4, 4, 4, 4, 4, 5, 5, 5,
                          5, 5, 5, 5, 5, 5, 5, 5,
                          5, 5, 5, 5, 5, 5, 5, 5,
                          5, 5])
    
        self.Col_part = np.array([0, 3, 6, 9, 1, 4, 7, 10,
                          2, 5, 8, 11, 0, 1, 2, 3,
                          4, 5, 6, 7, 8, 9, 10, 11,
                          12, 13, 14, 15, 16, 17, 30, 31, 32, 
                          0, 1, 2, 3, 4, 5, 6, 7,
                          8, 9, 10, 11, 18, 19, 20, 21, 22, 
                          23, 30, 31, 32, 0, 1, 2, 3, 4, 
                          5, 6, 7, 8, 9, 10, 11,
                          24, 25, 26, 27, 28, 29, 30, 31, 32])
                         
        return self.Row_part, self.Col_part

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        N = self.num_nodes
        P = self.num_par
        h = self.interval
        S = self.num_states
        C = self.num_cons
        
        a = self.accel_meas
        
        par = x[-self.num_par:]
        
        Jac = np.array([])

        for k in range(N-1):
            
            xp = x[(k)*S:(k+1)*S]
            xn = x[(k+1)*S:(k+2)*S]
            
            ap = a[k]
            an = a[k+1]
            
            
            if self.intergration_method == 'backward euler':
                Jac_part = np.zeros((C, 2*S+P))
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun(xn, (xn-xp)/h, par, an)  
                
                Jac_part[:, :S] = -dfdxdot/h
                Jac_part[:, S:2*S] = dfdx + dfdxdot/h
                Jac_part[:, 2*S:2*S+P] = dfdp
                   
                row, col, RA = find(Jac_part)
                Jac = np.hstack(Jac, RA)

            elif self.intergration_method == 'midpoint':
#        self.Row = np.zeros(((N-1)*(S*2 + S*(S/2) + (P-S/2) + S/2*S/2)))

                Jac_part = np.zeros((C, 2*S+P))
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun((xn + xp)/2, (xn - xp)/h, par, (ap + an)/2)
                
                Jac_part[:, :S] = dfdx/2 -dfdxdot/h
                Jac_part[:, S:2*S] = dfdx/2 + dfdxdot/h
                Jac_part[:, 2*S:2*S+P] = dfdp
                        
                RA = Jac_part[self.Row_part, self.Col_part]
                        
                Jac = np.hstack((Jac, RA))
                
            else:
                print 'Do not have the Intergration Method code'
           
        return Jac
       
    def dynamic_fun(self, x, xdot, p, a):
            
        l_L = self.parameter[0]
        l_U = self.parameter[1]
        d_L = self.parameter[2]
        d_U = self.parameter[3]
        d_T = self.parameter[4]
        m_L = self.parameter[5]/self.scaling
        m_U = self.parameter[6]/self.scaling
        m_T = self.parameter[7]/self.scaling
        
        I_Lz = self.parameter[8]/self.scaling
        I_Uz = self.parameter[9]/self.scaling
        I_Tz = self.parameter[10]/self.scaling

        g = 9.81
        
        theta_a = x[0]
        theta_k = x[1]
        theta_h = x[2]
        omega_a = x[3]
        omega_k = x[4]
        omega_h = x[5]
        
        theta_a_dot = xdot[0]
        theta_k_dot = xdot[1]
        theta_h_dot = xdot[2]
        omega_a_dot = xdot[3]
        omega_k_dot = xdot[4]
        omega_h_dot = xdot[5]
               
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
        
        f = np.zeros((self.num_states))
        dfdx = np.zeros((self.num_states, self.num_states))
        dfdxdot = np.zeros((self.num_states, self.num_states))
        dfdp = np.zeros((self.num_states, self.num_par))
        
    
        f[0] = omega_a - theta_a_dot
    
        dfdx[0,3] = 1
        dfdxdot[0,0] = -1
               
        f[1] = omega_k - theta_k_dot
         
        dfdx[1,4] = 1
        dfdxdot[1,1] = -1         
        
        f[2] = omega_h - theta_h_dot
        
        dfdx[2,5] = 1
        dfdxdot[2,2] = -1
               
        f[3] = (-a*d_L*m_L*cos(theta_a) -a*d_T*m_T*cos(theta_a+theta_k+theta_h) 
                - a*d_U*m_U*cos(theta_a+theta_k) - a*l_L*m_T*cos(theta_a) - a*l_L*m_U*cos(theta_a) 
                - a*l_U*m_T*cos(theta_a+theta_k) + d_L*g*m_L*sin(theta_a) + d_T*g*m_T*sin(theta_a+theta_k+theta_h) 
                + d_T*l_L*m_T*(omega_a+omega_k+omega_h)**2*sin(theta_k+theta_h) - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h) 
                - d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h) + d_T*l_U*m_T*(omega_a+omega_k+omega_h)**2*sin(theta_h)
                + d_U*g*m_U*sin(theta_a+theta_k) + d_U*l_L*m_U*(omega_a+omega_k)**2*sin(theta_k) - d_U*l_L*m_U*omega_a**2*sin(theta_k)
                + g*l_L*m_T*sin(theta_a) + g*l_L*m_U*sin(theta_a) + g*l_U*m_T*sin(theta_a+theta_k) + l_L*l_U*m_T*(omega_a+omega_k)**2*sin(theta_k)
                - l_L*l_U*m_T*omega_a**2*sin(theta_k) - (I_Tz+d_T*m_T*(d_T+l_L*cos(theta_k+theta_h) + l_U*cos(theta_h)))*omega_h_dot
                - (I_Tz+I_Uz+d_U*m_U*(d_U+l_L*cos(theta_k))+m_T*(d_T**2+d_T*l_L*cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h)+l_L*l_U*cos(theta_k)+l_U**2))*omega_k_dot
                - (I_Lz+I_Tz+I_Uz+d_L**2*m_L+m_T*(d_T**2+2*d_T*l_L*cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h) + l_L**2+2*l_L*l_U*cos(theta_k)+l_U**2) + m_U*(d_U**2+2*d_U*l_L*cos(theta_k)+l_L**2))*omega_a_dot
                - (k00*(theta_a-ref_a) + k01*(theta_k-ref_k) + k02*(theta_h-ref_h) + k03*omega_a + k04*omega_k + k05*omega_h))
                
        dfdx[3, 0] = (a*d_L*m_L*sin(theta_a) + a*d_T*m_T*sin(theta_a+theta_k+theta_h) + a*d_U*m_U*sin(theta_a+theta_k)+a*l_L*m_T*sin(theta_a)
                      + a*l_L*m_U*sin(theta_a) + a*l_U*m_T*sin(theta_a+theta_k) + d_L*g*m_L*cos(theta_a) + d_T*g*m_T*cos(theta_a+theta_k+theta_h)
                      + d_U*g*m_U*cos(theta_a+theta_k) + g*l_L*m_T*cos(theta_a) + g*l_L*m_U*cos(theta_a) + g*l_U*m_T*cos(theta_a+theta_k)
                      - k00)
                     
        dfdx[3, 1] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h) + a*d_U*m_U*sin(theta_a+theta_k) + a*l_U*m_T*sin(theta_a+theta_k) 
                      + d_T*g*m_T*cos(theta_a+theta_k+theta_h) + d_T*l_L*m_T*(omega_a+omega_k+omega_h)**2*cos(theta_k+theta_h)
                      - d_T*l_L*m_T*omega_a**2*cos(theta_k+theta_h) + d_T*l_L*m_T*sin(theta_k+theta_h)*omega_h_dot 
                      + d_U*g*m_U*cos(theta_a+theta_k) + d_U*l_L*m_U*(omega_a+omega_k)**2*cos(theta_k) - d_U*l_L*m_U*omega_a**2*cos(theta_k)
                      + g*l_U*m_T*cos(theta_a+theta_k) + l_L*l_U*m_T*(omega_a+omega_k)**2*cos(theta_k) - l_L*l_U*m_T*omega_a**2*cos(theta_k)
                      - (-2*d_U*l_L*m_U*sin(theta_k)+m_T*(-2*d_T*l_L*sin(theta_k+theta_h) - 2*l_L*l_U*sin(theta_k)))*omega_a_dot
                      - (-d_U*l_L*m_U*sin(theta_k)+m_T*(-d_T*l_L*sin(theta_k+theta_h)-l_L*l_U*sin(theta_k)))*omega_k_dot
                      - k01)
                     
        dfdx[3, 2] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h) + d_T*g*m_T*cos(theta_a+theta_k+theta_h) + d_T*l_L*m_T*(omega_a+omega_k+omega_h)**2*cos(theta_k+theta_h)
                      - d_T*l_L*m_T*omega_a**2*cos(theta_k+theta_h) - d_T*l_U*m_T*(omega_a+omega_k)**2*cos(theta_h) + d_T*l_U*m_T*(omega_a+omega_k+omega_h)**2*cos(theta_h)
                      - d_T*m_T*(-l_L*sin(theta_k+theta_h)-l_U*sin(theta_h))*omega_h_dot - m_T*(-2*d_T*l_L*sin(theta_k+theta_h)-2*d_T*l_U*sin(theta_h))*omega_a_dot
                      - m_T*(-d_T*l_L*sin(theta_k+theta_h)-2*d_T*l_U*sin(theta_h))*omega_k_dot
                      - k02)
                     
        dfdx[3, 3] = (d_T*l_L*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_k+theta_h) - 2*d_T*l_L*m_T*omega_a*sin(theta_k+theta_h) - d_T*l_U*m_T*2*(omega_a+omega_k)*sin(theta_h)
                      + d_T*l_U*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_h) + d_U*l_L*m_U*2*(omega_a+omega_k)*sin(theta_k) - 2*d_U*l_L*m_U*omega_a*sin(theta_k)
                      + l_L*l_U*m_T*2*(omega_a+omega_k)*sin(theta_k) - 2*l_L*l_U*m_T*omega_a*sin(theta_k)
                      - k03)
                           
        dfdx[3, 4] = (d_T*l_L*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_k+theta_h) - d_T*l_U*m_T*2*(omega_a+omega_k)*sin(theta_h) + d_T*l_U*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_h)
                      + d_U*l_L*m_U*2*(omega_a+omega_k)*sin(theta_k) + l_L*l_U*m_T*2*(omega_a+omega_k)*sin(theta_k)
                      - k04)
                     
        dfdx[3, 5] = (d_T*l_L*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_k+theta_h) + d_T*l_U*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_h)
                      - k05)
                     
        dfdxdot[3, 3] = -(I_Lz+I_Tz+I_Uz+d_L**2*m_L+m_T*(d_T**2+2*d_T*l_L*cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h) + l_L**2+2*l_L*l_U*cos(theta_k)+l_U**2) + m_U*(d_U**2+2*d_U*l_L*cos(theta_k)+l_L**2))
                     
        dfdxdot[3, 4] = -(I_Tz+I_Uz+d_U*m_U*(d_U+l_L*cos(theta_k))+m_T*(d_T**2+d_T*l_L*cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h)+l_L*l_U*cos(theta_k)+l_U**2))             
        
        dfdxdot[3, 5] = -(I_Tz+d_T*m_T*(d_T+l_L*cos(theta_k+theta_h) + l_U*cos(theta_h)))
        
        dfdp[3, 0] = -(theta_a-ref_a)
        dfdp[3, 1] = -(theta_k-ref_k)
        dfdp[3, 2] = -(theta_h-ref_h)
        dfdp[3, 3] = -omega_a
        dfdp[3, 4] = -omega_k
        dfdp[3, 5] = -omega_h
        dfdp[3, 18] = k00
        dfdp[3, 19] = k01
        dfdp[3, 20] = k02
        
        f[4] = (-a*d_T*m_T*cos(theta_a+theta_k+theta_h) - a*d_U*m_U*cos(theta_a+theta_k) - a*l_U*m_T*cos(theta_a+theta_k) + d_T*g*m_T*sin(theta_a+theta_k+theta_h)
                - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h) - d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h) + d_T*l_U*m_T*(omega_a+omega_k+omega_h)**2*sin(theta_h) + d_U*g*m_U*sin(theta_a+theta_k)
                - d_U*l_L*m_U*omega_a**2*sin(theta_k)+g*l_U*m_T*sin(theta_a+theta_k) - l_L*l_U*m_T*omega_a**2*sin(theta_k)-(I_Tz+d_T*m_T*(d_T+l_U*cos(theta_h)))*omega_h_dot
                - (I_Tz+I_Uz+d_U**2*m_U+m_T*(d_T**2+2*d_T*l_U*cos(theta_h)+l_U**2))*omega_k_dot
                - (I_Tz+I_Uz+d_U*m_U*(d_U+l_L*cos(theta_k))+m_T*(d_T**2+d_T*l_L*cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h)+l_L*l_U*cos(theta_k)+l_U**2))*omega_a_dot
                - (k10*(theta_a-ref_a) + k11*(theta_k-ref_k) + k12*(theta_h-ref_h) + k13*omega_a + k14*omega_k + k15*omega_h))
                
        dfdx[4, 0] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h) + a*d_U*m_U*sin(theta_a+theta_k) + a*l_U*m_T*sin(theta_a+theta_k) 
                     + d_T*g*m_T*cos(theta_a+theta_k+theta_h)+d_U*g*m_U*cos(theta_a+theta_k) + g*l_U*m_T*cos(theta_a+theta_k)
                     - k10)
                     
        dfdx[4, 1] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h) + a*d_U*m_U*sin(theta_a+theta_k) + a*l_U*m_T*sin(theta_a+theta_k) + d_T*g*m_T*cos(theta_a+theta_k+theta_h) 
                     - d_T*l_L*m_T*omega_a**2*cos(theta_k+theta_h)+d_U*g*m_U*cos(theta_a+theta_k) - d_U*l_L*m_U*omega_a**2*cos(theta_k) + g*l_U*m_T*cos(theta_a+theta_k)
                     - l_L*l_U*m_T*omega_a**2*cos(theta_k) - (-d_U*l_L*m_U*sin(theta_k)+m_T*(-d_T*l_L*sin(theta_k+theta_h)-l_L*l_U*sin(theta_k)))*omega_a_dot
                     - k11)
                     
        dfdx[4, 2] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h) + d_T*g*m_T*cos(theta_a+theta_k+theta_h) - d_T*l_L*m_T*omega_a**2*cos(theta_k+theta_h) - d_T*l_U*m_T*(omega_a+omega_k)**2*cos(theta_h)
                     + d_T*l_U*m_T*(omega_a+omega_k+omega_h)**2*cos(theta_h) + 2*d_T*l_U*m_T*sin(theta_h)*omega_k_dot + d_T*l_U*m_T*sin(theta_h)*omega_h_dot
                     - m_T*(-d_T*l_L*sin(theta_k+theta_h)-2*d_T*l_U*sin(theta_h))*omega_a_dot
                     - k12)
                     
        dfdx[4, 3] = (-2*d_T*l_L*m_T*omega_a*sin(theta_k+theta_h) - d_T*l_U*m_T*2*(omega_a+omega_k)*sin(theta_h) + d_T*l_U*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_h) 
                     - 2*d_U*l_L*m_U*omega_a*sin(theta_k) - 2*l_L*l_U*m_T*omega_a*sin(theta_k)
                     - k13)
                     
        dfdx[4, 4] = (-d_T*l_U*m_T*2*(omega_a+omega_k)*sin(theta_h) + d_T*l_U*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_h)
                     - k14)
                     
        dfdx[4, 5] = d_T*l_U*m_T*2*(omega_a+omega_k+omega_h)*sin(theta_h) - k15
                     
        dfdxdot[4, 3] = -(I_Tz+I_Uz+d_U*m_U*(d_U+l_L*cos(theta_k))+m_T*(d_T**2+d_T*l_L*cos(theta_k+theta_h)+2*d_T*l_U*cos(theta_h)+l_L*l_U*cos(theta_k)+l_U**2))
        
        dfdxdot[4, 4] = -(I_Tz+I_Uz+d_U**2*m_U+m_T*(d_T**2+2*d_T*l_U*cos(theta_h)+l_U**2))
        
        dfdxdot[4, 5] = -(I_Tz+d_T*m_T*(d_T+l_U*cos(theta_h))) 
        
        dfdp[4, 6] = -(theta_a-ref_a)
        dfdp[4, 7] = -(theta_k-ref_k)
        dfdp[4, 8] = -(theta_h-ref_h)
        dfdp[4, 9] = -omega_a
        dfdp[4, 10] = -omega_k
        dfdp[4, 11] = -omega_h
        dfdp[4, 18] = k10
        dfdp[4, 19] = k11
        dfdp[4, 20] = k12
            
        f[5] = (-a*d_T*m_T*cos(theta_a+theta_k+theta_h) + d_T*g*m_T*sin(theta_a+theta_k+theta_h) - d_T*l_L*m_T*omega_a**2*sin(theta_k+theta_h)-d_T*l_U*m_T*(omega_a+omega_k)**2*sin(theta_h)
                -(I_Tz+d_T**2*m_T)*omega_h_dot-(I_Tz+d_T*m_T*(d_T+l_U*cos(theta_h)))*omega_k_dot-(I_Tz+d_T*m_T*(d_T+l_L*cos(theta_k+theta_h)+l_U*cos(theta_h)))*omega_a_dot
                 - (k20*(theta_a-ref_a) + k21*(theta_k-ref_k) + k22*(theta_h-ref_h) + k23*omega_a + k24*omega_k + k25*omega_h))
        
        dfdx[5, 0] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h)+d_T*g*m_T*cos(theta_a+theta_k+theta_h)
                     - k20)
        
        dfdx[5, 1] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h) + d_T*g*m_T*cos(theta_a+theta_k+theta_h) 
                     - d_T*l_L*m_T*omega_a**2*cos(theta_k+theta_h) + d_T*l_L*m_T*sin(theta_k+theta_h)*omega_a_dot
                     - k21)
                     
        dfdx[5, 2] = (a*d_T*m_T*sin(theta_a+theta_k+theta_h) + d_T*g*m_T*cos(theta_a+theta_k+theta_h) - d_T*l_L*m_T*omega_a**2*cos(theta_k+theta_h)
                     - d_T*l_U*m_T*(omega_a+omega_k)**2*cos(theta_h) + d_T*l_U*m_T*sin(theta_h)*omega_k_dot - d_T*m_T*(-l_L*sin(theta_k+theta_h)-l_U*sin(theta_h))*omega_a_dot
                     - k22)
                     
        dfdx[5, 3] = (-2*d_T*l_L*m_T*omega_a*sin(theta_k+theta_h) - d_T*l_U*m_T*2*(omega_a+omega_k)*sin(theta_h)
                     - k23)
                     
        dfdx[5, 4] = (-d_T*l_U*m_T*2*(omega_a+omega_k)*sin(theta_h)
                     - k24)
                     
        dfdx[5, 5] =- k25
            
        dfdxdot[5, 3] = -(I_Tz+d_T*m_T*(d_T+l_L*cos(theta_k+theta_h)+l_U*cos(theta_h)))
        dfdxdot[5, 4] = -(I_Tz+d_T*m_T*(d_T+l_U*cos(theta_h)))
        dfdxdot[5, 5] = -(I_Tz+d_T**2*m_T)
        
        dfdp[5, 12] = -(theta_a-ref_a)
        dfdp[5, 13] = -(theta_k-ref_k)
        dfdp[5, 14] = -(theta_h-ref_h)
        dfdp[5, 15] = -omega_a
        dfdp[5, 16] = -omega_k
        dfdp[5, 17] = -omega_h
        dfdp[5, 18] = k20
        dfdp[5, 19] = k21
        dfdp[5, 20] = k22
        
        return f, dfdx, dfdxdot, dfdp

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print "Objective value at iteration #%d is - %g" % (iter_count, obj_value)