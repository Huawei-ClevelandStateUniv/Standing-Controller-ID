#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:02:38 2017

@author: huawei
"""
import numpy as np
#import sympy as sy
from numpy import sin, cos

class Model(object):
    def __init__(self, x_meas, accel_meas, num_nodes, interval, parameter, scaling =1, integration_method='backward euler'):
        
        self.x_meas = x_meas
        self.accel_meas = accel_meas
        self.num_nodes = num_nodes
        self.interval = interval
        self.parameter = parameter
        self.scaling = scaling
        self.intergration_method = integration_method
        
        self.num_states = 4
        self.num_cons = 4
        self.num_par = 8
        self.num_conspernode = 4
        
        self.itheta_a = np.linspace(0, self.num_states*self.num_nodes, self.num_nodes, endpoint=False, dtype=int)
        self.itheta_h = np.linspace(0, self.num_states*self.num_nodes, self.num_nodes, endpoint=False, dtype=int) + 1

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        
        f_theta_a = np.sum((x[self.itheta_a] - self.x_meas[self.itheta_a])**2)
        f_theta_h = np.sum((x[self.itheta_h] - self.x_meas[self.itheta_h])**2)
        return  self.interval*(f_theta_a + f_theta_h)
    
    def gradient(self, x):
        #
        # The callback for calculating the gradient
        
        grad = np.zeros_like(x)
        
#        grad[:self.num_states*self.num_nodes] = 2.0*self.interval*(x[:self.num_states*self.num_nodes] - self.x_meas)
        grad[self.itheta_a] = 2.0*self.interval*(x[self.itheta_a] - self.x_meas[self.itheta_a])
        grad[self.itheta_h] = 2.0*self.interval*(x[self.itheta_h] - self.x_meas[self.itheta_h])
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
            
            if self.intergration_method == 'backward euler':
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun(x[(p+1)*S:(p+2)*S], (x[(p+1)*S:(p+2)*S]-x[(p)*S:(p+1)*S])/h, par, a[p+1])            
            elif self.intergration_method == 'midpoint':
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun((x[(p+1)*S:(p+2)*S] + x[(p)*S:(p+1)*S])/2, (x[(p+1)*S:(p+2)*S]-x[(p)*S:(p+1)*S])/h, par, (a[p] + a[p+1])/2)
            else:
                print 'Do not have the Intergration Method code'
                
            cons[S*p: S*(p+1)] = f
                 
        return cons
    
    def jacobianstructure(self):
        
        N = self.num_nodes
        P = self.num_par
        S = self.num_states
        C = self.num_cons
        
        self.Row = np.zeros(((N-1)*C*(S*2 + P)))
        self.Col = np.zeros(((N-1)*C*(S*2 + P)))
        
        for j in range(0, N-1):
                          
            x = np.array([j*C, C*j+1, C*j+2, C*j+3])
            y = np.array([j*S, j*S+1, j*S+2, j*S+3, j*S+4, j*S+5, j*S+6, j*S+7,
                          N*S, N*S+1, N*S+2, N*S+3, N*S+4, N*S+5, N*S+6, N*S+7,])
    
            x.astype(int)
            y.astype(int)
            
            self.Row[j*(S*C*2 + C*P):(j+1)*(S*C*2 + C*P)] = np.repeat(x, 2*S+P)
            self.Col[j*(S*C*2 + C*P):(j+1)*(S*C*2 + C*P)] = np.tile(y, C)
        
        return (self.Row, self.Col)

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
                
        Jac = np.zeros(((N-1)*(S*C*2 + C*P)))

        for k in range(N-1):
            
            if self.intergration_method == 'backward euler':
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun(x[S*(k+1):S*(k+2)], (x[S*(k+1):S*(k+2)]-x[S*k:S*(k+1)])/h, par, a[k+1])
                for m in range(C):
                    Jac[k*(S*C*2 + C*P) + m*(2*S + P): k*(S*C*2 + C*P) + m*(2*S + P) + S ] = -dfdxdot[m,:S]/h
                    Jac[k*(S*C*2 + C*P) + m*(2*S + P) + S : k*(S*C*2 + C*P) + m*(2*S + P) + 2*S ] = dfdx[m,:S] + dfdxdot[m,:S]/h
                    Jac[k*(S*C*2 + C*P) + m*(2*S + P) + 2*S : k*(S*C*2 + C*P) + m*(2*S + P) + 2*S + P ] = dfdp[m,:P]

            elif self.intergration_method == 'midpoint':
                f, dfdx, dfdxdot, dfdp = self.dynamic_fun((x[S*k:S*(k+1)] + x[S*(k+1):S*(k+2)])/2, (x[S*(k+1):S*(k+2)]-x[S*k:S*(k+1)])/h, par, (a[k] + a[k+1])/2)
                for m in range(C):
                    Jac[k*(S*C*2 + C*P) + m*(2*S + P): k*(S*C*2 + C*P) + m*(2*S + P) + S ] = dfdx[m,:S]/2 -dfdxdot[m,:S]/h
                    Jac[k*(S*C*2 + C*P) + m*(2*S + P) + S : k*(S*C*2 + C*P) + m*(2*S + P) + 2*S ] = dfdx[m,:S]/2 + dfdxdot[m,:S]/h
                    Jac[k*(S*C*2 + C*P) + m*(2*S + P) + 2*S : k*(S*C*2 + C*P) + m*(2*S + P) + 2*S + P ] = dfdp[m,:P]
                
            else:
                print 'Do not have the Intergration Method code'
           
        return Jac
        
    def dynamic_fun(self, x, xdot, p, a):
        
        l_L = self.parameter[0]
        d_L = self.parameter[2]
        d_T = self.parameter[4]
        m_L = self.parameter[5]/self.scaling
        m_T = self.parameter[7]/self.scaling
        
        I_L = self.parameter[8]/self.scaling
        I_T = self.parameter[10]/self.scaling
        
        g = 9.81
        
#        l_L = 0.878
#        I_T = 2.4811635063274338/self.scaling
#        g = 9.81
#        d_T = 0.31449182352868588
#        d_L = 0.57159831527368432
#        I_L = 1.7991738234932386/self.scaling
#        m_L = 48.830566595091696/self.scaling
#        m_T = 32.125955184002557/self.scaling
        
        theta_a = x[0]
        theta_h = x[1]
        omega_a = x[2]
        omega_h = x[3]
        
        theta_a_dot = xdot[0]
        theta_h_dot = xdot[1]
        omega_a_dot = xdot[2]
        omega_h_dot = xdot[3]
        
        ref_a = 0
        ref_h = 0
               
        k00 = p[0]
        k01 = p[1]
        k02 = p[2]
        k03 = p[3]
        k10 = p[4]
        k11 = p[5]
        k12 = p[6]
        k13 = p[7]
        
        f = np.zeros((self.num_conspernode))
        dfdx = np.zeros((self.num_conspernode, self.num_states))
        dfdxdot = np.zeros((self.num_conspernode, self.num_states))
        dfdp = np.zeros((self.num_conspernode, self.num_par))
        

        f[0] = omega_a - theta_a_dot
    
        dfdx[0,2] = 1
        dfdxdot[0,0] = -1
        
        f[1] = omega_h - theta_h_dot
        
        dfdx[1,3] = 1
        dfdxdot[1,1] = -1
        
        f[2] = (-k03*omega_h + a*d_L*m_L*cos(theta_a) + a*d_T*m_T*cos(theta_a + theta_h) 
                + a*l_L*m_T*cos(theta_a) + d_L*g*m_L*sin(theta_a) + d_T*g*m_T*sin(theta_a + theta_h) 
                - d_T*l_L*m_T*omega_a**2*sin(theta_h) + d_T*l_L*m_T*(omega_a + omega_h)**2*sin(theta_h) 
                + g*l_L*m_T*sin(theta_a) - k00*(theta_a-ref_a) - k01*(theta_h-ref_h) - k02*omega_a 
                - omega_a_dot*(I_L + I_T + d_L**2*m_L + m_T*(d_T**2 + 2*d_T*l_L*cos(theta_h) + l_L**2)) 
                - omega_h_dot*(I_T + d_T*m_T*(d_T + l_L*cos(theta_h))))
  
        
        dfdx[2,0] = (-a*d_L*m_L*sin(theta_a) - a*d_T*m_T*sin(theta_a + theta_h)
                        - a*l_L*m_T*sin(theta_a) + d_L*g*m_L*cos(theta_a) 
                        + d_T*g*m_T*cos(theta_a + theta_h) + g*l_L*m_T*cos(theta_a) - k00)
                                       
        dfdx[2,1] = (-a*d_T*m_T*sin(theta_a + theta_h) + d_T*g*m_T*cos(theta_a + theta_h) 
                    - d_T*l_L*m_T*omega_a**2*cos(theta_h) + 2*d_T*l_L*m_T*omega_a_dot*sin(theta_h) 
                    + d_T*l_L*m_T*omega_h_dot*sin(theta_h) + d_T*l_L*m_T*(omega_a + omega_h)**2*cos(theta_h) - k01)
                                                 
        dfdx[2,2] = -2*d_T*l_L*m_T*omega_a*sin(theta_h) + d_T*l_L*m_T*(2*omega_a + 2*omega_h)*sin(theta_h) - k02
        
        dfdx[2,3] = -k03 + d_T*l_L*m_T*(2*omega_a + 2*omega_h)*sin(theta_h)
        
        dfdxdot[2,2] = -I_L - I_T - d_L**2*m_L - m_T*(d_T**2 + 2*d_T*l_L*cos(theta_h) + l_L**2)
        dfdxdot[2,3] = -I_T - d_T*m_T*(d_T + l_L*cos(theta_h))
        
        
        dfdp[2,0] = -theta_a
        dfdp[2,1] = -theta_h
        dfdp[2,2] = -omega_a
        dfdp[2,3] = -omega_h
            
        f[3] =  (a*d_T*m_T*cos(theta_a + theta_h) + d_T*g*m_T*sin(theta_a + theta_h) 
                - d_T*l_L*m_T*omega_a**2*sin(theta_h) - (I_T + d_T**2*m_T)*omega_h_dot 
                - (I_T + d_T*m_T*(d_T + l_L*cos(theta_h)))*omega_a_dot 
                - (k10*(theta_a-ref_a) + k11*(theta_h-ref_h) + k12*omega_a + k13*omega_h))
        
        
        dfdx[3,0] = -k10 - a*d_T*m_T*sin(theta_a + theta_h) + d_T*g*m_T*cos(theta_a + theta_h)
        
        dfdx[3,1] = (-a*d_T*m_T*sin(theta_a + theta_h) + d_T*g*m_T*cos(theta_a + theta_h) 
                    - d_T*l_L*m_T*omega_a**2*cos(theta_h) + d_T*l_L*m_T*omega_a_dot*sin(theta_h) - k11)
                                                                               
        dfdx[3,2] = -2*d_T*l_L*m_T*omega_a*sin(theta_h) - k12
        dfdx[3,3] = -k13
            
        dfdxdot[3,2] = -I_T - d_T*m_T*(d_T + l_L*cos(theta_h))
        dfdxdot[3,3] = -I_T - d_T**2*m_T
        
        dfdp[3,4] = -theta_a
        dfdp[3,5] = -theta_h
        dfdp[3,6] = -omega_a
        dfdp[3,7] = -omega_h
#        
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