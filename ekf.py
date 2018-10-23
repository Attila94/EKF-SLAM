# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:09:26 2018

@author: Attila
"""

import numpy as np

def predict(mu, cov, u, Rt):
    n = len(mu)
    
    # Define motion model f(mu,u)
    [dtrans, drot1, drot2] = u
    motion = np.array([[dtrans*np.cos(mu[2][0]+drot1)],
              [dtrans*np.sin(mu[2][0]+drot1)],
              [drot1 + drot2]])
    F = np.append(np.eye(3),np.zeros((3,n-3)),axis=1)
    
    # Predict new state
    mu_bar = mu + (F.T).dot(motion)
    
    # Define motion model Jacobian
    J = np.array([[0,0,-dtrans*np.sin(mu[2][0]+drot1)],
               [0,0,dtrans*np.cos(mu[2][0]+drot1)],
               [0,0,0]])
    G = np.eye(n) + (F.T).dot(J).dot(F)
    
    # Predict new covariance
    cov_bar = G.dot(cov).dot(G.T) + (F.T).dot(Rt).dot(F)
    
    print('Predicted location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu_bar[0][0],mu_bar[1][0],mu_bar[2][0]))
    return mu_bar, cov_bar

def update(mu,cov,obs,c_prob,Qt):
    N = len(mu)
    
    for [r, theta, j] in obs:
        j = int(j)
        # if landmark has not been observed before
        if cov[2*j+3][2*j+3] >= 1e6 and cov[2*j+4][2*j+4] >= 1e6:
            # define landmark estimate as current measurement
            mu[2*j+3][0] = mu[0][0] + r*np.cos(theta+mu[2][0])
            mu[2*j+4][0] = mu[1][0] + r*np.sin(theta+mu[2][0])

        
        # if landmark is static
        if c_prob[j] >= 0.5:
            # compute expected observation
            delta = np.array([mu[2*j+3][0] - mu[0][0], mu[2*j+4][0] - mu[1][0]])
            q = delta.T.dot(delta)
            sq = np.sqrt(q)
            z_theta = np.arctan2(delta[1],delta[0])
            z_hat = np.array([[sq], [z_theta-mu[2][0]]])
            
            # calculate Jacobian
            F = np.zeros((5,N))
            F[:3,:3] = np.eye(3)
            F[3,2*j+3] = 1
            F[4,2*j+4] = 1
            H_z = np.array([[-sq*delta[0], -sq*delta[1], 0, sq*delta[0], sq*delta[1]],
                            [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
            H = 1/q*H_z.dot(F)
    
            # calculate Kalman gain        
            K = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov).dot(H.T)+Qt))
            
            # calculate difference between expected and real observation
            z_dif = np.array([[r],[theta]])-z_hat
            z_dif = (z_dif + np.pi) % (2*np.pi) - np.pi
            
            # update state vector and covariance matrix        
            mu = mu + K.dot(z_dif)
            cov = (np.eye(N)-K.dot(H)).dot(cov)
    
    print('Updated location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu[0][0],mu[1][0],mu[2][0]))
    return mu, cov, c_prob