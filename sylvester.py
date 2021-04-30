# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 20:48:10 2021

@author: Rick Wang
"""
from math import floor
import numpy as np
import scipy

#def rtrsyct(A,B,C,uplo,blks):
#    '''
#    A (M*M) and B(N*N) in quasitrangular Schur form
#    C (M*N) dense matrix
#    '''
#    M,N = A.shape[0], B.shape[0]
#    midM,midN = int(M/2),int(N/2)
#    
#    if (M>=1 and M<=blks) and (N>=1 and N<=blks):
#        trsyct(A,B,C,uplo)
#        return
#    
#    if uplo==1:
#        if N>=1 and N<=midM:
#            k = split_matrix(A)
#            A11,A12,A22=A[:k+1,:k+1],A[:k+1,k+1:],A[k+1:,k+1:]
#            C1,C2=C[:k+1,:],C[k+1:,:]
#            rtrsyct(A22,B,C2,1,blks)
#            X2 = C2
#            gemm(-A12,X2,C1)
#            rtrsyct(A11,B,C1,1,blks)
#            X1 = C1
#        elif M>=1 and M<=midN:
#            k = split_matrix(B)
#            B11,B12,B22=B[:k+1,:k+1],B[:k+1,k+1:],B[k+1:,k+1:]
#            C1,C2=C[:,:k+1],C[:,k+1:]
#            rtrsyct(A,B11,C1,1,blks)
#            X1 = C1
#            gemm(X1,B12,C2)
#            rtrsyct(A,B22,C2,1,blks)
#            X2 = C2
#        else:
#            kA = split_matrix(A)
#            kB = split_matrix(B)
#            A11,A12,A22=A[:kA+1,:kA+1],A[:kA+1,kA+1:],A[kA+1:,kA+1:]
#            B11,B12,B22=B[:kB+1,:kB+1],B[:kB+1,kB+1:],B[kB+1:,kB+1:]
#            C11,C12,C21,C22=C[:kA+1,:kB+1],C[:kA+1,kB+1:],C[kA+1:,:kB+1],C[kA+1:,kB+1]
#            rtrsyct(A22,B11,C21,1,blks)
#            X21 = C21
#            gemm(X21,B12,C22)
#            gemm(-A12,X21,C11)
#            rtrsyct(A22,B22,C22,1,blks)
#            X22 = C22
#            rtrsyct(A11,B11,C11,1,blks)
#            X11 = C11
#
#            gemm(-A12,X22,C12)
#            gemm(X11,B12,C12)
#            rtrsyct(A11,B22,C12,1,blks)
#            X12 = C12
            
def rtrsyct(A,B,C,uplo,blks):
    '''
    A (M*M) and B(N*N) in quasitrangular Schur form
    C (M*N) dense matrix
    '''
    M,N = A.shape[0], B.shape[0]
    midM,midN = int(M/2),int(N/2)
    
    if (M>=1 and M<=blks) and (N>=1 and N<=blks):
        X = trsyct(A,B,C,uplo)
        return X
    
    if uplo==1:
        if N>=1 and N<=midM:
            k = split_matrix(A)
            A11,A12,A22=A[:k+1,:k+1],A[:k+1,k+1:],A[k+1:,k+1:]
            C1,C2=C[:k+1,:],C[k+1:,:]
            X2 = rtrsyct(A22,B,C2,1,blks)
            #C1 = C1-A12@X2
            #X1 = rtrsyct(A11,B,C1,1,blks)
            X1 = rtrsyct(A11,B,C1-A12@X2,1,blks)
            X=np.concatenate((X1,X2),axis=0)
        elif M>=1 and M<=midN:
            k = split_matrix(B)
            B11,B12,B22=B[:k+1,:k+1],B[:k+1,k+1:],B[k+1:,k+1:]
            C1,C2=C[:,:k+1],C[:,k+1:]
            X1 = rtrsyct(A,B11,C1,1,blks)
            #C2 = C2+X1@B12
            #X2 = rtrsyct(A,B22,C2,1,blks)
            X2 = rtrsyct(A,B22,C2+X1@B12,1,blks)
            X=np.concatenate((X1,X2),axis=1)
        else:
            kA = split_matrix(A)
            kB = split_matrix(B)
            A11,A12,A22=A[:kA+1,:kA+1],A[:kA+1,kA+1:],A[kA+1:,kA+1:]
            B11,B12,B22=B[:kB+1,:kB+1],B[:kB+1,kB+1:],B[kB+1:,kB+1:]
            C11,C12,C21,C22=C[:kA+1,:kB+1],C[:kA+1,kB+1:],C[kA+1:,:kB+1],C[kA+1:,kB+1:]
            X21 = rtrsyct(A22,B11,C21,1,blks)
            #C22 = C22+X21@B12
            #C11 = C11-A12@X21
            #X22 = rtrsyct(A22,B22,C22,1,blks)
            #X11 = rtrsyct(A11,B11,C11,1,blks)
            
            #can parallelized
            X22 = rtrsyct(A22,B22,C22+X21@B12,1,blks)
            X11 = rtrsyct(A11,B11,C11-A12@X21,1,blks) 
            
            #C12 = C12-A12@X22+X11@B12
            #X12 = rtrsyct(A11,B22,C12,1,blks)
            X12 = rtrsyct(A11,B22,C12-A12@X22+X11@B12,1,blks)
            X = np.concatenate((np.concatenate((X11,X12),axis=1),np.concatenate((X21,X22),axis=1)),axis=0)
    
    return X



#transform the base case to tensor structure
def trsyct(A,B,C,uplo):
    M,N = A.shape[0],B.shape[0]
    Z = np.kron(np.eye(N),A)-np.kron(B.T,np.eye(M))
    x = np.linalg.solve(Z,C.flatten('F'))
    return x.T.reshape((M,N),order='F')

def solve_sylvester(A,B,C):
    #decompose them in real schur form\
    T_A,Z_A = scipy.linalg.schur(A,output='real')
    T_B,Z_B = scipy.linalg.schur(B,output='real')
    
    X = rtrsyct(T_A,T_B,Z_A.T@C@Z_B,1,blks=4)
    
    return Z_A@X@Z_B.T


#find split point
def split_matrix(M):
    n = M.shape[0]
    m = floor(n / 2) - 1 
    if M[m+1, m] != 0.0:
        m += 1 

    return m