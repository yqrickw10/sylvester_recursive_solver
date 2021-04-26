# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 20:48:10 2021

@author: Rick Wang
"""
import numpy as np
import scipy

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
            A11,A12,A22=A[0:midM,0:midN],A[0:midM,midN:N],A[midM:M,midN:N]
            C1,C2=C[0:midM],C[midM:M]
            X2 = rtrsyct(A22,B,C2,1,blks)
            C1 = C1-A12@X2
            X1 = rtrsyct(A11,B,C1,1,blks)
            X=np.concatenate((X1,X2),axis=0)
        elif M>=1 and M<=midN:
            B11,B12,B22=B[0:midM,0:midN],B[0:midM,midN:N],B[midM:M,midN:N]
            C1,C2=C[:,0:midN],C[:,midN:N]
            X1 = rtrsyct(A,B11,C1,1,blks)
            C2 = C2+X1@B12
            X2 = rtrsyct(A,B22,C2,1,blks)
            X=np.concatenate((X1,X2),axis=1)
        else:
            A11,A12,A22=A[0:midM,0:midN],A[0:midM,midN:N],A[midM:M,midN:N]
            B11,B12,B22=B[0:midM,0:midN],B[0:midM,midN:N],B[midM:M,midN:N]
            C11,C12,C21,C22=C[0:midM,0:midN],C[0:midM,midN:N],C[midM:M,0:midN],C[midM:M,midN:N]
            X21 = rtrsyct(A22,B11,C21,1,blks)
            C22 = C22+X21@B12
            C11 = C11-A12@X21
            X22 = rtrsyct(A22,B22,C22,1,blks)
            X11 = rtrsyct(A11,B11,C11,1,blks)
            C12 = C12-A12@X22+X11@B12
            X12 = rtrsyct(A11,B22,C12,1,blks)
            X = np.concatenate((np.concatenate((X11,X12),axis=1),np.concatenate((X21,X22),axis=1)),axis=0)


#transform the base case to tensor structure
def trsyct(A,B,C,uplo):
    M,N = A.shape[0],B.shape[0]
    Z = np.kron(np.eye(N),A)-np.kron(B.T,np.eye(M))
    x = np.linalg.solve(Z,C.flatten('F').reshape((-1,1)))
    return x.reshape((M,N),order='F')

def solve_sylvester(A,B,C):
    #decompose them in real schur form\
    T_A,Z_A = scipy.linalg.schur(A,output='real')
    T_B,Z_B = scipy.linalg.schur(B,output='real')
    
    X = rtrsyct(T_A,T_B,Z_A.T@C@Z_B,1,blks=4)
    
    return Z_A@X@Z_B.T
    
    