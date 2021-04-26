# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:49:08 2021

@author: Rick Wang
"""

import numpy as np
import scipy.linalg
import sylvester as syl

def Test():
    A,B,C = TestCase(case=2)
    x1 = scipy.linalg.solve_sylvester(A, -B, C)
    x2 = syl.solve_sylvester(A,B,C)
    
    print(np.allclose(A.dot(x1) - x1.dot(B), C))
    print(np.allclose(A.dot(x2) - x2.dot(B), C))
    
    return 0

def TestCase(case):
    if case == 1:
        A = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])
        B = np.array([[-1]])
        C = np.array([[1],[2],[3]])
    
    if case == 2:
        A,B,C = GenerateRandomTestCase()
    
    
    return A,B,C

#generate random test case for sylvester
def GenerateRandomTestCase():
    
    M,N = np.random.randint(low=10,high=21,size=2)
    A = np.random.rand(M,M)
    B = np.random.rand(N,N)
    C = np.random.rand(M,N)
    return A,B,C

Test()