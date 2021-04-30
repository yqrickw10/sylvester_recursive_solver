# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:49:08 2021

@author: Rick Wang
"""

import numpy as np
import scipy.linalg
import sylvester as syl

def Test():
    
    #x1 = scipy.linalg.solve_sylvester(A, -B, C)
    #x2 = syl.trsyct(A,B,C,1)
    #print(np.allclose(A.dot(x1) - x1.dot(B), C))
    for case in range(1,5):
        A,B,C = TestCase(case)
        x2 = syl.solve_sylvester(A,B,C)
        assert np.allclose(A.dot(x2) - x2.dot(B), C)==True, "Case "+str(case)+ " not passed"
    
    return 0

def TestCase(case):
    if case == 1:
        A = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])
        B = np.array([[-1]])
        C = np.array([[1],[2],[3]])
    if case == 2:
        A,B,C = GenerateRandomTestCase(10,22)
    elif case == 3:
        A,B,C = GenerateRandomTestCase(22,10)
    elif case == 4:
        A,B,C = GenerateRandomTestCase(15,15)
    
    
    return A,B,C

#generate random test case for sylvester
def GenerateRandomTestCase(M,N):
    
    A = np.random.rand(M,M)
    B = np.random.rand(N,N)
    C = np.random.rand(M,N)
    return A,B,C

if __name__ == "__main__":
    Test()
    print("Everything passed")