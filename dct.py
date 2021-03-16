import numpy as np
import cv2
# import matplotlib.pyplot as plt
import sys


class DCT():
    def __init__(self):
        self.count = 0

    # The Function generates a N point DCT Matrix which is based on type 2 DCT
    # whose formula can be found as Equation 4 (pg 2) in the Paper.
    # We use N = 8 when we are dealing with JPEG Compression, hence the same 
    # value has been used here.

    def generate_N_Point_DCT(self, N = 8, roundingFactor = 4):
        D = np.random.randn(N,N)
        for i in range(N):
            for j in range(N):
                if i==0:
                    D[i,j] = np.round(1/np.sqrt(N),roundingFactor)
                else:
                    D[i,j] = np.round(np.sqrt(2/N) * np.cos(((2*j + 1)*i*np.pi)/(2*N)),roundingFactor)
        return D

    # This function calculates the DCT transform using the DCT Matrix using formula 5 (pg 3 in the paper).

    def calculateDCT(self, a):
        dctMatrix = self.generate_N_Point_DCT()
        return np.round(dctMatrix @ a @ dctMatrix.T,4)