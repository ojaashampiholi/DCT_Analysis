import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

class quantizationClass():

    def __init__(self):
        self.count = 0

    # The (8*8) Quantization Matrix which is predefined here is the standard
    # matrix that is used in JPEG Compression.
    # This function calculates the Quantization Matrix depending on the 
    # percentage of compressed image size required.
    # If the desired percentage is 90%, more information is retained and the 
    # compression ratio turns out to be less.
    # If the desired percentage is 10%, more compression takes place at the
    # cost of loss of information.
    # 50% is the Standard value that is chosen for JPEG compression, hence same 
    # value has been used here.

    def getQuantizationMatrix(self, requiredQualityLevel = 50):
        Q = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])
        if requiredQualityLevel == 50:
            return Q
        elif requiredQualityLevel>50:
            Q = (Q * ((100-requiredQualityLevel)/50)).astype('int')
            Q = np.where(Q>255,255,Q)
            return Q
        else:
            Q = (Q * (50/requiredQualityLevel)).astype('int')
            Q = np.where(Q>255,255,Q)
            return Q

    # This function gives us the quantized output which can be used to 
    # find the relevant compressions in the image.

    def quantizedOutputs(self, D):
        Q = self.getQuantizationMatrix()
        C = D//Q
        return C