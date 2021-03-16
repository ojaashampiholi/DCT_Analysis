# Importing the Libraries necessary for the program
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from dct import DCT
from quantization import quantizationClass

dct_Object = DCT()
quantization_Object = quantizationClass()

# This function takes one image channel as the input and returns the 
# DCT processed output for the same.
# We use DCT on the Image to convert it from Spectral Domain 
# (Y-Cb-Cr Channels) to its equivalent Frequency Domain.

def compressedInformation(Y, N = 8, compressionPercentage = 50, restrictingFactor = 5):
    h, w = Y.shape
    # N, restrictingFactor = 8, 5
    # dctMatrix = generate_N_Point_DCT(N)
    # Q = getQuantizationMatrix(compressionPercentage)
    outArr = np.random.randn(restrictingFactor,restrictingFactor,1)
    # print(outArr.shape)
    for i in range(0,h-N,N):
        for j in range(0,w-N,N):
            tempImg = Y[i:i+N, j:j+N]
            D = dct_Object.calculateDCT(tempImg)
            C = quantization_Object.quantizedOutputs(D)
            C = C[:restrictingFactor,:restrictingFactor].reshape(restrictingFactor, restrictingFactor,1)
            outArr = np.concatenate((outArr, C), axis = 2)
            # print(D.shape)
    outArr = outArr[:,:,1:]
    # print(outArr.shape)
    return outArr

# This function calculates the compression ratio between the 
# input and processed output information.

def getCompressionRate(Y, processedY):
    a,b = Y.shape
    inputImagePixels = a*b*3
    a1, b1, c1 = processedY.shape
    outputImagePixels = a1*b1*c1*3
    return np.round(1-(outputImagePixels/inputImagePixels),2)

# This function takes an image and the resizing heights 
# and weights as the input and returns processed images.

def processImage(url, resize_height=480, resize_width=480):
    BGRImage = cv2.imread(url)
    print("------------------------------------------------------------")
    print("Input Image Size", BGRImage.shape)
    # print("------------------------------------------------------------")
    # plt.imshow(BGRImage, interpolation='nearest')
    # plt.axis('off')
    # plt.show()
    BGRImage = cv2.resize(BGRImage, (resize_height, resize_width)) 
    print("------------------------------------------------------------")
    print("Shape of the Resized Image is", BGRImage.shape)
    YCrCbImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2YCR_CB)
    Y, Cb, Cr = YCrCbImage[:,:,0], YCrCbImage[:,:,1], YCrCbImage[:,:,2]
    Y = np.array(Y).astype(np.int16)
    Cb = np.array(Cb).astype(np.int16)
    Cr = np.array(Cr).astype(np.int16)
    Y, Cb, Cr = Y - 128, Cb - 128, Cr - 128

    processedY = compressedInformation(Y)
    print("------------------------------------------------------------")
    print("Shape of Processed Y Matrix is", processedY.shape)
    processedCb = compressedInformation(Cb)
    print("------------------------------------------------------------")
    print("Shape of Processed Cb Matrix is", processedCb.shape)
    processedCr = compressedInformation(Cr)
    print("------------------------------------------------------------")
    print("Shape of Processed Cr Matrix is", processedCr.shape)

    print("------------------------------------------------------------")
    print("Compression Rate Achieved is", getCompressionRate(Y, processedY))

    print("------------------------------------------------------------")
    print("Saving the Processed Image Channels to npy files")
    np.save("processedY.npy", processedY)
    np.save("processedCb.npy", processedCb)
    np.save("processedCr.npy", processedCr)
    print("------------------------------------------------------------")
    print("Processing Successful")
    return processedY, processedCb, processedCr


if __name__== "__main__":
    
    if(len(sys.argv) != 4):
        raise Exception('Format: python dct.py image_url resize_height resize_width') 
    url, resize_height, resize_width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print("------------------------------------------------------------")
    print("Processing with Input Image")
    _,_,_ = processImage(url, resize_height, resize_width)

