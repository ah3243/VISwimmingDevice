""" 
This module takes in images, analysese them and returns results

"""


import cv2
import numpy as np
from matplotlib import pyplot as plt 
from math import pi

# printing defines
INVERTED = False   
BLURRED = False
CLOSED = False
EDGES = False
LINES = True

# Kernel definition for dilation and errosion
kernel = np.ones((5,5), np.uint8)

# Threshold image values to remove additional noise
thresh = 100
maxVal = 255

# Standard image size
imgSize = (500,300)

def holderFunc(cur, score):
    if cur is None:
        return 0, score

    cur = preprocess(cur)
    plt, cur = edgeDetection(cur)
    score = lineDetection(cur, score)

    # Print canny edge results
    if EDGES:
        plt.show(100)

    cv2.destroyAllWindows()

    return 1, score


""" Resizes images to a common size """
def resizeImg(img):
    # Resize image to common dimensions
    if img.shape[0]<img.shape[1]:
        cur = cv2.resize(img, (500,300), interpolation = cv2.INTER_LINEAR)
    else:
        cur = cv2.resize(img, (300,500), interpolation = cv2.INTER_LINEAR) 
    return cur

def preprocess (cur):

    # convert to grayscale is RGB
    if(len(cur.shape)>2):
        cur = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)

    # Standardise the image size
    cur = resizeImg(cur)

    # invert the image
    cur =  cv2.bitwise_not(cur)

    if INVERTED:
        cv2.imshow("inverted", cur)

    # Apply a gausian or bilateral blur
    cur = cv2.GaussianBlur(cur, (3, 13), 0)
    # cur = cv2.bilateralFilter(cur, 9, 75, 75)

    if BLURRED:
        cv2.imshow("blurred", cur)

    # Perform Dilation and Erosion across several iterations to remove noise
    cur = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    if CLOSED:
        cv2.imshow("closed", cur)    

    # # th, cur = cv2.threshold(cur, thresh, maxVal, cv2.THRESH_BINARY)
    # th, cur = cv2.threshold(cur, thresh, maxVal, cv2.THRESH_TOZERO)
    # cv2.imshow("threshed", cur)

    return cur

def edgeDetection(cur):    
    # apply and plot canny edge detector
    cur = cv2.Canny(cur, 20, 60)

    plt.subplot(121),plt.imshow(cur,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(cur,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    return plt, cur

def lineDetection(cur, score):    

    # Detect lines
    rho = 1
    theta = np.pi/180
    lineThreshold = 100
    minLineLength = 80
    maxLineGap = 40

    lines = cv2.HoughLinesP(image = cur, rho=rho, theta=np.pi/180, threshold=lineThreshold, lines=np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)

    try:
        print("Found this many lines: " + str(lines.shape[0]))

        if lines.shape[0]>0:
            score[1]+=1
            print("Got another one..")

        a,b,c = lines.shape

        print(lines.shape)

        cur = cv2.cvtColor(cur, cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(cur,(x1,y1),(x2,y2),(0,255,0),2)

            # Print images to screen
            if LINES:
                cv2.imshow("houghlines", cur)
            cv2.waitKey(1)

    except:
        print("exception")
   
    # # invert the image
    # newImg =  cv2.bitwise_not(newImg)
    return score


def getAngles():
    pass


