""" 
This module takes in images, analysese them and returns results

"""

import CalcLine as calc
import parseOutput as pOutput
import miscFuncs

import cv2
import numpy as np
from matplotlib import pyplot as plt 
from math import pi
import math # for atan2(line angle)
from time

from cluster import * # for line clustering


if (miscFuncs.isPi()):
    # printing defines
    INVERTED = False   
    BLURRED = False
    CLOSED = False
    EDGES = False
    LINES = False    
else:
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

def holderFunc(cur, score, fps):
    if cur is None:
        print("cur is none, returning")
        return 0, score

    cur = preprocess(cur)
    plt, cur = edgeDetection(cur)
    score = lineDetection(cur, score, fps)

    # Print canny edge results
    if EDGES:
        plt.show(100)

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

    if miscFuncs.isPi() is not True:
        plt.subplot(121),plt.imshow(cur,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(cur,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    return plt, cur

# Calculate degrees for the input line coordinates
def lineAngle(x1, x2, y1, y2):
    deg = math.degrees(math.atan2(y1 - y2, x1 - x2))
    res = (deg + 360) % 360

    if res>360:
        print("ERROR...: Angle:")
        time.sleep(1000)
    return res

# Cluster the angles(degrees) of identified lines
def aggLines(angles, aggThres):
    dcPlaces = 0
    rtnClusters = []

    if angles is None:
        print("no angles returning\n\n")
        return

    # Cluster angles if outside of difference threshold and calculate/return means for each cluster
    cl = HierarchicalClustering(angles,lambda x,y: abs(x-y))

    clusters = ([np.mean(cluster) for cluster in cl.getlevel(aggThres)])

    # Round/convert floats to ints
    for i in clusters:
        rtnClusters.append(int(round(i, dcPlaces)))

    return rtnClusters

def lineDetection(cur, score, fps):    
    # Detect lines
    rho = 4 # Keep high to prevent multiple paralellel lines 
    theta = np.pi/180
    lineThreshold = 40
    minLineLength = 40
    maxLineGap = 10

    lines = cv2.HoughLinesP(image = cur, rho=rho, theta=np.pi/180, threshold=lineThreshold, lines=np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)

    angles = []
    intercepts = []
    lineList = []

    if lines is not None:       
        print("Found this many lines: " + str(lines.shape[0]))

        score[1]+=1

        a,b,c = lines.shape

        cur = cv2.cvtColor(cur, cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1,y1,x2,y2 = line[0]

            # Store line angle to list 
            angle = round(lineAngle(x1, x2, y1, y2),1)

            angles.append(angle)
            # print("This is line angle: " + str(lineAngle(x1, x2, y1, y2)))

            slope = calc.calcSlope((x1, y1), (x2, y2))
            intercept = calc.calcIntercept(slope, (x2, y2))
            intercepts.append(intercept)

            lineDict = {"P1": (x1, y1), "P2": (x2, y2), "Slope": slope, "Intercept": intercept, "Angle": angle}
            lineList.append(lineDict)



        # Cluster line angles to sort misc and perpendicular lines out, return cluster list
        angleThresh = 10.0
        angleClusters = aggLines(angles, angleThresh)
        print(angleClusters)

        pOutput.directionParse(angleClusters, cur)

        # interceptThresh = 150.0
        # interceptClusters = aggLines(intercepts, interceptThresh)

        # display lines in different colors based on their clusters
        lineColors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,255,0), (100,255,0), (255,100,0), (100,0,255), (100,255,255), (255,255,100)]

        for i in range(len(lineList)):
            localAng = lineList[i]["Angle"]
            # localIntercept = lineList[i]["Intercept"]

            for v in range(len(angleClusters)):
                if localAng <= angleClusters[v]+ angleThresh and localAng >= angleClusters[v]-angleThresh:                    
     
                    cv2.line(cur,(lineList[i]["P1"]),(lineList[i]["P2"]),pOutput.rgb2bgr(pOutput.hsv2rgb(localAng/80, 1, 1)),2)

            # for v in range(len(interceptClusters)):
            #     if localIntercept <= interceptClusters[v]+ interceptThresh and localIntercept >= interceptClusters[v]- interceptThresh:
    
        # Print images to screen
        if LINES:
            cv2.imshow("houghlines", cur)
        cv2.waitKey(3)

    else: 
        print("exception")

        # Print images to screen
        if LINES:
            cv2.imshow("houghlines", cur)
            cv2.waitKey(3)
   
    # # invert the image
    # newImg =  cv2.bitwise_not(newImg)
    return score


