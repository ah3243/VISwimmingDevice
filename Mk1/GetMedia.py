"""
    This module imports/handles images or video and passes it through to the BaseModule for analysis, then
    handles the return value passing it through to the actuation module.
    It has four main sections:
    
    1. Get images
    - import stored Images
    - import stored Video
    - captures live Video Stream

    2. Pass to analysis function
    - Call BaseModule and pass in one image

    3. Handle return value from analysis function
    - Direct return value to actuation module


"""

import BaseModule as base
import cv2
# import numpy as np

# Possible modes:
# 1: Local Image Import
# 2: Local Video Import
# 3: Video Stream

MODE = 2

# Printing flags
SCORE = False # print the 'score' for line detection

# A score of the number of frames where lines were found

localImgLocation = '../TrainingMedia/WebImages/web_pool_9.jpg'
localVideoLocation = '../TrainingMedia/CurtainVideo/piOutput_end.avi'

""" Imports + returns locally stored images in grayscale """
def importLocalImg(loc):
    # read in the training image
    img = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)

    # Pass to analysis function
    res, score = base.holderFunc(img, 0)
    print("This is the output.. " + str(res))

    return res

""" Imports + returns video file for processing """
def importLocalVideo(loc):
    score = [0,0]

    cap = cv2.VideoCapture(loc)
    while(cap.isOpened()):
        
        # increment number of frames imported for percentage score val
        score[0] += 1

        # import image
        ret, frame = cap.read()

        # if frame is null exit
        if frame is None or ret is False:
            print("Frame is none..")        
            cap.release()
            cv2.destroyAllWindows()
            return False

        # Pass image to processing module
        res, score = base.holderFunc(frame, score)

        if SCORE:
            # Calculate percentage score for number of frames with detected lines
            scoreCal(score)

    # clean up
    cap.release()
    cv2.destroyAllWindows()
    return True   

def scoreCal(score):
    pcnt = (int(score[1])/int(score[0]))*100
    print("Frames: " + str(score[0]) + " hits: " + str(score[1]) + " Percentage right: " + str(pcnt)) 

if MODE == 1:
    # Import image
    rtnVal = importLocalImg(localImgLocation)
elif MODE == 2:
    rtnVal = importLocalVideo(localVideoLocation) 
    if rtnVal == False:
        print("Exiting")
        exit(0) 

elif MODE == 3:
  pass  
     


cv2.waitKey(5000)
