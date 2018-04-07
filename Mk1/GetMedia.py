
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
import CalcLine as calc

import time 
import cv2

# Possible modes:
# 1: Local Image Import
# 2: Local Video Import
# 3: Video Stream
# 4: TEST

MODE = 2


# A score of the number of frames where lines were found
SCORE = False # print the 'score' for line detection

# Local file locations
localImgLocation = '../TrainingMedia/WebImages/web_pool_9.jpg'
localVideoLocation = '../TrainingMedia/CurtainVideo/piOutput_end.mov'


def importLocalImg(loc):
    """ Imports + returns locally stored images in grayscale """
    # read in the training image
    img = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)

    # Pass to analysis function
    res, score = base.holderFunc(img, 0)

    print("This is the output.. " + str(res))

    return res


def importLocalVideo(loc):
    """ Imports + returns video file for processing """
    score = [0,0]

    cap = cv2.VideoCapture(loc)
    
    start = time.time()
    count =0
    fps =0
    while(cap.isOpened()):
        fps
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
        res, score = base.holderFunc(frame, score, fps)

        if SCORE:
            # Calculate percentage score for number of frames with detected lines
            scoreCal(score)

        # Calculate the approximate Frames per second
        count = count+1
        intTime = time.time()
        fps = count/(intTime- start)
        print ("This is the fps: ", fps)

    # clean up
    cap.release()
    cv2.destroyAllWindows()
    return True   

# Calculate a detection score assuming a line is always visible in training media
def scoreCal(score):
    pcnt = (int(score[1])/int(score[0]))*100
    print("Frames: " + str(score[0]) + " hits: " + str(score[1]) + " Percentage right: " + str(pcnt)) 

# Function used to route program depending on selected Flags
def holderFunc():
    try:
        if MODE == 1:
            # Import image
            rtnVal = importLocalImg(localImgLocation)
        elif MODE == 2:
            rtnVal = importLocalVideo(localVideoLocation) 
            if rtnVal == False:
                print("Exiting")
                exit(0) 
        elif MODE == 3:
            print(calc.calcSlope((0,0), (2,2)))
            pass  
        elif MODE == 4:
            P1 = (0,0)
            P2 = (1,2)
            slope = 0.0
            slope = calc.calcSlope(P1, P2)
            intercept = calc.calcIntercept(slope, P2)
            print("This is the slope: ", str(slope), "and the intercept: ", str(intercept))

            # Store these values in a dictionary list
            lineVals = []
            one = {"P1": P1, "P2": P2, "Slope": slope, "Intercept": intercept }
            two = {"P1": P1, "P2": P2, "Slope": 0.00, "Intercept": 0.000 }

            lineVals.append(one)
            lineVals.append(two)
            print(lineVals[1]["Slope"])
    except KeyboardInterrupt:
        print("\n\nExiting..")
        cv2.waitKey(1000)



holderFunc()
