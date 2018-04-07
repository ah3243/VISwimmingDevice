"""
This Module intakes the dominant line angle and parses the output either for visual, audio or vibration
"""

import cv2
from numpy import interp 
import colorsys

def directionParse(angles, img):
    """Main holder function for handling parsing and outputting of direction commands

    Arguments:
        angles {[List[int]]} -- an array of the clutered angles
        img {[Mat]} -- Allows 'virtual LEDs' to be draw on the output display for testing and development
    """
    if len(angles)>1:
    
        detectEndOfLane(angles)


    # The direction of travel
    goal = 90
    dirThresh = 45

    # Make sure all angles are 0-180 degrees
    newAngles = []
    for i in angles:
        angle = i
        if angle > 180:
            angle = i -180
    
        if angle >90 and angle< (90+ dirThresh) and abs(angle-90)<=dirThresh:
            drawLights(angle - 90, img, dirThresh)  

        elif angle <90 and angle > (90-dirThresh) and abs(angle-90)<=dirThresh:
            drawLights(angle - 90, img, dirThresh)
        elif angle == 90:
            print("angle ==90")
        else: 
            print("Not within limits: ", abs(angle), " angle: ", angle )

def drawLights(direction, img, dirThresh):
    """Intakes direction and image, the direction is parsed and the virtual output drawn on the displayed image.    
    Arguments:
        direction {[type]} -- [description]
        img {[type]} -- [description]
    """
    imgH = int(img.shape[0]/6)
    imgW = int(img.shape[1]/10)

    # Remap direciton values to 10-100 to allow easier generation of outputs
    newVal = int(interp(abs(direction),[0,dirThresh],[10,100]))

    # Initiate placeholder variables
    redColor, greenColor = ((0,0,0),(0,0,0))

    # Calculate the bgr colors based on the intensity of the angle deviation
    if direction >0:        
        redColor = rgb2bgr(hsv2rgb(0, 1, newVal/100))
    elif direction<0:
        greenColor = rgb2bgr(hsv2rgb(122/360, 1, newVal/100))

    # Draw circles on displayed screen
    cv2.circle(img, (imgW, imgH), 50, greenColor, -1)
    cv2.circle(img, (imgW*9, imgH), 50, redColor, -1)


# Convert HSV into denormalised rgb
def hsv2rgb(h,s,v):
    return tuple(int(i*255) for i in colorsys.hsv_to_rgb(h,s,v))

# Rearrange RGB to BGR
def rgb2bgr(rgb):
    r, g, b = rgb
    return (int(b),int(g),int(r))

def displayLights(direction):
    """Intakes -dirThreshold <-> +dirThreshold value and parses it to allow LED, vibration or audio output

    Arguments:
        direction {[float]} -- a negative or positive value within a range determined by the dirThreshold. 
        By limiting the angles with which we respond provides greater range to allow fine adjustment for a stright
        swim.
    """

def detectEndOfLane(angles):
    distThresh = 15
    if abs(angles[0]-angles[1])>distThresh:
        print("End of lane detected..\n\n\n")

