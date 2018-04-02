import cv2
import numpy as np
from matplotlib import pyplot as plt 
from math import pi

# read in the training image
img = cv2.imread('../TrainingImages/web_pool_9.jpg', cv2.IMREAD_GRAYSCALE)

# Resize image to common dimensions
if img.shape[0]<img.shape[1]:
    cur = cv2.resize(img, (500,300), interpolation = cv2.INTER_LINEAR)
else:
    cur = cv2.resize(img, (300,500), interpolation = cv2.INTER_LINEAR)    

# invert the image
cur =  cv2.bitwise_not(cur)
cv2.imshow("inverted", cur)

# Apply a gausian blur(anisometric likely)
cur = cv2.GaussianBlur(cur, (3, 13), 0)
# cur = cv2.bilateralFilter(cur, 9, 75, 75)
cv2.imshow("blurred", cur)

kernel = np.ones((5,5), np.uint8)
cur = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, kernel, iterations=4)
cv2.imshow("closed", cur)

thresh = 100
maxVal = 255
# th, cur = cv2.threshold(cur, thresh, maxVal, cv2.THRESH_BINARY)
th, cur = cv2.threshold(cur, thresh, maxVal, cv2.THRESH_TOZERO)
cv2.imshow("threshed", cur)

# apply and plot canny edge detector
cur = cv2.Canny(cur, 20, 60)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cur,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# Detect lines
rho = 1
theta = np.pi/180
lineThreshold = 50
minLineLength = 50
maxLineGap = 1
lines = cv2.HoughLinesP(cur, rho, theta, lineThreshold, (minLineLength,  maxLineGap))

# check if any lines for detected draw them to image if so
# try:

# except Exception:
#     print("No lines found sadly..")

print("Found this many lines: " + str(len(lines[0])))

for x1, y1, x2, y2 in lines[0]:
    cv2.line(cur, (lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (255,255,0), 5)

# for x1, y1, x2, y2 in lines[0]:
#     cv2.line(cur, (x1, y1), (x2, y2), (0,255,0), 5)

# invert the image
cur =  cv2.bitwise_not(cur)

# Print images to screen
cv2.imshow("houghlines", cur)
plt.show(100)

# # Attempt to sharpen the image after bluring
# sharp = np.zeros((img.shape[0], img.shape[1]), np.int8)
# # cv2.addWeighted(blur, 2.0, img, -0.1, 0, img)
# # cv2.imshow("sharp", sharp)
# # cv2.imshow("original", img)