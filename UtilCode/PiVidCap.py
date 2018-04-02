# Imports
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2


frameSize = (640, 480) # 0.5 the size of webcam raw video(same aspect ratio)

# Initialise Camera objects
camera = PiCamera()
camera.resolution = frameSize
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = frameSize)

# Define saving codec and create writing object
# fourCC = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("piOutput.avi", fourCC, 20.0, frameSize)

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Resize frame to output size
    image = cv2.resize(image, frameSize, interpolation = cv2.INTER_AREA)
    # Convert frame to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Add frame to writing object
    # out.write(image)

    cv2.imshow("img", image)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("q"):
        break

# out.release()
cv2.destroyAllWindows()