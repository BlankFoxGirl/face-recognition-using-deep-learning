#!/usr/local/bin/python
# import libraries
import os, datetime, base64, socket, json
import cv2
import imutils
import time, sys
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

# This file is used to listen to a video stream and send the frame into a socket for processing.

ACCURACY_THREASHOLD=0.98
COMPARE_SIZE=100
CONFIDENCE_THREASHOLD=0.85
FPS_LIMIT=60
EXTRACT_PADDING=60
CONTAINER_PADDING=60
TRAINING_MODE=False
MY_ID="Test"
MOTION_SENSITIVITY=10000
INPUT_HEIGHT=1080
INPUT_WIDTH=1920

INPUT_FILE="rtmp://10.2.1.143:1935/live/app"
if sys.argv[1] is not None:
    INPUT_FILE=sys.argv[1]
    print("[INFO] Using input file: {}".format(INPUT_FILE))

if sys.argv[2] is not None:
    MY_ID=sys.argv[2]
    print("[INFO] Setting listener to identifier: {}".format(MY_ID))

if sys.argv[3] is not None:
    MOTION_SENSITIVITY=int(sys.argv[3])
    print("[INFO] Setting sensitivity to: {}. (Higher values are less sensitive)".format(MOTION_SENSITIVITY))

# Methods
def slugToName(slug):
    return slug.replace("-", " ").title()

def cleanUpPeople(peopleFound):
    toDelete = []
    now = datetime.datetime.now()
    nowDateTime = now.strftime("%Y-%m-%d_%H:%M:%S")
    for person in peopleFound:
        if "lastSeen" not in peopleFound[person]:
            toDelete.append(person)
            continue

        if peopleFound[person]["lastSeen"] < now - datetime.timedelta(seconds=5):
            toDelete.append(person)
            if peopleFound[person]["hits"] < 5:
                continue
            print("[TRACKING_STOP] [{}] {} left after {} seconds.".format(nowDateTime, person, str(now - datetime.timedelta(seconds=5) - peopleFound[person]["firstSeen"])))

    for person in toDelete:
        del peopleFound[person]

    return peopleFound

def rectangleInsidePersonBoundary(startX, startY, endX, endY):
    global peopleFound
    for person in peopleFound:
        if peopleFound[person]["hits"] < 5:
            continue
        if startX > peopleFound[person]["startX"] and startY > peopleFound[person]["startY"] and endX < peopleFound[person]["endX"] and endY < peopleFound[person]["endY"]:
            return peopleFound[person]
    return None

def sendToSocket(frame):
    data = {
        'camera': '',
        'frame': base64.b64encode(frame).decode('utf-8'),
    }
    json_data = json.dumps(data)

    socketPath = '/app/p.sock'
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socketPath)
        sock.sendall(json_data.encode('utf-8'))

    except OSError as e:
        print(f'Error connecting to socket: {e}')

    finally:
        sock.close()
        print("Closed socket.")

# load serialized face detector
print("[WAIT] Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
detector = cv2.FaceDetectorYN.create(
    "face_detection_model/face_detection_yunet_2023mar.onnx",
    "",
    (COMPARE_SIZE, COMPARE_SIZE),
    0.9,
    0.3,
    5000
)

# load serialized face embedding model
print("[WAIT] Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[WAIT] Starting Video Stream...")
vs = cv2.VideoCapture(INPUT_FILE) # Kasa.

time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
print("[READY] Video Stream Started.")
out = None
# loop over frames from the video file stream
peopleFound = {}
background = None
timeoutMotion = 0
trainingImagesCaptured=0
frameCounter=0

while vs.isOpened():
    # grab the frame from the threaded video stream
    ret, frame = vs.read()

    if ret == False:
        break
    if frame is None:
        continue

    sendToSocket(frame)

    frameCounter += 1
    if frameCounter % (5 * 60 * 30) == 1: # Every aggregate 5 minutes of motion.
        # Reset background.
        frameCounter = 0

    (h, w) = frame.shape[:2]

# stop the timer and display FPS information
fps.stop()
print("[END] Elasped time: {:.2f}".format(fps.elapsed()))
print("[END] Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()