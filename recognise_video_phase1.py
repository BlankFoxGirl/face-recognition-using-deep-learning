#!/usr/local/bin/python
'''
    Phase 1 Features;
    - Motion detection.
    - Face detection.
    - Face recognition.
    - Local Storage of images, metadata, and video.
'''
# import libraries
import os, datetime, cv2, json, imutils, time, pickle, utilities, video, argparse, log
import numpy as np
import motion_detection as mp
import face_recogniser as fr
from imutils.video import FPS

ACCURACY_THREASHOLD=0.98
COMPARE_SIZE=100
CONFIDENCE_THREASHOLD=0.85
DRAW_FACE_BOXES=False
FPS_LIMIT=60
EXTRACT_PADDING=60
CONTAINER_PADDING=60
TRAINING_MODE=False
CAPTURE_UNKNOWNS=True
RECORD_VIDEO=True
MY_ID="Test"
MOTION_SENSITIVITY=10000
MOTION_THRESHOLD=60
MOTION_TIMEOUT=15
MINIMUM_MOTION_SIZE=20
IDENTITY_FRAME_HITS=3
INPUT_HEIGHT=1080
INPUT_WIDTH=1920

INPUT_FILE="rtmp://10.2.1.143:1935/live/app"

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help="Input, must be a URL or File.", type=str, default=INPUT_FILE)
parser.add_argument('--name', '-n', help="Name of the listener identity", type=str, default="Test")
parser.add_argument('--motion-sensitivity', '-s', help="Motion Sensitivity", type=int, default=10000)
parser.add_argument('--motion-threshold', '-t', help="Motion Threshold", type=int, default=60)
parser.add_argument('--record-video', '-r', help="Record Video", type=bool, default=True)
parser.add_argument('--draw-face-boxes', '-d', help="Draw Face Boxes", type=bool, default=False)
parser.add_argument('--training-mode', '-l', help="Training Mode aka Learning Mode", type=bool, default=False)
ARGS=parser.parse_args()

INPUT_FILE = ARGS.input
MY_ID = ARGS.name
MOTION_SENSITIVITY = ARGS.motion_sensitivity
MOTION_THRESHOLD = ARGS.motion_threshold
RECORD_VIDEO = ARGS.record_video
DRAW_FACE_BOXES = ARGS.draw_face_boxes
TRAINING_MODE = ARGS.training_mode

# Methods
def captureSnapshot(frame, person, proba, name, nowDateTime, PATH):
    frame = video.drawInFrame(frame, person, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, peopleConfig)
    filename='{}/{}_{}_{}_{:.0f}.jpg'.format(PATH, MY_ID, name, nowDateTime, proba * 10000)
    cv2.imwrite(filename,frame)

def resetBackground():
    global background
    background = None

def getMotion(frame):
    global motionBg, CONTAINER_PADDING

    if motionBg is None:
        motionBg = mp.getGrey(frame)
        return []

    return mp.getMotion(frame, motionBg, MOTION_SENSITIVITY, MOTION_THRESHOLD, CONTAINER_PADDING)

def captureTrainingImage(frame, startX, startY, endX, endY, proba, name, PATH=None):
    if proba < 0.96 and CAPTURE_UNKNOWNS == False:
        return
    extractionForTraining = frame[(startY-EXTRACT_PADDING):(endY+EXTRACT_PADDING), (startX-EXTRACT_PADDING):(endX+EXTRACT_PADDING)]
    if proba < 0.8:
        name = "unknown"

    if extractionForTraining is None:
        extractionForTraining = frame[startY:endY, startX:endX]
    try:
        datetimeStamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        if PATH is None:
            PATH = 'captured/Unrecognised/{}/{}/{}'.format(name, MY_ID, datetimeStamp)
        if os.path.exists(PATH) == False:
            os.makedirs(PATH)
        imageName='{}/UNKN_{}_{}_{}_{}.jpg'.format(PATH, name, MY_ID, datetimeStamp, MY_ID, name, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), proba * 10000)
        cv2.imwrite(imageName, extractionForTraining)
    except Exception as e:
        log.exception("[TRAINING] Failed to write image {} - {}".format(imageName, e))
        log.error("({}x,{}y) - ({}x,{}y) {} {}".format(startX, startY, endX, endY, proba, name))

def handleDetections(frame, detections, embedder, recognizer, le, sourceFrame=None, offsetX=0, offsetY=0):
    global peopleConfig
    faces = fr.getRecognisedFaces(frame, detections, embedder, recognizer, le, peopleConfig, offsetX, offsetY)
    now = datetime.datetime.now()
    nowDateTime = now.strftime("%Y-%m-%d_%H-%M")
    p = utilities.createRecursivePath("captured/{}/{}/{}/{}/{}".format(now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H"), now.strftime("%M")))
    # Loop through faces, if the face is below probability then capture training image otherwise write a meta file to the captured directory.
    for face in faces:
        fileName = '{}/{}_{}_{}.json'.format(p, MY_ID, face["name"], nowDateTime)
        if os.path.exists(fileName) == True:
            continue
        if face["proba"] < ACCURACY_THREASHOLD:
            captureTrainingImage(frame if sourceFrame is None else sourceFrame, face["face"]["x"], face["face"]["y"], face["face"]["x"] + face["face"]["w"], face["face"]["h"]+face["face"]["y"], face["proba"], face["name"])
            continue
        else:
            captureSnapshot(frame if sourceFrame is None else sourceFrame, face, face["proba"], face["name"], nowDateTime, p)
        with open(fileName, 'w') as outfile:
            json.dump(face, outfile)

# load serialized face detector

log.info("[WAIT] Loading Face Detector...")
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
log.info("[WAIT] Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
log.info("[WAIT] Starting Video Stream...")

vs = cv2.VideoCapture(INPUT_FILE) # Kasa.
vs.set(cv2.CAP_PROP_POS_MSEC, 1)

time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
log.info("[READY] Video Stream Started.")
out = None
# loop over frames from the video file stream
peopleConfig = json.load(open("dataset/config.json"))
peopleFound = {}
motionSignatures = {}

background = None
timeoutMotion = 0
trainingImagesCaptured=0
frameCounter=0
motionBg=None

while vs.isOpened():
    # grab the frame from the threaded video stream
    ret, frame = vs.read()

    if ret == False:
        break
    if frame is None:
        continue

    frameCounter += 1
    motion = getMotion(frame)

    if frameCounter % (5 * 60 * 30) == 1: # Every 5 minutes.
        mp.resetBackground()
        frameCounter = 0

    if mp.isMotionDetectedIn(motion) == False:
        if time.time() > timeoutMotion:
            out = video.stopRecording(out, RECORD_VIDEO)
    else:
        out = video.startRecording("{}-Activity-{}".format(MY_ID, datetime.datetime.now().strftime("%Y%m%d%H%M%S")), out, INPUT_WIDTH, INPUT_HEIGHT, RECORD_VIDEO)
        timeoutMotion = time.time() + MOTION_TIMEOUT

    (h, w) = frame.shape[:2]

    if mp.isMotionDetectedIn(motion) == True:
    # Extract movement from frame.
        log.debug("[MOTION] Motion detected.")
        for idx, m in enumerate(motion):
            movementFrame = frame.copy()
            movementFrame = movementFrame[((motion[idx]["startY"] + CONTAINER_PADDING) * 4):((motion[idx]["endY"] - CONTAINER_PADDING)*4), ((motion[idx]["startX"] + CONTAINER_PADDING) * 4):((motion[idx]["endX"] - CONTAINER_PADDING) * 4)]

            (mH, mW) = movementFrame.shape[:2]

            if mW < MINIMUM_MOTION_SIZE or mH < MINIMUM_MOTION_SIZE:
                continue

            # apply OpenCV's deep learning-based face detector to localize faces in the input image
            detections = fr.getDetectedFaces(detector, frame)
            if detections[1] is not None:
                handleDetections(frame, detections, embedder, recognizer, le)

    # update the FPS counter
    fps.update()
    out = video.writeFrame(frame, motion, out, motionSignatures, RECORD_VIDEO, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, peopleConfig)
    # show the output frame
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
log.info("[END] Elasped time: {:.2f}".format(fps.elapsed()))
log.info("[END] Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()