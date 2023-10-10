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
import concurrent.futures
import paho.mqtt.client as paho
import events as ev

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
MIN_MOTION_AREA=10000
MAX_MOTION_AREA=50000
MINIMUM_MOTION_SIZE=20
IDENTITY_FRAME_HITS=3
INPUT_HEIGHT=1080
INPUT_WIDTH=1920

INPUT_FILE="rtmp://10.2.1.143:1935/live/app"

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help="Input, must be a URL or File.", type=str, default=INPUT_FILE)
parser.add_argument('--name', '-n', help="Name of the listener identity", type=str, default="Test")
parser.add_argument('--motion-threshold', '-t', help="Motion Threshold", type=int, default=60)
parser.add_argument('--motion-min-area', help="Min area of motion box.", type=int, default=10000)
parser.add_argument('--motion-max-area', help="Max area of motion box.", type=int, default=50000)
parser.add_argument('--record-video', '-r', help="Record Video", type=int, default=1)
parser.add_argument('--draw-face-boxes', '-d', help="Draw Face Boxes", type=bool, default=False)
parser.add_argument('--training-mode', '-l', help="Training Mode aka Learning Mode", type=bool, default=False)
ARGS=parser.parse_args()

INPUT_FILE = ARGS.input
MY_ID = ARGS.name
MOTION_THRESHOLD = ARGS.motion_threshold
RECORD_VIDEO = True if ARGS.record_video == 1 else False
DRAW_FACE_BOXES = ARGS.draw_face_boxes
TRAINING_MODE = ARGS.training_mode
MIN_MOTION_AREA = ARGS.motion_min_area
MAX_MOTION_AREA = ARGS.motion_max_area

log.setId(MY_ID)

broker="mqtt"
port=1883

def on_publish(client, userdata, result):
    log.debug("Data published.", DUMP={result,userdata}, APPEND=["MQTT"])

client= paho.Client(MY_ID)
client.on_publish = on_publish
log.info("Connecting to broker {}:{}".format(broker,port), ["START"])
client.connect(broker, port)
log.info("Broker connection successful", ["START"])

log.info("Running with the following settings:", ["START"])
log.info("Input: {}".format(INPUT_FILE), ["START"])
log.info("Name: {}".format(MY_ID), ["START"])
log.info("Motion Threshold: {}".format(MOTION_THRESHOLD), ["START"])
log.info("Record Video: {}".format(RECORD_VIDEO), ["START"])
log.info("Draw Face Boxes: {}".format(DRAW_FACE_BOXES), ["START"])
log.info("Training Mode: {}".format(TRAINING_MODE), ["START"])
log.info("Motion Min Area: {}".format(MIN_MOTION_AREA), ["START"])
log.info("Motion Max Area: {}".format(MAX_MOTION_AREA), ["START"])

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

    return mp.getMotion(frame, MIN_MOTION_AREA=MIN_MOTION_AREA, MAX_MOTION_AREA=MAX_MOTION_AREA, MOTION_THRESHOLD=MOTION_THRESHOLD, CONTAINER_PADDING=CONTAINER_PADDING)

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
        log.exception("Failed to write image {} - {}".format(imageName, e), ["TRAINING"])
        log.error("({}x,{}y) - ({}x,{}y) {} {}".format(startX, startY, endX, endY, proba, name), ["TRAINING"])

def sendPing():
    message = {
        "id": MY_ID,
        "event": ev.PING,
    }
    message = utilities.encodeMessage(message)
    ret = client.publish("/data", message)

def sendInit():
    message = {
        "id": MY_ID,
        "event": ev.CAMERA_INITIALISED,
    }
    message = utilities.encodeMessage(message)
    ret = client.publish("/data", message)

def sendMotionDetected(motion):
    message = {
        "id": MY_ID,
        "event": ev.MOTION,
        "motion": motion,
    }
    message = utilities.encodeMessage(message)
    ret = client.publish("/data", message)

def sendCaptureStart(file):
    message = {
        "id": MY_ID,
        "event": ev.RECORDING_STARTED,
        "filename": file,
    }
    message = utilities.encodeMessage(message)
    ret = client.publish("/data", message)

def sendCaptureStop(out, file):
    if out is None:
        return
    message = {
        "id": MY_ID,
        "event": ev.RECORDING_STOPPED,
        "filename": file,
    }
    message = utilities.encodeMessage(message)
    ret = client.publish("/data", message)

def sendDetectionToMQTT(frame, detections, filename, offsetX=0, offsetY=0):
    global client
    message = {
        "id": MY_ID,
        "event": ev.DETECTION,
        "detections": detections[1].tolist(),
        "frame": utilities.encodeImage(frame),
        "offsetX": offsetX,
        "offsetY": offsetY,
        "filename": filename,
    }

    message = utilities.encodeMessage(message)
    ret = client.publish("/data", message)

# This will be put into it's own worker which reads from a queue.
def handleDetections(frame, detections, embedder, recognizer, le, sourceFrame=None, offsetX=0, offsetY=0):
    global peopleConfig
    faces = fr.getRecognisedFaces(frame, detections, embedder, recognizer, le, peopleConfig, offsetX, offsetY, UP_SAMPLE_MULTIPLIER=4)
    now = datetime.datetime.now()
    nowDateTime = now.strftime("%Y-%m-%d_%H-%M")
    p = utilities.createRecursivePath("captured/{}/{}/{}/{}/{}/{}".format(MY_ID, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H"), now.strftime("%M")))
    # Loop through faces, if the face is below probability then capture training image otherwise write a meta file to the captured directory.
    log.debug("Found {} faces".format(len(faces)), ["FACE_DETECT"])
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

log.info("Loading Face Detector...", ["WAIT"])

detector = cv2.FaceDetectorYN.create(
    "face_detection_model/face_detection_yunet_2023mar.onnx",
    "",
    (COMPARE_SIZE, COMPARE_SIZE),
    0.9,
    0.3,
    5000
)

sourceFPS=0

# initialize the video stream, then allow the camera sensor to warm up
log.info("Starting Video Stream...", ["WAIT"])
try:
    vs = cv2.VideoCapture(INPUT_FILE, cv2.CAP_FFMPEG) # Kasa.
    vs.set(cv2.CAP_PROP_POS_MSEC, 1)
    time.sleep(2.0)
    sourceFPS = vs.get(cv2.CAP_PROP_FPS)
    log.info("Video Stream FPS {}.".format(sourceFPS), ["WAIT"])
    ret, frame = vs.read()
    if ret == False:
        log.error("Failed to read frame from video stream.")
        exit(1)
except Exception as e:
    log.exception("Failed to open video stream {}".format(e))
    exit(1)

# start the FPS throughput estimator
fps = FPS().start()
log.success("Video Stream Started.", ["READY"])
out = None
currentVideoFilename = ""

peopleConfig = json.load(open("dataset/config.json"))
peopleFound = {}
motionSignatures = {}

background = None
timeoutMotion = 0
trainingImagesCaptured=0
frameCounter=0
motionBg=None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
sendInit()
lastPing = time.time()

# loop over frames from the video file stream
while vs.isOpened():
    # grab the frame from the threaded video stream
    ret, frame = vs.read()

    if ret == False:
        break
    if frame is None:
        continue

    frameCounter += 1
    motion = getMotion(frame)

    if lastPing < (time.time() - 30): # Every 30 seconds.
        sendPing()
        lastPing = time.time()

    if frameCounter % (5 * 60 * 30) == 1: # Every 5 minutes.
        mp.resetBackground()
        frameCounter = 0

    if mp.isMotionDetectedIn(motion) == False:
        if time.time() > timeoutMotion:
            sendCaptureStop(out, currentVideoFilename)
            out = video.stopRecording(out, RECORD_VIDEO)
    else:
        (out, fn) = video.startRecording("Activity-{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")), out, INPUT_WIDTH, INPUT_HEIGHT, RECORD_VIDEO, MY_ID=MY_ID)
        if fn != "":
            currentVideoFilename = fn

        sendCaptureStart(currentVideoFilename)
        timeoutMotion = time.time() + MOTION_TIMEOUT

    (h, w) = frame.shape[:2]
    quaterWidth = int(w / 4)
    quaterHeight = int(h / 4)

    if mp.isMotionDetectedIn(motion) == True:
    # Extract movement from frame.
        log.debug("Motion detected.", ["MOTION"])
        sendMotionDetected(motion)
        validMotion = 0
        for idx, m in enumerate(motion):
            movementFrame = frame.copy()
            movementFrame = movementFrame[((motion[idx]["startY"] + CONTAINER_PADDING) * 4):((motion[idx]["endY"] - CONTAINER_PADDING)*4), ((motion[idx]["startX"] + CONTAINER_PADDING) * 4):((motion[idx]["endX"] - CONTAINER_PADDING) * 4)]

            (mH, mW) = movementFrame.shape[:2]

            if mW < MINIMUM_MOTION_SIZE or mH < MINIMUM_MOTION_SIZE:
                continue
            validMotion += 1

        if validMotion > 0:
            # apply OpenCV's deep learning-based face detector to localize faces in the input image

            detectFrame = imutils.resize(frame, width=quaterWidth, height=quaterHeight)
            detections = fr.getDetectedFaces(detector, detectFrame)
            if detections[1] is not None:
                log.debug("Detected {} faces with motion.".format(len(detections[1])), ["DETECT"])
                sendDetectionToMQTT(frame, detections, currentVideoFilename)

    elif (out is not None): # If we are recording, but haven't detected active motion in this frame, still run face detection.
        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detectFrame = imutils.resize(frame, width=quaterWidth, height=quaterHeight)
        detections = fr.getDetectedFaces(detector, detectFrame)
        if detections[1] is not None:
            log.debug("Detected {} faces without motion.".format(len(detections[1])), ["DETECT"])
            sendDetectionToMQTT(frame, detections, currentVideoFilename)

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
log.info("Elasped time: {:.2f}".format(fps.elapsed()), ["END"])
log.info("Approx. FPS: {:.2f}".format(fps.fps()), ["END"])

# cleanup
cv2.destroyAllWindows()