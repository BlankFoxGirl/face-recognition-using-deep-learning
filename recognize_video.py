#!/usr/local/bin/python
# import libraries
import os, datetime
import cv2, uuid, json
import imutils
import time, sys
import pickle
import numpy as np
import motion_detection as mp
import utilities, video
from imutils.video import FPS
from imutils.video import VideoStream

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
MOTION_TIMEOUT=5
MINIMUM_MOTION_SIZE=20
IDENTITY_FRAME_HITS=3
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

if sys.argv[4] is not None:
    MOTION_THRESHOLD=int(sys.argv[4])
    print("[INFO] Setting threshold to: {}. (Higher values are less sensitive)".format(MOTION_THRESHOLD))

if sys.argv[5] is not None:
    RECORD_VIDEO=sys.argv[5] == "True"
    print("[INFO] Setting RECORD_VIDEO to: {}.".format(RECORD_VIDEO))

# Methods
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
            idFrames = IDENTITY_FRAME_HITS
            name = person
            if (name in peopleConfig["people"]):
                    if "identity-frames" in peopleConfig["people"][name]:
                        idFrames=peopleConfig["people"][name]["identity-frames"]
            if peopleFound[person]["hits"] < idFrames:
                continue
            print("[TRACKING_STOP] [{}] {} left after {} seconds.".format(nowDateTime, person, str(now - datetime.timedelta(seconds=5) - peopleFound[person]["firstSeen"])))

    for person in toDelete:
        del peopleFound[person]

    return peopleFound

def detectedInPersonMotionBoundary(startX, startY, endX, endY):
    global peopleFound
    centrePoint = utilities.getCenterPoint(startX, startY, endX, endY)
    returnPeople = []

    for person in peopleFound:
        idFrames = IDENTITY_FRAME_HITS
        name = peopleFound[person]["name"]
        if (name in peopleConfig["people"]):
                    if "identity-frames" in peopleConfig["people"][name]:
                        idFrames=peopleConfig["people"][name]["identity-frames"]
        if peopleFound[person]["hits"] < idFrames:
            continue
        if utilities.pointInScaledRectangle(centrePoint["x"], centrePoint["y"], peopleFound[person]["moveX"], peopleFound[person]["moveY"], peopleFound[person]["moveEndX"], peopleFound[person]["moveEndY"], 0.6):
            if person == "unknown":
                continue
            peopleFound[person]["lastSeen"] = datetime.datetime.now()
            peopleFound[person]["hits"] += 1
            peopleFound[person]["moveX"] = startX
            peopleFound[person]["moveY"] = startY
            peopleFound[person]["moveEndX"] = endX
            peopleFound[person]["moveEndY"] = endY

            return peopleFound[person]
    return None

def resetBackground():
    global background
    background = None

def getMotion(frame):
    global motionBg

    if motionBg is None:
        motionBg = mp.getGrey(frame)
        return []

    return mp.getMotion(frame, motionBg, MOTION_SENSITIVITY, MOTION_THRESHOLD)

def captureTrainingImage(frame, startX, startY, endX, endY, proba, name):
    if proba < 0.96 and CAPTURE_UNKNOWNS == False:
        return
    extractionForTraining = frame[(startY-EXTRACT_PADDING):(endY+EXTRACT_PADDING), (startX-EXTRACT_PADDING):(endX+EXTRACT_PADDING)]
    if proba < 0.8:
        name = "unknown"

    if extractionForTraining is None:
        extractionForTraining = frame[startY:endY, startX:endX]
    try:
        datetimeStamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        if os.path.exists('captured/Unrecognised/{}/{}/{}'.format(name, MY_ID, datetimeStamp)) == False:
            os.makedirs('captured/Unrecognised/{}/{}/{}'.format(name, MY_ID, datetimeStamp))
        imageName='captured/Unrecognised/{}/{}/{}/UNKN_{}_{}_{}_{:.0f}.jpg'.format(name, MY_ID, datetimeStamp, MY_ID, name, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), proba * 10000)
        cv2.imwrite(imageName, extractionForTraining)
    except Exception as e:
        print("[TRAINING] Failed to write image {} - {}".format(imageName, e))
        pass

def getMotionSignature(x, y, w, h):
    for signature in motionSignatures:
        if signature["x"] == x and signature["y"] == y and signature["w"] == w and signature["h"] == h:
            return signature
    newSig = {
        "uuid": str(uuid.uuid4()),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "hits": 0,
        "lastSeen": datetime.datetime.now(),
        "firstSeen": datetime.datetime.now(),
    }
    motionSignatures.append(newSig)
    return newSig

def handleDetections(frame, detections, sourceFrame=None, offsetX=0, offsetY=0):
    global trainingImagesCaptured, peopleFound
    for idx, detection in enumerate(detections[1]):
        confidence = detection[-1]
        startX = int(detection[0])
        startY = int(detection[1])
        endX = int(detection[0]) + int(detection[2])
        endY = int(detection[1]) + int(detection[3])

        if confidence < 0.6:
            print("Breaking")
            continue

        # extract the face ROI
        face = frame[startY:endY, startX:endX]
        if face is None:
            continue
        (fH, fW) = face.shape[:2]
        (mH, mW) = frame.shape[:2]

        # # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            print("[DETECT] Face too small. {}W {}H".format(fW, fH))
            continue

        identified = False
        # Check if the face is inside a person boundary.
        # person = rectangleInsidePersonBoundary(startX, startY, endX, endY)
        person = detectedInPersonMotionBoundary(offsetX, offsetY, offsetX + mW, offsetY + mH)
        if person is not None:
            # print("Face inside {}'s boundary.".format(person["name"]))
            person["hits"] += 1
            person["startX"] = startX-CONTAINER_PADDING+offsetX
            person["startY"] = startY-CONTAINER_PADDING+offsetY
            person["endX"] = endX+CONTAINER_PADDING+offsetX
            person["endY"] = endY+CONTAINER_PADDING+offsetY
            identified = True

            continue

        # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
        try:
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
        except:
            print("[DETECT] Failed to create faceBlob at {}x{} - {}x{}".format(startX, startY, endX, endY))
            continue

        if TRAINING_MODE==True:
            captureTrainingImage(frame, startX, startY, endX, endY, 0, 'TRAINING')
            trainingImagesCaptured += 1
            print("[TRAINING_MODE] Captured image #{}".format(trainingImagesCaptured))
            continue

        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        if (name == 'not-a-face'):
            continue

        if (name in peopleConfig["people"]):
            if proba < peopleConfig["people"][name]["required-probability"]:
                print("Probability {} less than {} for {}".format(proba, peopleConfig["people"][name]["required-probability"], name))
                if proba < peopleConfig["people"][name]["training.ignore.threashold"]:
                    continue # Ignore.
                name = "unknown" # Override name if probability is too low to identify this person.

        now = datetime.datetime.now()
        nowDateTime = now.strftime("%Y-%m-%d_%H:%M:%S")

        if ((proba < ACCURACY_THREASHOLD or confidence < CONFIDENCE_THREASHOLD) and name != 'unknown'):
            captureTrainingImage(frame if sourceFrame is None else sourceFrame, startX+offsetX, startY+offsetY, endX+offsetX, endY+offsetY, proba, name)
            continue
        elif name == 'unknown':
            captureTrainingImage(frame if sourceFrame is None else sourceFrame, startX+offsetX, startY+offsetY, endX+offsetX, endY+offsetY, proba, name)

        if name in peopleFound: # Update last seen, but do not notify.
            peopleFound[name]["lastSeen"] = datetime.datetime.now()
            peopleFound[name]["hits"] += 1
            peopleFound[name]["proba"] = proba
            idFrames = IDENTITY_FRAME_HITS
            if (name in peopleConfig["people"]):
                    if "identity-frames" in peopleConfig["people"][name]:
                        idFrames=peopleConfig["people"][name]["identity-frames"]
            if peopleFound[name]["hits"] >= idFrames:
                # Update with positions.
                peopleFound[name]["startX"] = startX-CONTAINER_PADDING+offsetX
                peopleFound[name]["startY"] = startY-CONTAINER_PADDING+offsetY
                peopleFound[name]["endX"] = endX+CONTAINER_PADDING+offsetX
                peopleFound[name]["endY"] = endY+CONTAINER_PADDING+offsetY
                peopleFound[name]["moveX"] = offsetX
                peopleFound[name]["moveY"] = offsetY
                peopleFound[name]["moveEndX"] = offsetX + mW
                peopleFound[name]["moveEndY"] = offsetY + mH

            if peopleFound[name]["hits"] == idFrames:
                if (name in peopleConfig["people"]):
                    if peopleConfig["people"][name]["skip-snapshot"] == True:
                        continue
                frame = video.drawInFrame(frame if sourceFrame is None else sourceFrame, peopleFound, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, peopleConfig)
                filename='captured/Snapshots/{}_{}_{}_{:.0f}.jpg'.format(MY_ID, name, now.strftime("%Y%m%d%H%M%S"), proba * 10000)
                print("[TRACKING_START] [{}] ({:.2f}%) {}'s Face detected at ({}x,{}y) - ({}x,{}y), confidence: {} (File: {})".format(nowDateTime, proba * 100, name, startX, startY, endX, endY, confidence, filename))
                cv2.imwrite(filename,frame)

            continue

        peopleFound[name] = {
            "name": name,
            "lastSeen": datetime.datetime.now(),
            "firstSeen": datetime.datetime.now(),
            "hits": 1,
            "proba": proba,
            "moveX": offsetX,
            "moveY": offsetY,
            "endMoveX": offsetX + mW,
            "endMoveY": offsetY + mH
        }

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
# vs = VideoStream(src=0).start()
# vs = cv2.VideoCapture("rtsp://Xf6N0dIENhKR:Ha62gamgMX2f@10.2.1.202/live0", cv2.CAP_FFMPEG) # Eufy

# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
vs = cv2.VideoCapture(INPUT_FILE, cv2.CAP_FFMPEG) # Kasa.
# vs = cv2.VideoCapture("dianne.mp4")
vs.set(cv2.CAP_PROP_POS_MSEC, 1)

time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
print("[READY] Video Stream Started.")
out = None
# loop over frames from the video file stream
peopleConfig = json.load(open("dataset/config.json"))
peopleFound = {}
motionSignatures = []

background = None
timeoutMotion = 0
trainingImagesCaptured=0
frameCounter=0
motionBg=None

# Limit FPS to 60FPS.
while vs.isOpened():
    # time.sleep(1/FPS_LIMIT)
    # grab the frame from the threaded video stream
    ret, frame = vs.read()

    peopleFound = cleanUpPeople(peopleFound)

    if ret == False:
        break
    if frame is None:
        continue

    frameCounter += 1
    motion = getMotion(frame)

    if frameCounter % (5 * 60 * 30) == 1: # Every 5 minutes.
        if len(peopleFound) == 0:
            mp.resetBackground()
            frameCounter = 0

    if mp.isMotionDetectedIn(motion) == False:
        if time.time() > timeoutMotion:
            if len(peopleFound) == 0:
                out = video.stopRecording(out, RECORD_VIDEO)
            continue
    else:
        out = video.startRecording("{}-Activity-{}".format(MY_ID, datetime.datetime.now().strftime("%Y%m%d%H%M%S")), out, INPUT_WIDTH, INPUT_HEIGHT, RECORD_VIDEO)
        timeoutMotion = time.time() + MOTION_TIMEOUT

    (h, w) = frame.shape[:2]

    if mp.isMotionDetectedIn(motion) == True:
    # Extract movement from frame.
        for idx, m in enumerate(motion):
            # print("[MOTION] Motion detected at {}x,{}y - {}x,{}y".format(m["startX"], m["startY"], m["endX"], m["endY"]))
            movementFrame = frame.copy()
            movementFrame = movementFrame[motion[idx]["startY"]:motion[idx]["endY"], motion[idx]["startX"]:motion[idx]["endX"]]

            (mH, mW) = movementFrame.shape[:2]

            if mW < MINIMUM_MOTION_SIZE or mH < MINIMUM_MOTION_SIZE:
                continue

            # apply OpenCV's deep learning-based face detector to localize faces in the input image
            detector.setInputSize((mW, mH))
            detections = detector.detect(movementFrame)
            if detections[1] is not None:
                handleDetections(movementFrame, detections, sourceFrame=frame, offsetX=m["startX"], offsetY=m["startY"])
            else:
                # print("[MOTION] No faces detected in movement frame.")
                person=detectedInPersonMotionBoundary(m["startX"], m["startY"], m["endX"], m["endY"])
                if person is not None:
                    print("[MOTION] Inferring person {} at location (({},{}),({},{})).".format(person["name"], m["startX"], m["startY"], m["endX"], m["endY"]))
    else:
        detector.setInputSize((INPUT_WIDTH, INPUT_HEIGHT))
        detections = detector.detect(frame)

        if detections[1] is not None:
            # print("[DETECT] {} faces detected.".format(len(detections[1])))
            handleDetections(frame, detections)

    # update the FPS counter
    fps.update()
    out = video.writeFrame(frame, motion, out, peopleFound, RECORD_VIDEO, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, peopleConfig)
    # show the output frame
    # cv2.imshow("Frame", frame)
    # out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[END] Elasped time: {:.2f}".format(fps.elapsed()))
print("[END] Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
# vs.stop()