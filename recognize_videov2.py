#!/usr/local/bin/python
# import libraries
import os, datetime
import cv2, uuid, json
import imutils
import time, sys
import pickle, face_recognition
import numpy as np
import motion_detection as mp
import utilities, video, quadrants
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

def cleanupStaleSignatures():
    signaturesToClear = []
    for signature in motionSignatures:
        if motionSignatures[signature]["lastSeen"] < datetime.datetime.now() - datetime.timedelta(seconds=5):
            signaturesToClear.append(signature)
            continue

    for signature in signaturesToClear:
        del motionSignatures[signature]

def getSignaturesFromMotion(motion):
    global motionSignatures
    signatures = []
    for m in motionSignatures:
        identifiedMotion = None
        if "quadrants" in motionSignatures[m]:
            for mot in motion:
                # do.
                cq = quadrants.centerQuadrant(motionSignatures[m]["quadrants"])
                if (quadrants.rectangleOccupiesQuadrant(cq, mot["startX"], mot["startY"], mot["endX"] - mot["startX"], mot["endY"] - mot["startY"]) == True):
                    identifiedMotion = mot
                    break

        if identifiedMotion is not None:
            signatures.append({
                "signatureId": m,
                "motion": identifiedMotion
            })
    return signatures

def peopleInSignatures(signatures):
    people = []
    for sig in signatures:
        if "people" not in motionSignatures[sig["signatureId"]]:
            continue
        for person in motionSignatures[sig["signatureId"]]["people"]:
            if person not in people:
                people.append(person)
    return people

def getMotionSignature(x, y, w, h):
    global motionSignatures
    triggeredQuadrants = quadrants.quadrantsInRectangle(x, y, w, h)
    for signature in motionSignatures:
        centerQuadrant = quadrants.centerQuadrant(triggeredQuadrants)
        if centerQuadrant in motionSignatures[signature]["quadrants"]:
            motionSignatures[signature]["people"] = cleanUpPeople(motionSignatures[signature]["people"])
            motionSignatures[signature]["quadrants"] = triggeredQuadrants
            motionSignatures[signature]["x"] = x
            motionSignatures[signature]["y"] = y
            motionSignatures[signature]["w"] = w
            motionSignatures[signature]["h"] = h
            motionSignatures[signature]["rect"] = quadrants.rectFromQuadrants(motionSignatures[signature]["quadrants"])
            updateSignature(motionSignatures[signature])

            if "rect" not in motionSignatures[signature] or motionSignatures[signature]["rect"]["viewportPercent"] >= 30: # Filter out unrealistic.
                continue

            print(motionSignatures[signature])
            return motionSignatures[signature]

    newSig = {
        "uuid": str(uuid.uuid4()),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "hits": 0,
        "lastSeen": datetime.datetime.now(),
        "firstSeen": datetime.datetime.now(),
        "quadrants": triggeredQuadrants,
        "rect": quadrants.rectFromQuadrants(triggeredQuadrants),
        "people": {}
    }

    if newSig["rect"]["viewportPercent"] > 30: # Filter out unrealistic movement.
        return None

    updateSignature(newSig)
    print("[MOTION] Signature {} created".format(newSig["uuid"]))
    return newSig

def updateSignature(signature):
    global motionSignatures

    if len(signature["people"]) > 0:
        signature["hits"] += 1
        signature["lastSeen"] = datetime.datetime.now()

    if signature["uuid"] not in motionSignatures:
        motionSignatures[signature["uuid"]] = signature
        return

    motionSignatures[signature["uuid"]] = signature

def handleDetections(frame, detections, signature, sourceFrame=None, offsetX=0, offsetY=0, motion={}):
    global trainingImagesCaptured, peopleFound, knownData
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
 
        if TRAINING_MODE==True:
            captureTrainingImage(frame, startX, startY, endX, endY, 0, 'TRAINING')
            trainingImagesCaptured += 1
            print("[TRAINING_MODE] Captured image #{}".format(trainingImagesCaptured))
            continue

        small_frame = cv2.resize(frame if sourceFrame is None else sourceFrame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if len(face_locations) == 0:
            continue

        if rgb_small_frame is None:
            continue

        encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="large")
        name = "unknown"
        proba = 0.7

        for faceEnc in encodings:
            matches = face_recognition.compare_faces(knownData["embeddings"], faceEnc, tolerance=0.6)
            face_distances = face_recognition.face_distance(knownData["embeddings"], faceEnc)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = knownData["names"][best_match_index]
                proba = 1.0
                break

        sigs = getSignaturesFromMotion(motion)
        people = peopleInSignatures(sigs)
        if (name != "unknown" and (name == 'not-a-face' or name in people)):
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

        if sourceFrame is None:
            signature = getMotionSignature(startX, startY, endX, endY)

        if signature == None:
            continue

        peopleFound = signature["people"]
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

            updateSignature(signature)
            if peopleFound[name]["hits"] == idFrames:
                if (name in peopleConfig["people"]):
                    if peopleConfig["people"][name]["skip-snapshot"] == True:
                        continue
                frame = video.drawInFrame(frame if sourceFrame is None else sourceFrame, [], peopleFound, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, peopleConfig)
                filename='captured/Snapshots/{}_{}_{}_{:.0f}.jpg'.format(MY_ID, name, now.strftime("%Y%m%d%H%M%S"), proba * 10000)
                print("[TRACKING_START] [{}] ({:.2f}%) {}'s Face detected at ({}x,{}y) - ({}x,{}y), confidence: {} (File: {})".format(nowDateTime, proba * 100, name, startX, startY, endX, endY, confidence, filename))
                cv2.imwrite(filename,frame)

            continue

        signature["people"][name] = {
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
        updateSignature(signature)

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
# embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# # load the actual face recognition model along with the label encoder
# recognizer = pickle.loads(open("output/recognizerv2.pickle", "rb").read())
# le = pickle.loads(open("output/lev2.pickle", "rb").read())

knownData = pickle.loads(open("output/embeddingsv2.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[WAIT] Starting Video Stream...")
# vs = VideoStream(src=0).start()
# vs = cv2.VideoCapture("rtsp://Xf6N0dIENhKR:Ha62gamgMX2f@10.2.1.202/live0", cv2.CAP_FFMPEG) # Eufy

# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
vs = cv2.VideoCapture(INPUT_FILE) # Kasa.
# vs = cv2.VideoCapture("dianne.mp4")
vs.set(cv2.CAP_PROP_POS_MSEC, 1)

time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
print("[READY] Video Stream Started.")
out = None
out2 = None
# loop over frames from the video file stream
peopleConfig = json.load(open("dataset/config.json"))
peopleFound = {}
motionSignatures = {}

background = None
timeoutMotion = 0
trainingImagesCaptured=0
frameCounter=0
motionBg=None
firstFrame=None

# Limit FPS to 60FPS.
while vs.isOpened():
    if out is None:
        out = video.startRecording("tmp-{}".format(time.time()), out, 1920, 1080, True)
    if out2 is None:
        out2 = video.startRecording("tmp-actual-{}".format(time.time()), out2, 1920, 1080, True)
    # time.sleep(1/FPS_LIMIT)
    # grab the frame from the threaded video stream
    ret, frame = vs.read()

    # peopleFound = cleanUpPeople(peopleFound)
    cleanupStaleSignatures()

    if ret == False:
        break
    if frame is None:
        continue

    frameCounter += 1
    if firstFrame is None:
        firstFrame = frame
        continue

    sigFrame = mp.getSignatureFrame(frame, firstFrame)
    out.write(sigFrame)
    out2.write(frame)
    # out = video.writeFrame(sigFrame, [], out, {}, True, False, 900, {})
    if frameCounter % 30 == 1:
        print("Frame write")
    continue
    motion = getMotion(frame)

    if frameCounter % 30 == 1:
        print("Frame write")

    if frameCounter % (5 * 60 * 30) == 1: # Every 5 minutes.
        # if len(peopleFound) == 0:
        mp.resetBackground()
        frameCounter = 0

    if mp.isMotionDetectedIn(motion) == False:
        if time.time() > timeoutMotion:
            if len(motionSignatures) == 0:
                out = video.stopRecording(out, RECORD_VIDEO)
            continue
    else:
        out = video.startRecording("{}-Activity-{}".format(MY_ID, datetime.datetime.now().strftime("%Y%m%d%H%M%S")), out, INPUT_WIDTH, INPUT_HEIGHT, RECORD_VIDEO)
        timeoutMotion = time.time() + MOTION_TIMEOUT

    (h, w) = frame.shape[:2]

    # if mp.isMotionDetectedIn(motion) == True:
        
    # # Extract movement from frame.
    #     for idx, m in enumerate(motion):
    #         # print("[MOTION] Motion detected at {}x,{}y - {}x,{}y".format(m["startX"], m["startY"], m["endX"], m["endY"]))
    #         movementFrame = frame.copy()
    #         movementFrame = movementFrame[(m["startY"] * 4):(m["endY"]*4), (m["startX"] * 4):(m["endX"] * 4)]

    #         (mH, mW) = movementFrame.shape[:2]

    #         if mW < MINIMUM_MOTION_SIZE or mH < MINIMUM_MOTION_SIZE:
    #             continue

    #         signature = getMotionSignature((m["startX"] * 4), (m["startY"] * 4), (m["endX"] * 4), (m["endY"] * 4))
    #         if signature is None:
    #             continue

    #         if len(signature["people"]) > 0:
    #             continue

    #         # apply OpenCV's deep learning-based face detector to localize faces in the input image
    #         detector.setInputSize((mW, mH))
    #         detections = detector.detect(movementFrame)
    #         if detections[1] is not None:
    #             handleDetections(movementFrame, detections, signature, sourceFrame=frame, offsetX=m["startX"]*4, offsetY=m["startY"]*4)
    #         # else:
    #         #     # print("[MOTION] No faces detected in movement frame.")
    #         #     person=detectedInPersonMotionBoundary(m["startX"], m["startY"], m["endX"], m["endY"])
    #         #     if person is not None:
    #         #         print("[MOTION] Inferring person {} at location (({},{}),({},{})).".format(person["name"], m["startX"], m["startY"], m["endX"], m["endY"]))
    # else:
    # if frameCounter % 5 == 1:
        # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInputSize((INPUT_WIDTH, INPUT_HEIGHT))
    detections = detector.detect(frame)

    if detections[1] is not None:
        print("[DETECT] {} faces detected.".format(len(detections[1])))
        handleDetections(frame, detections, {}, motion=motion)

    # update the FPS counter
    fps.update()
    # out = video.writeFrame(frame, motion, out, motionSignatures, RECORD_VIDEO, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, peopleConfig)
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