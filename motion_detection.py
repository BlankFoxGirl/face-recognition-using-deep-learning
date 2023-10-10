import os, datetime, time, cv2, imutils, log
import numpy as np

motionBg=None
MOTION_SENSITIVITY=10000
USE_V2_MOTION_DETECTION=True
DEBUG_FRAME=False
lastFrame=None

def setMotionSensitivity(sensitivity):
    global MOTION_SENSITIVITY
    MOTION_SENSITIVITY = sensitivity

def resetBackground():
    global motionBg
    motionBg = None

def isMotionDetected(frame):
    global motionBg, MOTION_SENSITIVITY
    return len(getMotion(frame)) != 0

def isMotionDetectedIn(motion):
    return len(motion) != 0

def getSignatureFrame(frame, firstFrame):
    signatures = []
    diff_frame = cv2.absdiff(frame, firstFrame)
    small = imutils.resize(diff_frame, width=32, height=18) # 576 Pixels to scan.
    # 16 square pixels per block with 36 blocks.
    # loop through each block and average the color values.
    # calculate x and y pixels per block.
    # blockX = int(small.shape[1] / 4)
    # blockY = int(small.shape[0] / 4)
    for x in range(32):
        for y in range(18):
            if np.all(small[y, x] > [10, 10, 10]) and np.any(small[y, x] > [20, 20, 20]):
                signatures.append({
                    "coords": (x, y),
                    "color": small[y, x]
                })
            else:
                small[y, x] = [255, 0, 0] # Set to odd colour.

    if len(signatures) > 0:
        groups = formatSignatureBlocksIntoSignatureGroups(signatures)
        log.debug("Signatures Dump", DUMP=formatSignatureGroupsIntoAverageColoursAndCoords(groups))

    return imutils.resize(small, width=1920, height=1080)

# Because, fuck python for not supporting "break 3" >:(
class ExitLoop3(Exception):
    pass
class ExitLoop2(Exception):
    pass
class ExitLoop(Exception):
    pass

def formatSignatureBlocksIntoSignatureGroups(signatureBlocks, minimumSignatureGroupSize=3):
    signatureGroups = {}
    signatureId = 0
    try:
        for signature in signatureBlocks:
            # are these coordinates adjacent to an existing set of coordinates in a signature group?
            if len(signatureGroups) == 0:
                signatureGroups[signatureId] = [signature]
                signatureId += 1
                continue
            try:
                for group in signatureGroups:
                    for coords in signatureGroups[group]:
                        if isCoordsAdjacent(signature["coords"], coords["coords"]):
                            signatureGroups[group].append(signature)
                            raise ExitLoop2
                signatureGroups[signatureId] = [signature]
                signatureId += 1
            except ExitLoop2:
                continue
    except ExitLoop3:
        pass

    toDel=[]
    for group in signatureGroups:
        if len(signatureGroups[group]) < minimumSignatureGroupSize:
            toDel.append(group)

    for group in toDel:
        del signatureGroups[group]

    return signatureGroups

def formatSignatureGroupsIntoAverageColoursAndCoords(signatureGroups):
    signatures={}
    for group in signatureGroups:
        colors=[]
        coords=[]
        for pixel in signatureGroups[group]:
            colors.append(pixel["color"])
            coords.append(pixel["coords"])
        signatures[group] = {
            "color": np.average(colors, axis=0),
            "coords": np.average(coords, axis=0)
        }
    return signatures

def isCoordsAdjacent(coords1, coords2):
    return abs(coords1[0] - coords2[0]) <= 1 and abs(coords1[1] - coords2[1]) <= 1

def drawMotion(frame, motion):
    for m in motion:
        cv2.rectangle(frame, (m["startX"] * 4, m["startY"] * 4), (m["endX"] * 4, m["endY"] * 4), (0, 0, 255), 2)
    return frame

def getMotionOnEveryNthFrame(frame, nthFrame, currentIdx):
    if (currentIdx + 1) % nthFrame != 1:
        return []
    return getMotion(frame)

def getGrey(frame):
    (fH, fW) = frame.shape[:2]

    quaterWidth = int(fW / 4)
    quaterHeight = int(fH / 4)

    frame = imutils.resize(frame, width=quaterWidth, height=quaterHeight)

    # Converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Converting gray scale image to blur 
    # so that change can be found easily
    gray = cv2.blur(gray, (5, 5))

    return gray

def getMotion(frame, motionBg=None, sensitivity=MOTION_SENSITIVITY, MIN_MOTION_AREA=5, MAX_MOTION_AREA=10, threshold=30, MOTION_THRESHOLD=30, CONTAINER_PADDING=20):
    if sensitivity != MOTION_SENSITIVITY:
        log.warning("Argument sensitivity in getMotion is no longer used. Please use MIN_MOTION_AREA and MAX_MOTION_AREA instead.", ["DEPRECATED"])
    if threshold != 30:
        log.warning("Argument threshold in getMotion is deprecated. Please use MOTION_THRESHOLD instead.", ["DEPRECATED"])
        MOTION_THRESHOLD=threshold

    if USE_V2_MOTION_DETECTION == True:
        if motionBg is not None:
            log.warning("Argument motionBg in getMotion is deprecated and no longer used in V2 Motion Detection.", ["DEPRECATED"])
        return getMotionV2(frame, MIN_MOTION_AREA, MAX_MOTION_AREA, MOTION_THRESHOLD, CONTAINER_PADDING)

    return getMotionV1(frame, motionBg, MIN_MOTION_AREA, MAX_MOTION_AREA, MOTION_THRESHOLD, CONTAINER_PADDING)

def getMotionV1(frame, motionBg, MIN_MOTION_AREA=5, MAX_MOTION_AREA=10, MOTION_THRESHOLD=30, CONTAINER_PADDING=20):
    # Initializing motion (no motion)
    motion = []

    # Converting color image to gray_scale image
    gray = getGrey(frame)

    # In first iteration we assign the value 
    # of static_back to our first frame
    if motionBg is None:
        motionBg = gray
        log.debug("BG SET")
        return motion

    # Difference between static background 
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(motionBg, gray)
  
    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 1)

    # Finding contour of moving object
    cnts,_ = cv2.findContours(thresh_frame.copy(),
             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < MIN_MOTION_AREA or cv2.contourArea(contour) > MAX_MOTION_AREA:
            continue

        log.debug("Motion detected with area {}".format(cv2.contourArea(contour)))
        (x, y, w, h) = cv2.boundingRect(contour)
        motion.append({
            "startX": max(x - CONTAINER_PADDING, 0),
            "startY": max(y - CONTAINER_PADDING, 0),
            "endX": x + w + CONTAINER_PADDING,
            "endY": y + h + CONTAINER_PADDING,
        })

    return motion

def getMotionV2(frame, MIN_MOTION_AREA=5, MAX_MOTION_AREA=10, MOTION_THRESHOLD=30, CONTAINER_PADDING=20):
    global lastFrame
    # Subtract the current frame from the last frame. lastFrame
    # Initializing motion (no motion)
    motion = []

    # Converting color image to gray_scale image
    gray = getGrey(frame)

    # In first iteration we assign the value 
    # of static_back to our first frame
    if lastFrame is None:
        lastFrame = gray
        log.debug("BG SET")
        return motion

    # Difference between static background 
    # and current frame(which is blurred)
    diff_frame = cv2.absdiff(lastFrame, gray)

    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 1)

    # Finding contour of moving object
    cnts,_ = cv2.findContours(thresh_frame.copy(),
             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < MIN_MOTION_AREA or cv2.contourArea(contour) > MAX_MOTION_AREA:
            continue

        log.debug("Motion detected with area {}".format(cv2.contourArea(contour)))
        (x, y, w, h) = cv2.boundingRect(contour)
        motion.append({
            "startX": max(x - CONTAINER_PADDING, 0),
            "startY": max(y - CONTAINER_PADDING, 0),
            "endX": x + w + CONTAINER_PADDING,
            "endY": y + h + CONTAINER_PADDING,
        })

    if len(motion) > 0 and DEBUG_FRAME == True:
        cv2.imwrite("captured/debug_last_{}.jpg".format(time.time()), lastFrame)
        cv2.imwrite("captured/debug_curr_{}.jpg".format(time.time()), gray)
        cv2.imwrite("captured/debug_diff_{}.jpg".format(time.time()), diff_frame)
        cv2.imwrite("captured/debug_thre_{}.jpg".format(time.time()), thresh_frame)

    # Update lastFrame with the current gray grabbed frame.
    lastFrame = gray
    return motion

def getArea(w, h):
    return w * h