import os, datetime, time, cv2

motionBg=None
MOTION_SENSITIVITY=10000

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

def drawMotion(frame, motion):
    for m in motion:
        cv2.rectangle(frame, (m["startX"], m["startY"]), (m["endX"], m["endY"]), (0, 0, 255), 2)
    return frame

def getMotionOnEveryNthFrame(frame, nthFrame, currentIdx):
    if (currentIdx + 1) % nthFrame != 1:
        return []
    return getMotion(frame)

def getGrey(frame):
    # Converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Converting gray scale image to GaussianBlur 
    # so that change can be find easily
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def getMotion(frame, motionBg, sensitivity=MOTION_SENSITIVITY, threashold=30):
    # Initializing motion (no motion)
    motion = []

    # Converting color image to gray_scale image
    gray = getGrey(frame)
  
    # In first iteration we assign the value 
    # of static_back to our first frame
    if motionBg is None:
        motionBg = gray
        print("BG SET")
        return motion
  
    # Difference between static background 
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(motionBg, gray)
  
    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, threashold, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
  
    # Finding contour of moving object
    cnts,_ = cv2.findContours(thresh_frame.copy(),
             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < sensitivity:
            continue
  
        (x, y, w, h) = cv2.boundingRect(contour)
        motion.append({
            "startX": x,
            "startY": y,
            "endX": x + w,
            "endY": y + h
        })

    return motion

def getArea(w, h):
    return w * h