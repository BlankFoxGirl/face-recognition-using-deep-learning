import motion_detection as mp, log
import cv2, datetime, utilities, quadrants
videoMaxLength = 3 * 60 * 30 # 3 minutes @ 30 FPS.
videoLength = 0

def startRecording(name, out, INPUT_WIDTH=1920, INPUT_HEIGHT=1080, RECORD_VIDEO=True, MY_ID="Camera"):
    now = datetime.datetime.now()
    if out is not None or RECORD_VIDEO == False:
        if RECORD_VIDEO == False:
            p = utilities.createRecursivePath("captured/{}/{}/{}/{}/{}".format(MY_ID, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H"), now.strftime("%M")))
            currentVideoFilename = "{}/{}.avi".format(p, name)
            return (out, currentVideoFilename)
        return (out, "")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    p = utilities.createRecursivePath("captured/{}/{}/{}/{}/{}".format(MY_ID, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H"), now.strftime("%M")))
    currentVideoFilename = "{}/{}.avi".format(p, name)
    out = cv2.VideoWriter("{}/{}.avi".format(p, name),fourcc, 20.0, (INPUT_WIDTH,INPUT_HEIGHT))
    log.info("[CAPTURE] Recording to file: {}/{}.avi".format(p, name))
    return (out, currentVideoFilename)

def stopRecording(out, RECORD_VIDEO=True):
    if out is None or RECORD_VIDEO == False:
        return out
    out.release()
    del out
    out = None
    log.info("[CAPTURE] Stopped recording.")
    return out

def writeFrame(frame, motion, out, motionSignatures, RECORD_VIDEO=True, DRAW_FACE_BOXES=True, IDENTITY_FRAME_HITS=5, PEOPLE_CONFIG={}):
    global videoLength, videoMaxLength
    if out is None or RECORD_VIDEO == False:
        return out

    mp.drawMotion(frame, motion)

    frame = drawInFrame(frame, motion, False, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, PEOPLE_CONFIG)

    out.write(frame)
    videoLength += 1
    if videoLength >= videoMaxLength:
        out = stopRecording(out, RECORD_VIDEO)
        videoLength = 0
    return out

def drawInFrame(frame, motion, motionSignatures, DRAW_FACE_BOXES=True, IDENTITY_FRAME_HITS=5, PEOPLE_CONFIG={}):
    now = datetime.datetime.now()
    activeQuadrants = []
    if motionSignatures is not False:
        for m in motionSignatures:
            identifiedMotion = None

            if "quadrants" in motionSignatures[m]:
                activeQuadrants += (motionSignatures[m]["quadrants"])
                for mot in motion:
                    # do.
                    cq = quadrants.centerQuadrant(motionSignatures[m]["quadrants"])
                    if (quadrants.rectangleOccupiesQuadrant(cq, mot["startX"], mot["startY"], mot["endX"] - mot["startX"], mot["endY"] - mot["startY"]) == True):
                        identifiedMotion = mot
                        break
            
            # if motionSignatures[m]["lastSeen"] < now - datetime.timedelta(seconds=1) and "x" in motionSignatures[m]:
            #     cv2.putText(frame, "Sig: {}".format(m), (motionSignatures[m]["x"], motionSignatures[m]["y"] + motionSignatures[m]["h"] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)

            if "people" not in motionSignatures[m]:
                continue
            peopleFound = motionSignatures[m]["people"]
            for name in peopleFound:
                idFrames = IDENTITY_FRAME_HITS
                if name in PEOPLE_CONFIG and "identity-frames" in PEOPLE_CONFIG[name]:
                    idFrames = PEOPLE_CONFIG[name]["identity-frames"]

                if peopleFound[name]["hits"] >= idFrames:
                    # draw the bounding box of the face along with the associated probability
                    text = "{}: {:.2f}%".format(utilities.slugToName(peopleFound[name]["name"]), peopleFound[name]["proba"] * 100)

                    startX = peopleFound[name]["startX"]
                    startY = peopleFound[name]["startY"]
                    endX = peopleFound[name]["endX"]
                    endY = peopleFound[name]["endY"]
                    if identifiedMotion is not None:
                        startX = identifiedMotion["startX"]*4
                        startY = identifiedMotion["startY"]*4
                        endX = identifiedMotion["endX"]*4
                        endY = identifiedMotion["endY"]*4

                    y = endY if endY < 1080 else 1080 # Put it at the bottom.
                    if DRAW_FACE_BOXES == True:
                        # Draw shadow under box.
                        cv2.rectangle(frame, (startX-1, startY-1), (endX+1, endY+1),
                            (0, 0, 0), 5)

                        # Draw box.
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 255, 0), 1)

                    cv2.rectangle(frame, (startX, y), (endX, y - 30),
                            (0, 0, 255), -1)

                    # Draw text.
                    cv2.putText(frame, text, (startX + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # quadrants.drawToFrame(frame, False, activeQuadrants)
    return frame