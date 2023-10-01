import motion_detection as mp
import cv2, datetime, utilities
videoMaxLength = 3 * 60 * 30 # 3 minutes @ 30 FPS.
videoLength = 0

def startRecording(name, out, INPUT_WIDTH=1920, INPUT_HEIGHT=1080, RECORD_VIDEO=True):
    if out is not None or RECORD_VIDEO == False:
        return out
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("captured/Videos/{}.avi".format(name),fourcc, 20.0, (INPUT_WIDTH,INPUT_HEIGHT))
    print("[CAPTURE] Recording to file: captured/Videos/{}.avi".format(name))
    return out

def stopRecording(out, RECORD_VIDEO=True):
    if out is None or RECORD_VIDEO == False:
        return out
    out.release()
    del out
    out = None
    return out

def writeFrame(frame, motion, out, peopleFound, RECORD_VIDEO=True, DRAW_FACE_BOXES=True, IDENTITY_FRAME_HITS=5, PEOPLE_CONFIG={}):
    global videoLength, videoMaxLength
    if out is None or RECORD_VIDEO == False:
        return out

    mp.drawMotion(frame, motion)

    frame = drawInFrame(frame, peopleFound, DRAW_FACE_BOXES, IDENTITY_FRAME_HITS, PEOPLE_CONFIG)

    out.write(frame)
    videoLength += 1
    if videoLength >= videoMaxLength:
        out = stopRecording(out, RECORD_VIDEO)
        videoLength = 0
    return out

def drawInFrame(frame, peopleFound, DRAW_FACE_BOXES=True, IDENTITY_FRAME_HITS=5, PEOPLE_CONFIG={}):
    now = datetime.datetime.now()
    for name in peopleFound:
        if name == "unknown":
            continue

        if peopleFound[name]["lastSeen"] < now - datetime.timedelta(seconds=1):
            continue

        idFrames = IDENTITY_FRAME_HITS
        if name in PEOPLE_CONFIG and "identity-frames" in PEOPLE_CONFIG[name]:
            idFrames = PEOPLE_CONFIG[name]["identity-frames"]

        if peopleFound[name]["hits"] >= idFrames:
            # draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(utilities.slugToName(peopleFound[name]["name"]), peopleFound[name]["proba"] * 100)
            y = peopleFound[name]["moveY"] - 10 if peopleFound[name]["moveY"] - 10 > 10 else peopleFound[name]["moveY"] + 10

            if DRAW_FACE_BOXES == True:
                # Draw shadow under box.
                cv2.rectangle(frame, (peopleFound[name]["startX"]-1, peopleFound[name]["startY"]-1), (peopleFound[name]["endX"]+1, peopleFound[name]["endY"]+1),
                    (0, 0, 0), 5)

                # Draw box.
                cv2.rectangle(frame, (peopleFound[name]["startX"], peopleFound[name]["startY"]), (peopleFound[name]["endX"], peopleFound[name]["endY"]),
                    (0, 255, 0), 1)

            # Draw shadow under text.
            cv2.putText(frame, text, (peopleFound[name]["moveX"]-1, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 6)

            # Draw text.
            cv2.putText(frame, text, (peopleFound[name]["moveX"], y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame