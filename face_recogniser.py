import datetime, cv2, time, numpy as np, log

def getDetectedFaces(detector, frame):
    (h, w) = frame.shape[:2]
    detector.setInputSize((w, h))
    return detector.detect(frame)

def getRecognisedFaces(
        frame,
        detections,
        embedder,
        recognizer,
        le,
        peopleConfig,
        offsetX=0,
        offsetY=0,
        MIN_FACE_WIDTH=20,
        MIN_FACE_HEIGHT=20,
        UP_SAMPLE_MULTIPLIER=1
    ):
    recognisedFaces = []
    if detections[1] is None:
        return recognisedFaces
    for idx, detection in enumerate(detections[1]):
        confidence = detection[-1]
        startX = int(detection[0]) * UP_SAMPLE_MULTIPLIER
        startY = int(detection[1]) * UP_SAMPLE_MULTIPLIER
        endX = (int(detection[0]) + int(detection[2])) * UP_SAMPLE_MULTIPLIER
        endY = (int(detection[1]) + int(detection[3])) * UP_SAMPLE_MULTIPLIER

        if confidence < 0.6:
            log.info("[DETECT] Confidence too low at ({}x,{}y),({}x,{}y) {}".format(startX, startY, endX, endY, confidence))
            continue

        # extract the face ROI
        face = frame[startY:endY, startX:endX]
        if face is None:
            log.info("[DETECT] Couldn't extract face from frame with ({}x,{}y),({}x,{}y)".format(startX, startY, endX, endY))
            continue

        (fH, fW) = face.shape[:2]
        (mH, mW) = frame.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < MIN_FACE_WIDTH or fH < MIN_FACE_HEIGHT:
            log.info("[DETECT] Face too small. {}W {}H".format(fW, fH))
            continue

        # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
        try:
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
        except:
            log.exception("[DETECT] Failed to create faceBlob at {}x{} - {}x{}".format(startX, startY, endX, endY))
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
                log.debug("Probability {} less than {} for {}".format(proba, peopleConfig["people"][name]["required-probability"], name))
                if proba < peopleConfig["people"][name]["training.ignore.threashold"]:
                    continue # Ignore.
                name = "unknown" # Override name if probability is too low to identify this person.

        recognisedFaces.append({
            "name": name,
            "proba": proba,
            "face": {
                "x": startX + offsetX,
                "y": startY + offsetY,
                "w": fW,
                "h": fH
            },
            "location": {
                "x": offsetX,
                "y": offsetY,
                "w": offsetX + mW,
                "h": offsetY + mH
            },
            "moveX": offsetX,
            "moveY": offsetY,
            "endMoveX": offsetX + mW,
            "endMoveY": offsetY + mH,
            "detected": time.time()
        })

    return recognisedFaces