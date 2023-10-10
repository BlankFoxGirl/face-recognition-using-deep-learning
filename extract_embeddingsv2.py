#!/usr/local/bin/python
'''
NOT IN USE.
'''
# import libraries
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os, math
import face_recognition, json

COMPARE_SIZE=100
PROCESS_AMOUNT=200
RUN_FRESH=False

def saveToFile():
    global knownNames, knownEmbeddings, imageNames
    data = {"embeddings": knownEmbeddings, "names": knownNames, "paths": imageNames}
    print("[INFO] serializing {} encodings...".format(len(knownEmbeddings)))

    # with open("output/embeddingsv2.json", "w") as f:
    #     f.write(json.dumps(data))

    with open("output/embeddingsv2.pickle", "wb") as f2:
        f2.write(pickle.dumps(data))

def getBatchEncodings(images, name, path=None):
    global knownNames, knownEmbeddings, imageNames, total, noFace
    # ensure at least one face was found
    imgChunks = np.array_split(np.array(images), math.ceil(len(images)/PROCESS_AMOUNT))
    c = 0

    for chunk in imgChunks:
        imgs = []
        detections = []
        c += 1
        for img in chunk:
            im=face_recognition.load_image_file(img)
            imgs.append(np.array(im))

        print("--[CHUNK: {}/{}] Processing images {}-{}/{}".format(c, len(imgChunks), max(((c - 1) * PROCESS_AMOUNT) + 1, 1), min(c * PROCESS_AMOUNT, len(images)), len(images)))

        try:
            print("Attempting to process images in batch...")
            detections = face_recognition.batch_face_locations(imgs, number_of_times_to_upsample=0, batch_size=PROCESS_AMOUNT)
            print("Batch successful.")
        except Exception as e:
            print("Batch failed. {}, switching to sequential processing.".format(e))
            i = 0
            for img in imgs:
                imageIndex = i + ((c - 1) * len(imgs))
                print("--[{}/{}] Processing image {}.".format(imageIndex, len(images), chunk[i]))
                detections.append(face_recognition.face_locations(img, number_of_times_to_upsample=0))
                i += 1

        if len(detections) <= 0:
            print("--[NO_FACES] No faces detected.")
            continue

        for i in range(len(imgs)):
            imageIndex = i + ((c - 1) * len(imgs))
            if len(detections[i]) >= 1:
                (startX, startY, endX, endY) = detections[i][0]

                # add the name of the person + corresponding face embedding to their respective lists
                knownNames.append(name)
                print("--[{}/{}] Encoding image {}.".format(imageIndex, len(images), images[imageIndex]))
                imgs2 = cv2.imread(images[imageIndex])
                if imgs2 is None:
                    print("--[{}/{}] Image {} is Missing.".format(imageIndex, len(images), images[imageIndex]))
                    continue
                imgs2 = imutils.resize(imgs2, width=600)
                knownEmbeddings.append(face_recognition.face_encodings(imgs2, known_face_locations=[(startY, endX, endY, startX)], model="large")[0])
                imageNames.append(images[imageIndex])
                total += 1
            else:
                print("--[{}/{}] No face found in image {}.".format(imageIndex, len(images), images[imageIndex]))
                if os.path.exists(images[imageIndex]):
                    os.remove(images[imageIndex])
                noFace += 1

        saveToFile() # Save memory to file after each chunk is processed.

# grab the paths to the input images in our dataset
print("Quantifying Faces...")
imagePaths = list(paths.list_images("dataset"))

# initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []
imageNames = []

# Load existing embeddings.
data = pickle.loads(open("output/embeddingsv2.pickle", "rb").read())
# data = json.load(open("output/embeddingsv2.json", "r"))

if RUN_FRESH == True or data == None or "paths" not in data:
    data = {"embeddings": [], "names": [], "paths": []}
    print("Running fresh.")

# initialize the total number of faces processed
total = 0
newProcessed = 0
tooSmall = 0
lowConfidence = 0
noFace = 0
existingProcessed = 0

images = {}
# loop over the image paths
print("Loading images...")
for (i, imagePath) in enumerate(imagePaths):
    if imagePath in data["paths"]:
        # print("Skipping {}".format(imagePath))
        idx = data["paths"].index(imagePath)
        knownNames.append(data["names"][idx])
        knownEmbeddings.append(data["embeddings"][idx])
        imageNames.append(imagePath)
        total += 1
        continue
    else:
        newProcessed += 1
        total += 1

    name = imagePath.split(os.path.sep)[-2]
    if name not in images:
        images[name] = []
        print("Loading images for {}...".format(name))

    images[name].append(imagePath)

existingProcessed = total - newProcessed
for person in images:
    print("Processing images for {}...".format(person))
    getBatchEncodings(images[person], person)
    print("Images for {} processed.".format(person))

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames, "paths": imageNames}

saveToFile()

print("[STATS] {} Missing Faces, {} Too Small, {} Low Confidence, {} Newly Processed, {} Total".format(noFace, tooSmall, lowConfidence, newProcessed, total))
