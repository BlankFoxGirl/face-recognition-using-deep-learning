#!/usr/local/bin/python
# import libraries
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os, log

COMPARE_SIZE=100
PROCESS_AMOUNT=500
RUN_FRESH=False

# load serialized face detector
log.info("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
log.info("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# grab the paths to the input images in our dataset
log.info("Quantifying Faces...")
imagePaths = list(paths.list_images("dataset"))

# initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []
imageNames = []

# Load existing embeddings.
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

if RUN_FRESH == True or data == None or "paths" not in data:
	data = {"embeddings": [], "names": [], "paths": []}
	log.info("Running fresh.")

# initialize the total number of faces processed
total = 0
newProcessed = 0
tooSmall = 0
lowConfidence = 0
noFace = 0
# loop over the image paths
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

	# extract the person name from the image path
	if (i%PROCESS_AMOUNT == 0):
		log.info("Processing image {}/{}".format(i, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (COMPARE_SIZE, COMPARE_SIZE)), 1.0, (COMPARE_SIZE, COMPARE_SIZE),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				log.warning("Face too small. {}W {}H - {}".format(fW, fH, imagePath))
				os.remove(imagePath)
				tooSmall += 1
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			imageNames.append(imagePath)
			total += 1
		else:
			log.warning("Confidence too low. {} - {}".format(confidence, imagePath))
			lowConfidence += 1
			os.remove(imagePath)
	else:
		log.warning("No faces found. {}".format(imagePath))
		noFace += 1
		os.remove(imagePath)

# dump the facial embeddings + names to disk
log.info("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames, "paths": imageNames}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
log.info("[STATS] {} Missing Faces, {} Too Small, {} Low Confidence, {} Newly Processed, {} Total".format(noFace, tooSmall, lowConfidence, newProcessed, total))
