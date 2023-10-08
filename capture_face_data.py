# import libraries
import os, datetime
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
os.remove("out.avi")
# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
# vs = VideoStream(src=0).start()
# vs = cv2.VideoCapture("rtsp://Xf6N0dIENhKR:Ha62gamgMX2f@10.2.1.202/live0", cv2.CAP_FFMPEG)
vs = cv2.VideoCapture("rtmp://camera1:1935/live/camera1", cv2.CAP_FFMPEG) # Kasa.
time.sleep(2.0) # Delay to let camera warm up.

# start the FPS throughput estimator
fps = FPS().start()
print("Video Stream Started.")
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))
# loop over frames from the video file stream
name = 'sarah-allen'
capture = False
captureAfter = 0

# Start;
print("Beginning capture in 15 seconds...")
capture = True
captureAfter = time.time() + 15
frameIndex = 0
while True:
		# update the FPS counter
	fps.update()

	# show the output frame
	# cv2.imshow("Frame", frame)
	ret, frame = vs.read()

	if frame is None:
		continue

	(h, w) = frame.shape[:2]
	# h = 1080
	# w = 1920

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	out.write(frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	if key == ord("s"):
		print("Stop capture")
		capture = False

	if key == ord("b"):
		print("Beginning capture in 15 seconds...")
		capture = True
		captureAfter = time.time() + 15

	if capture == False or time.time() < captureAfter:
		continue

	print("Capturing frame {}".format(frameIndex))
	frameIndex += 1
	nowDateTime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
	# grab the frame from the threaded video stream

	# imageBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255,
	# 	(96, 96), (0, 0, 0), swapRB=True, crop=False)

	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = imutils.resize(frame, width=600)
	cv2.imwrite('captured/{}/{}-{}-{}.jpg'.format(name, nowDateTime, name, frameIndex),frame)

# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()