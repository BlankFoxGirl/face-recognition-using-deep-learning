import argparse, log, numpy as np, cv2, vsms as mem, utilities, time
import paho.mqtt.client as mqtt
import face_recogniser as fr, pickle
import events as ev

COMPARE_SIZE=100

parser = argparse.ArgumentParser()
parser.add_argument('--host', '-i', help="Broker hostname / IP Address", type=str, default="mqtt", required=False)
parser.add_argument('--port', '-p', help="Broker port", type=int, default=1883, required=False)
parser.add_argument('--ttl', '-t', help="Connection TTL", type=int, default=60, required=False)

ARGS=parser.parse_args()

broker=ARGS.host
port=ARGS.port
timelive=ARGS.ttl

log.setId("Worker")

# load serialized face embedding model
log.info("Loading Face Recognizer...", ["WAIT"])
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

log.info("Connecting to broker {}:{} with timelive {}".format(broker,port,timelive), ["START"])

lastKeepalive = time.time()

def recogniseFace(data):
    frame = utilities.decodeImage(data["frame"])
    detections = data["detections"]
    camera = data["id"]

    if frame is None:
        log.warning("Invalid frame", ["RECOGNITION"])
        return
    detections = [detections, np.array(detections)]
    appendToLogFile(fr.getRecognisedFaces(frame, detections, embedder, recognizer, le, UP_SAMPLE_MULTIPLIER=4, ZONE=camera), data["filename"])

def appendToLogFile(data, filename):
    # Get the path but not the filename
    path = utilities.getPathWithoutFile(filename)
    log.info("Appending to log file {}people-stream".format(path), ["LOG"])
    logFile = open(path + "people-stream", "a")
    for line in data:
        logFile.write(utilities.encodeMessage(line) + "\n")
    logFile.close()

def on_connect(client, userdata, flags, rc):
    log.info("Connected with result code {}".format(rc), ["MQTT"])
    client.subscribe("/data")
    log.success("Successfully subscribed to /data", ["READY"])

def on_disconnect(client, userdata, flags, rc):
    log.info("Disconnected with result code {}".format(rc), ["MQTT"])

def on_message(client, userdata, msg):
    global lastKeepalive
    lastKeepalive = time.time()
    data = msg.payload.decode()
    if data is None:
        log.warning("Invalid data {}".format(data), ["RECEIVED"])
        return
    data = utilities.decodeMessage(data)
    log.debug("Message", ["RECEIVED"])

    if data["event"] == ev.PING:
        log.debug("PONG {}".format(data["id"]), ["PING"])
    elif data["event"] == ev.DETECTION:
        log.info("Received detections event with frame of size {}KB.".format(len(data["frame"]) / 1024), ["RECOGNITION"])
        recogniseFace(data)
    elif data["event"] == ev.MOTION:
        log.info("Received motion event", ["MOTION", data["id"]])
    elif data["event"] == ev.CAMERA_INITIALISED:
        log.success("Received camera initialised event for {}.".format(data["id"]), ["CAMERA"])
    elif data["event"] == ev.RECORDING_STARTED:
        log.info("Recording to file: {}".format(data["filename"]), ["CAPTURE", data["id"]])
    elif data["event"] == ev.RECORDING_STOPPED:
        log.info("Finished recording to file: {}".format(data["filename"]), ["CAPTURE", data["id"]])

client = mqtt.Client()
client.connect(broker,port=port,keepalive=timelive)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

run = True
while run:
    client.loop(timeout=1)
    if lastKeepalive < time.time() - 45:
        log.error("Keepalive timed out. Exiting.")
        exit(1)