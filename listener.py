import subprocess, time, json
import log
listenerProcesses = {}
CONFIG_FILE="listener.config.json"
PROCESS_FILE="/app/recognise_video_phase1.py"

running = True

def startCamera(camera):
    global listenerProcesses
    cameraName = camera["name"]
    if listenerProcesses is not None and cameraName in listenerProcesses:
        if (listenerProcesses[cameraName].poll() != None):
            del listenerProcesses[cameraName]
        else:
            return

    log.debug("[CAMERA] Starting camera {}.".format(cameraName))
    args = [
        "python",
        PROCESS_FILE,
        "-i",
        camera["url"],
        "-n",
        cameraName,
        "-t",
        str(camera["motion.threshold"]),
        "-r",
        str("0" if camera["record"] != True else "1"),
        '--motion-min-area',
        str(camera["motion.area.minimum"]),
        '--motion-max-area',
        str(camera["motion.area.maximum"]),
    ]
    log.info("Starting camera {} with command `{}`".format(cameraName, " ".join(args)))
    cameraProcess = subprocess.Popen(args)
    listenerProcesses[cameraName] = cameraProcess

def startCameras(config):
    for camera in config["cameras"]:
        if camera["enabled"] == True:
            startCamera(camera)

config = json.load(open(CONFIG_FILE))

while running == True:
    time.sleep(5)
    startCameras(config)

log.info("Listener end")