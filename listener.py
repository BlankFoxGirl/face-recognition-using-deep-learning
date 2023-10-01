import os, subprocess, datetime, time, json
listenerProcesses = {}
CONFIG_FILE="listener.config.json"
running = True

def startCamera(camera):
    global listenerProcesses
    cameraName = camera["name"]
    if listenerProcesses is not None and cameraName in listenerProcesses:
        if (listenerProcesses[cameraName].poll() != None):
            del listenerProcesses[cameraName]
        else:
            return

    print("[CAMERA] Starting camera {}.".format(cameraName))
    cameraProcess = subprocess.Popen(["python", "/app/recognize_video.py", camera["url"], cameraName, str(camera["motion.sensitivity"]), str(camera["motion.threshold"]), str("True" if camera["record"] else "False")])
    listenerProcesses[cameraName] = cameraProcess

def startCameras(config):
    for camera in config["cameras"]:
        if camera["enabled"] == True:
            startCamera(camera)
        else:
            print("Camera {} is not enabled".format(camera["name"]))

config = json.load(open(CONFIG_FILE))

while running == True:
    time.sleep(5)
    startCameras(config)



print("Listener end")