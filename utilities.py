import os, base64, json, numpy as np, cv2, time
def slugToName(slug):
    return slug.replace("-", " ").title()

def getCenterPoint(startX, startY, endX, endY):
    return {
        "x": (startX + endX) / 2,
        "y": (startY + endY) / 2
    }

def pointInRectangle(x, y, startX, startY, endX, endY):
    if x > startX and x < endX and y > startY and y < endY:
        return True
    return False

def pointInScaledRectangle(x, y, startX, startY, endX, endY, scale):
    centerPosition = getCenterPoint(startX, startY, endX, endY)
    height = endY - startY
    width = endX - startX
    newX = centerPosition["x"] - (width * scale / 2)
    newY = centerPosition["y"] - (height * scale / 2)
    newHeight = height * scale
    newWidth = width * scale

    if x > newX and x < (newWidth+newX) and y > newY and y < (newY + newHeight):
        return True
    return False


def createRecursivePath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# Wraps base64.b64encode with ascii decoding.
def encodeImage(image):
    retval, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('ascii')

# Wraps base64.b64decode
def decodeImage(image):
    nparr = np.frombuffer(base64.b64decode(image), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Wraps json.dumps
def encodeMessage(message):
    message["sent"] = time.time()
    return json.dumps(message)

# Wraps json.loads
def decodeMessage(message):
    return json.loads(message)

def getPathWithoutFile(filename):
    path = filename.split("/")
    path.pop()
    return "/".join(path) + "/"