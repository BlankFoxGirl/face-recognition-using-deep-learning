import os
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