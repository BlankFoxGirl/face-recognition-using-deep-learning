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
    # print("Rect (({},{}),({},{})) scaled by {} is (({},{}),({},{})".format(startX, startY, endX, endY, scale, newX, newY, newX + newWidth, newY + newHeight))

    if x > newX and x < (newWidth+newX) and y > newY and y < (newY + newHeight):
        # print("Point ({}x, {}y) is within scaled rect".format(x, y))
        return True
    # print("Point ({}x, {}y) is outside of scaled rect".format(x, y))
    return False
