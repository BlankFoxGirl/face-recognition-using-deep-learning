import math, cv2

WIDTH=1920
HEIGHT=1080
NUMBER_OF_QUADRANTS=400

# def draw():
#     output = '<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">'.format(WIDTH, HEIGHT)
#     for i in range(int(NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS)))):
#         for j in range(int(NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS)))):
#             output += drawQuadrant(i, j)
#     return output + "</svg>"

# def drawV2(image=None):
#     output = '<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">\n'.format(WIDTH, HEIGHT)

#     if image is not None:
#         output += '<image href="{}" width="{}" height="{}" />\n'.format(image, WIDTH, HEIGHT)

#     x = 700
#     y = 380
#     h = 990
#     w = 1300
#     hl = quadrantsInRectangle(x, y, w - x, h - y)
#     output += drawRect(x, y, w - x, h - y)

#     for i in range(NUMBER_OF_QUADRANTS):
#         quadrant = quadrantIdToXY(i)
#         output += drawQuadrant(quadrant["x"], quadrant["y"], quadrant["id"], str(i) in hl)

#     return output + "</svg>"

def drawToFrame(frame, DRAW_GRID=True, HIGHLIGHT_QUADRANTS=[]):
    for i in range(NUMBER_OF_QUADRANTS):
        quadrant = quadrantIdToXY(i)
        if DRAW_GRID == True:
            drawQuadrantToFrame(frame, (quadrant["x"], quadrant["y"]), i, str(i) in HIGHLIGHT_QUADRANTS)
    # return frame

def drawQuadrantToFrame(frame, coords=(0,0), id=None, highlight=False):
    quadrantWidth = math.ceil(WIDTH / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
    quadrantHeight = math.ceil(HEIGHT / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
    x = math.floor((coords[0]) * quadrantWidth)
    y = math.floor((coords[1]) * quadrantHeight)
    color=(255, 255, 255) if highlight == False else (255, 0, 0)
    if id is not None:
        cv2.putText(frame, "{}".format(id), (x+4, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        _draw(frame, (x, y), (x+quadrantWidth, y+quadrantHeight), color, 1)
    # return frame

def _draw(frame, coords=(0, 0), size=(), color=(0, 255, 0), stroke=1):
    cv2.rectangle(frame, coords, size, color, stroke)
    # return frame

# def drawQuadrant(x, y, id=None, highlight=False):
#     quadrantWidth = int(WIDTH / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
#     quadrantHeight = int(HEIGHT / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
#     x = int(x * quadrantWidth)
#     y = int(y * quadrantHeight)
#     if id is not None:
#         if NUMBER_OF_QUADRANTS < 100:
#             size='{} rect({}, {}, {}, {})'.format(quadrantXYToId(x, y), x, y, x+quadrantWidth, y+quadrantHeight)
#         else:
#             size='{}'.format(quadrantXYToId(x, y))
#         hl = 'fill:rgb(255,0,0);' if highlight == True else ''
#         return '<g><rect style="stroke: rgb(255,255,255);stroke-width:3;opacity: 0.2;{}" width="{}" height="{}" x="{}" y="{}" /><text font-size="12" x="{}" y="{}" fill="#fff">{}</text></g>\n'.format(hl, quadrantWidth, quadrantHeight, x, y, x + 6, y + 22, size)

#     return '<rect style="stroke: rgb(255,255,255);stroke-width:3;opacity: 0.2;" width="{}" height="{}" x="{}" y="{}" />\n'.format(quadrantWidth, quadrantHeight, x, y)

def quadrantIdToXY(quadrantId):
    return {
        "x": int(quadrantId) % (math.sqrt(NUMBER_OF_QUADRANTS)),
        "y": math.floor(int(quadrantId) / (math.sqrt(NUMBER_OF_QUADRANTS))),
        "id": int(quadrantId)
    }

def quadrantXYToId(x, y):
    quadrantWidth = int(WIDTH / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
    quadrantHeight = int(HEIGHT / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))

    y = math.floor(y / quadrantHeight)
    x = math.floor(x / quadrantWidth)

    return "{}".format(math.floor((y * math.sqrt(NUMBER_OF_QUADRANTS)) + x))

def quadrantsInRectangle(rX, rY, rWidth, rHeight):
    quadrantWidth = int(WIDTH / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
    quadrantHeight = int(HEIGHT / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))

    xCoord = math.floor(rX / quadrantWidth)
    yCoord = math.floor(rY / quadrantHeight)
    xEndCoord = math.floor((rX + rWidth) / quadrantWidth) + 1
    yEndCoord = math.floor((rY + rHeight) / quadrantHeight)

    quadrants = []
    for i in range(yCoord, yEndCoord):
        for j in range(xCoord, xEndCoord):
            # If the supplied rectangle occupies more than 25% of the quadrant add the quadrant to the list

            quadrantId = quadrantXYToId(j * quadrantWidth, i * quadrantHeight)
            if (rectangleOccupiesQuadrant(quadrantId, rX, rY, rWidth, rHeight) == True):
                if quadrantId not in quadrants:
                    quadrants.append(quadrantId)

    return quadrants

def rectangleOccupiesQuadrant(quadrantId, x, y, width, height):
    # create a smaller rectangle based on the x, w of the rectangle and the quadrant endX and endY.
    quadrantWidth = int(WIDTH / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
    quadrantHeight = int(HEIGHT / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
    quadrant = quadrantIdToXY(quadrantId)
    quadrantArea = quadrantWidth * quadrantHeight
    quadrantStartX = quadrant["x"]*quadrantWidth
    quadrantStartY = quadrant["y"]*quadrantHeight
    quadrantEndX = quadrantStartX+quadrantWidth
    quadrantEndY = quadrantStartY+quadrantHeight

    rect = {
        "startX": x if x > quadrantStartX else quadrantStartX,
        "startY": y if y > quadrantStartY else quadrantStartY,
        "endX": (x + width if (x + width) < quadrantEndX else quadrantEndX),
        "endY": (y + height if (y + height) < quadrantEndY else quadrantEndY)
    }
    rectWidth = rect["endX"] - rect["startX"]
    rectHeight = rect["endY"] - rect["startY"]
    rectangleArea = rectWidth * rectHeight
    quadrantOccupancyPercent = abs(1 - ((quadrantArea - rectangleArea) / quadrantArea))

    return quadrantOccupancyPercent > 0.25

def rectFromQuadrants(quadrants):
    startQuadrant = quadrantIdToXY(quadrants[0])
    endQuadrant = quadrantIdToXY(quadrants[-1])
    quadrantWidth = int(WIDTH / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))
    quadrantHeight = int(HEIGHT / (NUMBER_OF_QUADRANTS / (math.sqrt(NUMBER_OF_QUADRANTS))))

    return {
        "startX": startQuadrant["x"] * quadrantWidth,
        "startY": startQuadrant["y"] * quadrantHeight,
        "endX": (endQuadrant["x"] * quadrantWidth) + quadrantWidth,
        "endY": (endQuadrant["y"] * quadrantHeight) + quadrantHeight,
        "area": (((endQuadrant["x"] - startQuadrant["x"]) * quadrantWidth) + quadrantWidth) * (((endQuadrant["y"] - startQuadrant["y"]) * quadrantHeight) + quadrantHeight),
        "screen": (HEIGHT * WIDTH),
        "viewportPercent": math.ceil((((endQuadrant["x"] - startQuadrant["x"]) * quadrantWidth) + quadrantWidth) * (((endQuadrant["y"] - startQuadrant["y"]) * quadrantHeight) + quadrantHeight) / (HEIGHT * WIDTH) * 100)
    }

def centerQuadrant(quadrants):
    rect = rectFromQuadrants(quadrants)
    x = (rect["endX"] - rect["startX"]) / 2 + rect["startX"]
    y = (rect["endY"] - rect["startY"]) / 2 + rect["startY"]
    return quadrantXYToId(x, y)

# def drawRect(x, y, width, height):
#     return '<rect style="stroke: rgb(255,255,255);stroke-width:3;opacity: 0.1;fill: rgb(0, 255, 0);" width="{}" height="{}" x="{}" y="{}" />\n'.format(width, height, x, y)

# print(drawV2("file:///Users/sarahjabado/development/face-recognition-using-opencv/captured/Snapshots/Camera2_sarah-allen_20231001235916_9953.jpg"))