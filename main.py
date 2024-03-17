import cv2
import numpy as np
import os
import HT_Module as htm
import time

folderPath = "header"
myList = os.listdir(folderPath)
# print(myList)

overlayList = []
for imfath in myList:
    image = cv2.imread(f"{folderPath}/{imfath}")
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]


paintThickness = 15
eraserThickness = 50
paintColor = (0, 0, 255)
eraseColor = (0, 0, 0)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector()

while True:
    # import the frames
    _, img = cap.read()
    # img = cv2.flip(img, 1)

    # find hand landmarks
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # tip of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # If selection mode -  two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection mode")
            xp, yp = 0, 0
            # checking for click
            if y1 < 80:
                if 5 < x1 < 80:
                    imgCanvas.fill(0)
                if 95 < x1 < 170:
                    header = overlayList[4]
                    paintColor = (255, 255, 255)
                if 185 < x1 < 265:
                    header = overlayList[0]
                    paintColor = (0, 0, 255)
                elif 282 < x1 < 361:
                    header = overlayList[1]
                    paintColor = (0, 255, 0)
                elif 376 < x1 < 455:
                    header = overlayList[2]
                    paintColor = (255, 0, 0)
                elif 474 < x1 < 550:
                    header = overlayList[3]
                    paintColor = (0, 255, 255)
                elif 582 < x1 < 642:
                    paintThickness = 20
                    eraserThickness = 70
                elif 655 < x1 < 715:
                    paintThickness = 15
                    eraserThickness = 50
                elif 734 < x1 < 790:
                    paintThickness = 10
                    eraserThickness = 30

        # If drawing mode -  index finger are up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, paintColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if paintColor == (255, 255, 255):
                cv2.line(img, (xp, yp), (x1, y1), paintColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), eraseColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), paintColor, paintThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), paintColor, paintThickness)

            xp, yp = x1, y1

    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the image at top
    # 1. clear - 5 - 80, 2. erase - 95 - 170, 3. red - 185 - 265, 4. green - 282 - 361, 5. blue - 376 - 455, 6. yellow - 474 - 550
    # brush-size - 1. 582 - 642, 2. 655 - 715, 3. 734 - 790

    img[0: 80, 0: 1280] = header
    cv2.imshow("Image", img)
    # cv2.imshow("ImageCanvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
