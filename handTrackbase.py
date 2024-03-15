import cv2
import mediapipe as mp
import time

video = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    succes, img = video.read()

    imgRGB = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
    result= hands.process(imgRGB)
    # print(result.multi_hand_landmarks) -- see the values
    cTime = 0
    pTime = 0

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx, cy = int (lm.x*w), int (lm.y*h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx,cy), 10, (25,200,255), cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3, (0,25,255),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

