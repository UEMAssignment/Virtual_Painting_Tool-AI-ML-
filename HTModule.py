import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.pTime = 0
        self.cTime = 0

    def track_hands(self):
        while True:
            success, img = self.video.read()

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.hands.process(imgRGB)

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (25, 200, 255), cv2.FILLED)

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(img, "FPS:" + str(int(fps)), (10, 700), cv2.FONT_HERSHEY_PLAIN, 1, (0, 25, 255), 1)
            cv2.putText(img, "Enter 'Q' to exit...", (10, 715), cv2.FONT_HERSHEY_PLAIN, 1, (0, 25, 255), 1)
            cv2.imshow("Hand Tracking", img, )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


