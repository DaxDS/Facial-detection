import cv2

import mediapipe as mp
import time


cap = cv2.VideoCapture("video/FD 4.mp4")
ptime = 0

mpFD = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
FD = mpFD.FaceDetection()

while True:
    sucess, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FD.process(imgRGB)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpdraw.draw_detection(img, detection)
            #print(id, detection)
            #print(detection.score)
            print(detection.location_data.relative_bounding_box)

            x = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            y = int(x.xmin * w), int(x.ymin * h), \
            int(x.width * w), int(x.height * h)
            cv2.rectangle(img,y,(0,255,255),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (y[0], y[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    cv2.imshow("image",img)
    cv2.waitKey(1)