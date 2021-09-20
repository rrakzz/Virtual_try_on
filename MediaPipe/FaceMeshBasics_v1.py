import cv2
import mediapipe as mp
import time
import numpy as np

#cap = cv2.VideoCapture("D:\\Face\\MediaPipe\\output_25fps.mp4") #"Videos/1.mp4"
pTime = 0

nu = 3

imgPath = f"D:\\Face\lips\\images\\{nu}.jpg"

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

lips = [61, 40, 37, 0, 267, 270, 291,
        375, 405, 17, 84, 91, 61,
        78, 80, 82, 13, 312, 310, 308,
        324, 402, 14, 87, 88, 78]

lips_up = [61, 40, 37, 0, 267, 270, 291]
        
while True:
    #success, img = cap.read()
    img = cv2.imread(imgPath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        #print(results.multi_face_landmarks)
        for faceLms in results.multi_face_landmarks:
            #mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec,drawSpec)
            ih, iw, ic = img.shape
            
            i, X,Y,Z = [], [], [], []
            points = []
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                if id in lips:
                    if id in lips_up:
                        X.append(int(lm.x*iw-lm.x*iw*0.5))
                        Y.append(int(lm.y*ih-lm.y*ih*0.5))                    
                    else:
                        X.append(int(lm.x*iw))
                        Y.append(int(lm.y*ih))
                    i.append(id)
                    points.append((int(lm.x*iw), int(lm.y*ih)))
            
            points1 = []
            for id in range(468):
                #points1.append((faceLms.landmark[lm].x*iw, faceLms.landmark[lm].y*ih))
                #print(lm, (faceLms.landmark[lm].x*iw, faceLms.landmark[lm].y*ih))
                
                
                #try:
                    #print(points1[-1], points1[-2])
                    #cv2.line(img, points1[-1], points1[-2], (0, 0, 255), 1)
                cv2.circle(img, (int(faceLms.landmark[id].x*iw), int(faceLms.landmark[id].y*ih)), 2, (255,0,0), 1)
                    #cv2.putText(img, f'fgfbfgbef', (100,100), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0), 1)
                #except:
                    #pass
                cv2.imshow("Image", img)
                cv2.waitKey(3)
        #cv2.putText(img, f'fdgmmriogierio', (100,100), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0), 1)            
        
        #points1.append(points1[0])    
        #points1 = np.reshape(points1, (-1, 1, 2))  
        #cv2.fillPoly(img, np.int32([points1]), (0, 0, 255), 8)        
        #cv2.line(img, points[0], points[-1], (0, 0, 255), 1)
        #cv2.circle(img, (int(lm.x*iw),int(lm.y*ih)), 1, (255,0,0), 2)
                #Z.append(face_landmarks.landmark[i].z)




    cTime = time.time()
    #fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 0), 3)
    #cv2.imshow("Image", img)
    #cv2.waitKey(1000)