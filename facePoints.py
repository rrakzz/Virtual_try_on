import cv2
import numpy as np
import cv2
import pandas as pd


# This below mehtod will draw all those points which are from 0 to 67 on face one by one.
def drawPoints(image, faceLandmarks, startpoint, endpoint, face_mark, isClosed=False):
  #points = []
  #for i in range(startpoint, endpoint+1):
  #  point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
  #  points.append(point)
  #points = np.array(points, dtype=np.int32)
  #cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)
  pointsX = []
  pointsY = []
  for i in range(startpoint, endpoint+1):
    pointsX.append(faceLandmarks.part(i).x)  
    pointsY.append(faceLandmarks.part(i).y)
  cv2.rectangle(image, (min(pointsX), min(pointsY)), (max(pointsX), max(pointsY)), (255,0,0), 2)
  
  return [face_mark, min(pointsX), min(pointsY), max(pointsX), max(pointsY)]
  
  
  
# Use this function for 70-points facial landmark detector model
# we are checking if points are exactly equal to 68, then we draw all those points on face one by one
def facePoints(image, faceLandmarks):
    assert(faceLandmarks.num_parts == 68)
    #rects = []
    #drawPoints(image, faceLandmarks, 0, 16)           # Jaw line
    #drawPoints(image, faceLandmarks, 17, 21)          # Left eyebrow
    #drawPoints(image, faceLandmarks, 22, 26)          # Right eyebrow
    #drawPoints(image, faceLandmarks, 27, 30)          # Nose bridge
    #drawPoints(image, faceLandmarks, 30, 35)          # Lower nose
    #leye = drawPoints(image, faceLandmarks, 36, 41, 'left_eye', True)    # Left eye
    #reye = drawPoints(image, faceLandmarks, 42, 47, 'right_eye', True)    # Right Eye
    lips = drawPoints(image, faceLandmarks, 48, 59, 'lips', True)    # Outer lip
    #drawPoints(image, faceLandmarks, 60, 67, True)    # Inner lip
    #rects.append(leye)
    #rects.append(reye)
    #rects.append(lips)
    with open('rects.csv', 'w') as fp:
        #print(lips)
        fp.write(','.join('%s' % x for x in lips))
    #res = pd.DataFrame({'list': rects})
    #res.to_csv('rects.csv', index=False)
    
# Use this function for any model other than
# 70 points facial_landmark detector model
def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
  for p in faceLandmarks.parts():
    cv2.circle(im, (p.x, p.y), radius, color, -1)