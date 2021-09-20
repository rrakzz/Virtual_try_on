#USAGE: python facial_68_Landmark.py

import dlib,cv2
import numpy as np
#from facePoints import facePoints
import glob, os
import pandas as pd

Model_PATH = "shape_predictor_68_face_landmarks.dat"
Input_path1 = "1.jpg"
Input_path2 = "1_tryon.jpg"
Output_path = "1_out.jpg"

# This below mehtod will draw all those points which are from 0 to 67 on face one by one.
def drawPoints(image, faceLandmarks, startpoint, endpoint, face_mark, isClosed=False):
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

    with open('temp.csv', 'w') as fp:
        fp.write(','.join('%s' % x for x in lips))
    
# Use this function for any model other than
# 70 points facial_landmark detector model
def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
  for p in faceLandmarks.parts():
    cv2.circle(im, (p.x, p.y), radius, color, -1)
    
def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
  with open(fileName, 'w') as f:
    for p in faceLandmarks.parts():
      f.write("%s %s\n" %(int(p.x),int(p.y)))

  f.close()

def compare(img, img1):
    count = 0
    change_count = 0
    for x in range(rects[1][0]+2, rects[3][0]-2):
        for y in range(rects[2][0]+2, rects[4][0]-2):
            count = count + 1
            if int(img[y, x][0]) != int(img1[y, x][0]):
                change_count = change_count + 1
    
    if change_count/count > 0.7:
        return round(change_count/count, 2), 'PASS' 
    elif change_count/count < 0.5:
        return round(change_count/count, 2), 'FAIL' 
    else:
        return round(change_count/count, 2), 'BORDER_LINE'       

# now from the dlib we are extracting the method get_frontal_face_detector()
# and assign that object result to frontalFaceDetector to detect face from the image with 
# the help of the 68_face_landmarks.dat model
frontalFaceDetector = dlib.get_frontal_face_detector()


# Now the dlip shape_predictor class will take model and with the help of that, it will show 
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

img= cv2.imread(Input_path1)
imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Now this line will try to detect all faces in an image either 1 or 2 or more faces
allFaces = frontalFaceDetector(imageRGB, 0)

#print("List of all faces detected: ",len(allFaces))

# List to store landmarks of all detected faces
allFacesLandmark = []

# Below loop we will use to detect all faces one by one and apply landmarks on them

for k in range(0, len(allFaces)):
    # dlib rectangle class will detecting face so that landmark can apply inside of that area
    faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
      int(allFaces[k].right()),int(allFaces[k].bottom()))

    # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
    detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)

    # Svaing the landmark one by one to the output folder
    allFacesLandmark.append(detectedLandmarks)

    # Now finally we drawing landmarks on face
    facePoints(img, detectedLandmarks)

    
cv2.imwrite(Output_path, img)
cv2.imshow("Face landmark result", img)

# Pause screen to wait key from user to see result
cv2.waitKey(0)
cv2.destroyAllWindows()

# lakme_out_file = Input_path2 + file.split('\\')[-1]
img1 = cv2.imread(Input_path2)

rects = pd.read_csv('temp.csv', header = None)

cv2.rectangle(img1, (rects[1][0],rects[2][0]), (rects[3][0], rects[4][0]), (255,0,0), 2)
cv2.imshow("Face landmark result on Lakme output", img1)

percentage_match, status = compare(img, img1)
print(percentage_match, status)
# Pause screen to wait key from user to see result
cv2.waitKey(0)
cv2.destroyAllWindows()   
    