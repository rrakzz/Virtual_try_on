import cv2
import dlib
import numpy as np
from PIL import Image
from PIL import Image, ImageFilter
import math

def AngleBtw2Points(pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return math.degrees(math.atan2(changeInY,changeInX))

def get_eyelashes_coord(eye_pts, width, height, side):
    p1 = [eye_pts[0][0], eye_pts[0][1]]
    p2 = [eye_pts[3][0], eye_pts[3][1]]

    wx = abs(eye_pts[3][0] - eye_pts[0][0])/2
    wy = abs(eye_pts[3][1] - eye_pts[0][1])/2

    if side == 'left':
        q1 = [eye_pts[1][0], eye_pts[0][1]+wy]
        q2 = [eye_pts[1][0], eye_pts[1][1]]
    else:
        q1 = [eye_pts[2][0], eye_pts[3][1]+wy]
        q2 = [eye_pts[2][0], eye_pts[2][1]]        

    rot = AngleBtw2Points(p1, p2)
    eye_w = width/math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    eye_h = height/math.sqrt(((q1[0]-q2[0])**2)+((q1[1]-q2[1])**2))
    return rot, eye_w, eye_h

def lashes_change_color(imagename, rgb, alpha):
    img = cv2.imread(imagename)
    # Add alpha layer with OpenCV
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) 
    # Set alpha layer semi-transparent with Numpy indexing, B=0, G=1, R=2, A=3
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j][0] != 0 or img[i,j][1] != 0 or img[i,j][2] != 0 :
                bgra[i,j][0] = rgb[0]
                bgra[i,j][1] = rgb[1]
                bgra[i,j][2] = rgb[2]
                bgra[i,j][3] = alpha
            else:
                bgra[i,j][0] = 255
                bgra[i,j][1] = 255
                bgra[i,j][2] = 255
                bgra[i,j][3] = 0
    cv2.imwrite(f'trans_{imagename}',bgra)
    return

lashes_change_color('lashes_up1.png', [0,0,0], 255)
lashes_change_color('lashes_down.png', [0,0,0], 255)

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
videofile = "webcam.mp4"
# read the image
cap = cv2.VideoCapture(0)

#lashes1_up
uwidth = 500
uheight = 130
uYmargin_le = 480
uXmargin_le = 230
uYmargin_ri = 340
uXmargin_ri = -100

# width = 560
# height = 130
# Ymargin_le = 290
# Xmargin_le = 230
# Ymargin_ri = 360
# Xmargin_ri = -100

#lashes1_down
dwidth = 600
dheight = 60
dYmargin_le = -30
dXmargin_le = 80
dYmargin_ri = 10
dXmargin_ri = -210

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    overlay = cv2.imread('red-coat.png')
    overlay= cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        points = []
        for n in range(48, 69):
            try:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                # print(x, y)
                points.append((x, y))
            # # Draw a circle
            # # cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
#                 cv2.line(frame, points[-1], points[-2], (0, 0, 255), 1)
            except:
                pass
        points.append(points[0])
        points = np.reshape(points, (-1, 1, 2))
#         cv2.fillPoly(frame, [points], (0, 0, 255), 8)
#         try:
#             cv2.line(frame, points[0], points[-1], (0, 0, 255), 1)
#         except:
#             pass
        eyes_points_right = []
        for n in range(36, 42):
            try:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
#                 print(x, y)
                eyes_points_right.append((x, y))
                # # Draw a circle
                # # cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
                cv2.line(frame, eyes_points_right[-1], eyes_points_right[-2], (0, 0, 255), 1)
            except:
                pass
        try:
            cv2.line(frame, eyes_points_right[0], eyes_points_right[-1], (0, 0, 255), 1)
        except:
            pass
        eyes_points_left = []
        for n in range(42, 48):
            try:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
#                 print(x, y)
                eyes_points_left.append((x, y))
                # # Draw a circle
                # # cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
                cv2.line(frame, eyes_points_left[-1], eyes_points_left[-2], (0, 0, 255), 1)
            except:
                pass
        try:
            cv2.line(frame, eyes_points_left[0], eyes_points_left[-1], (0, 0, 255), 1)
        except:
            pass   
    
    try:
        cv2.imwrite('frame_temp.png', frame)
        background = cv2.filter2D(frame, -1, sharpen_filter)
        added_image = cv2.addWeighted(background,0.5,overlay,0.6,0)  
        cv2.imwrite('frame_temp1.png', added_image)
        im2 = Image.open('frame_temp.png')
        im1 = Image.open('frame_temp1.png')
        empty_mask = np.zeros((im2.size[1], im2.size[0]), dtype=np.uint8)
        cv2.fillPoly(empty_mask, [points], (255,255,255), 8)
        cv2.imwrite('frame_temp2.png', empty_mask)
        mask = Image.open('frame_temp2.png').convert('L').resize(im1.size)
        mask_blur = mask.filter(ImageFilter.GaussianBlur(3))
        im_final = Image.composite(im1, im2, mask_blur)


        rot1, eye1_w, eye1_h = get_eyelashes_coord(eyes_points_right, uwidth, uheight, 'left')   
        rot2, eye2_w, eye2_h = get_eyelashes_coord(eyes_points_left, uwidth, uheight, 'right')    

        img2_o = Image.open('trans_lashes_up1.png')
        img2 = img2_o.resize((int(im2.size[0]/eye1_w),int(im2.size[1]/eye1_h)))
        img2 = img2.rotate(-4-rot1, expand=True)

        img3 = img2_o.transpose(Image.FLIP_LEFT_RIGHT)
        img3 = img3.resize((int(img3.size[0]/eye2_w),int(img3.size[1]/eye2_h)))
        img3 = img3.rotate(6-abs(rot2), expand=True)
        # Pasting img2 image on top of img1 
        # starting at coordinates (0, 0)
        im_final.paste(img2, (eyes_points_right[0][0]-int(uXmargin_le/eye1_w), eyes_points_right[0][1]-int(uYmargin_le/eye1_h)), mask = img2)
        im_final.paste(img3, (eyes_points_left[0][0]-int(uXmargin_ri/eye2_w), eyes_points_left[0][1]-int(uYmargin_ri/eye2_h)), mask = img3)

        im_final.save('frame_temp3.png')
        final_frame = cv2.imread('frame_temp3.png')         
        cv2.imshow(winname="Face", mat=final_frame)   
    except:
        pass

    
#     cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()