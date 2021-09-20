import cv2
import dlib
import numpy as np
# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
videofile = "webcam.mp4"
# read the image
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

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
                cv2.line(frame, points[-1], points[-2], (0, 0, 255), 1)
            except:
                pass
        points = np.reshape(points, (-1, 1, 2))
        cv2.fillPoly(frame, [points], (0, 0, 255), 8)
        try:
            cv2.line(frame, points[0], points[-1], (0, 0, 255), 1)
        except:
            pass
        eyes_points_right = []
        for n in range(36, 42):
            try:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                print(x, y)
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
                print(x, y)
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
    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()