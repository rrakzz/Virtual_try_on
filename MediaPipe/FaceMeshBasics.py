import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import image_similarity_measures
from image_similarity_measures.quality_metrics import ssim
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import sys, os
import scipy
import scipy.misc
import scipy.cluster
import binascii
import struct

#cap = cv2.VideoCapture("D:\\Face\\MediaPipe\\output_25fps.mp4") #"Videos/1.mp4"
pTime = 0

nu = 1
imgPath = f"D:\\Face\lips\\images\\{nu}.jpg"
paletteImages = "red-coat.png,ruby-rush.png,red-rust.png"
temp_image = r"temp.jpg"

def get_dominant_colour(img_path):
    try:
        NUM_CLUSTERS = 5
        im = Image.open(img_path)
        #im = im.resize((150, 150))      # optional, to reduce time
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences    
        index_max = scipy.argmax(counts)                    # find most frequent
        peak = codes[index_max]
        colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
        r = int(peak[0])
        g = int(peak[1])
        b = int(peak[2])
        bgr = (b, g, r)
        return bgr
    except:
        pass
    return (-1,-1,-1)

def compare_color(bgr1, bgr2):
    # Red Color
    color1_rgb = sRGBColor(bgr1[2],bgr1[1], bgr1[0]);

    # Blue Color
    color2_rgb = sRGBColor(bgr2[2],bgr2[1], bgr2[0]);

    # Convert from RGB to Lab Color Space
    color1_lab = convert_color(color1_rgb, LabColor);

    # Convert from RGB to Lab Color Space
    color2_lab = convert_color(color2_rgb, LabColor);

    # Find the color difference
    delta_e = delta_e_cie2000(color1_lab, color2_lab);

#     print ("The difference between the 2 color = ", delta_e)
    return round(delta_e, 2)
    
def compare_SSIM(img1,img2):
#     print(img1,img2)
    in_img1 = cv2.imread(img1)
    in_img2 = cv2.imread(img2)

    img1_outputimage = 'temp1.jpg'
    img2_outputimage = 'temp2.jpg'

    temp1_image = cv2.resize(in_img1, (100, 30), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(img1_outputimage, temp1_image)

    temp2_image = cv2.resize(in_img2, (100, 30), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(img2_outputimage, temp2_image)

    out_ssim = ssim(temp1_image, temp2_image) # Structural Similar Index Measure (SSIM)
#     print('SSIM : ', out_ssim)
    return round(out_ssim,2)
    
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

lips = [61, 40, 37, 0, 267, 270, 291,
        375, 405, 17, 84, 91, 61,
        78, 80, 82, 13, 312, 310, 308,
        324, 402, 14, 87, 88, 78]

lips_up = [61, 40, 37, 0, 267, 270, 291]

l_eye = [33, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163]
        
#while True:
    #success, img = cap.read()
img = cv2.imread(imgPath)
ih, iw, ic = img.shape
overlay = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
paletteImageList = paletteImages.split(',')
index =1
print(paletteImageList)
height = overlay.shape[0]
width = overlay.shape[1]
empty_mask = np.zeros((height, width), dtype=np.uint8)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = faceMesh.process(imgRGB)
if results.multi_face_landmarks:
#print(results.multi_face_landmarks)
    for faceLms in results.multi_face_landmarks:
        mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec,drawSpec)
        
       
        i, X,Y,Z = [], [], [], []
        points = []
        for id,lm in enumerate(faceLms.landmark):
            #print(lm)
            if id in lips:
                if id in lips_up:
                    X.append(int(lm.x*iw-lm.x*iw*0.1))
                    Y.append(int(lm.y*ih-lm.y*ih*0.1))                    
                else:
                    X.append(int(lm.x*iw))
                    Y.append(int(lm.y*ih))
                i.append(id)
                points.append((int(lm.x*iw), int(lm.y*ih)))
        
        points = []
        for id,lm in enumerate(lips):
            xx = faceLms.landmark[lm].x
            yy = faceLms.landmark[lm].y
            if id in lips_up:
                points.append((xx*iw-xx*iw*0.1, yy*ih-yy*ih*0.1))
            else:
                points.append((xx*iw, yy*ih))


points.append(points[0])    
points = np.reshape(points, (-1, 1, 2))  
points1 = np.int32([points])
cv2.fillPoly(img, points1, (0, 0, 255), 8)        

for paletteImage in paletteImageList:
    print(paletteImage)
    #imgOutPath = outputFolder + str(index) + '.jpg'
    bgr_p = get_dominant_colour(paletteImage)
    print(bgr_p)
    
    #cv2.imwrite('raw.jpg',img)

    cv2.fillPoly(empty_mask, points1, (255))
    res = cv2.bitwise_and(overlay,overlay,mask = empty_mask)

    background = cv2.imread(imgPath)
    overlay = cv2.imread(paletteImage)
    overlay= cv2.resize(overlay, (background.shape[1], background.shape[0]))
    
    rect = cv2.boundingRect(points.astype(np.int))# returns (x,y,w,h) of the rect
    cropped = background[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    cv2.imwrite('temp2.jpg', cropped)

    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    background = cv2.filter2D(background, -1, sharpen_filter)

    color_control = []
    color_SSIM = []
    color_Labspace = []
    for i in range(2,8,1):
        added_image = cv2.addWeighted(background,i/10,overlay,0.6,0)
        time.sleep(1)
        cv2.imwrite(temp_image, added_image)

        orig = cv2.imread(temp_image)
        rect = cv2.boundingRect(points.astype(np.int)) # returns (x,y,w,h) of the rect
        cropped = orig[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        cv2.imwrite('temp1.jpg', cropped)

        bgr = get_dominant_colour('temp1.jpg')

        d=((bgr[0]-bgr_p[0])**2+(bgr[1]-bgr_p[1])**2+(bgr[2]-bgr_p[2])**2)**0.5
        p=d*100/((255)^2+(255)^2+(255)^2)**0.5
        
        color_control.append(i/10)
        color_SSIM.append(round(compare_SSIM('temp2.jpg','temp1.jpg'),2))
        color_Labspace.append(round(compare_color(bgr_p, bgr),2))
        print(nu, paletteImage, i/10 ,bgr, round(d,2), round(p,2), compare_color(bgr_p, bgr),compare_SSIM('temp2.jpg','temp1.jpg'))

    color_control1 = []
    color_SSIM1 = []
    color_Labspace1 = []
    for i, val in enumerate(color_SSIM):
        if val > 0.88:
            color_control1.append(color_control[i])
            color_SSIM1.append(color_SSIM[i])
            color_Labspace1.append(color_Labspace[i])
            
        
    added_image = cv2.addWeighted(background,color_control1[color_Labspace1.index(min(color_Labspace1))],overlay,0.6,0)
    cv2.imwrite(temp_image, added_image)
    print('Best match: ', color_control1[color_Labspace1.index(min(color_Labspace1))], min(color_Labspace1))
    
    im2 = Image.open(imgPath)
    im1 = Image.open(temp_image)

    height = im2.size[1]
    width = im2.size[0]

    empty_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(empty_mask, points1, (255,255,255))
    cv2.imwrite('mask.jpg', empty_mask)


    mask = Image.open('mask.jpg').convert('L').resize(im1.size)
    mask_blur = mask.filter(ImageFilter.GaussianBlur(3))
    im_final = Image.composite(im1, im2, mask_blur)
    im_final.save(f"Out_{paletteImage}")
    
    index = index +1
    break
os.remove('temp1.jpg')
os.remove('temp2.jpg')


#cTime = time.time()
#fps = 1 / (cTime - pTime)
#pTime = cTime
#cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 0), 3)
#cv2.imshow("Image", img)
#cv2.waitKey(10000)