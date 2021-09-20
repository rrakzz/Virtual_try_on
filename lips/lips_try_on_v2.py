from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys, os
import binascii
import struct
from PIL import Image, ImageDraw
import scipy
import scipy.misc
import scipy.cluster
from sys import argv
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from statistics import mean
import time
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import image_similarity_measures
from image_similarity_measures.quality_metrics import ssim

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

def apply_lipstick(img, rgb):
    if(len(a)):
        name, (i, j)=a[0]
        points= []
        
        for (x, y) in shape[i:j]:
            points.append([x,y])
        points= np.reshape(points, (-1, 1, 2))
        cv2.fillPoly(img, [points], rgb, 8)
    
    return img, points


def get_dominant_colour(img_path):
    try:
        NUM_CLUSTERS = 2
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


# define our function for preparing mask
def prepare_mask(polygon, image):
    """Returns binary mask based on input polygon presented as list of coordinates of vertices
    Params:
        polygon (list) - coordinates of polygon's vertices. Ex: [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        image (numpy array) - original image. Will be used to create mask of the same size. Shape (H, W, C).
    Output:
        mask (numpy array) - boolean mask. Shape (H, W).
    """
    # create an "empty" pre-mask with the same size as original image
    width = image.shape[1]
    height = image.shape[0]
    mask = Image.new('L', (width, height), 0)
    # Draw your mask based on polygon
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    # Covert to np array
    mask = np.array(mask).astype(bool)
    mask_blur = cv2.GaussianBlur(np.float32(mask),(99,99),0)
    
    return mask, mask_blur

def compute_histogram(mask, image):
    """Returns histogram for image region defined by mask for each channel
    Params:
        image (numpy array) - original image. Shape (H, W, C).
        mask (numpy array) - boolean mask. Shape (H, W).
    Output:
        list of tuples, each tuple (each channel) contains 2 arrays: first - computed histogram, the second - bins.

    """
    # Apply binary mask to your array, you will get array with shape (N, C)
    region = image[mask]

    red = np.histogram(region[..., 0].ravel(), bins=256, range=[0, 256])
    green = np.histogram(region[..., 1].ravel(), bins=256, range=[0, 256])
    blue = np.histogram(region[..., 2].ravel(), bins=256, range=[0, 256])

    return [red, green, blue]


def plot_histogram(histograms):
    """Plots histogram computed for each channel.
    Params:
        histogram (list of tuples) - [(red_ch_hist, bins), (green_ch_hist, bins), (green_ch_hist, bins)]
    """

    colors = ['r', 'g', 'b']
    for hist, ch in zip(histograms, colors):
        plt.bar(hist[1][:256], hist[0], color=ch)


Model_PATH =Model_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(Model_PATH)

num = 3
for nu in range(1,num):
    imgPath = f"D:\\Face\lips\\images\\{nu}.jpg"
    imgOutputPath = f"D:\\Face\\lips\\lakme_out\\{nu}_out.jpg"
    imgOutputPath2 =f"D:\\Face\\lips\\lakme_out\\{nu}_out_tryOn"
    paletteImages = "ruby-rush.png,red-coat.png,red-rust.png" #red-coat.png" #ruby-rush.png red-rust.png, 
    imgOutputPath3 = f"D:\\Face\\lips\\lakme_out\\{nu}_out_mask.jpg"
    temp_image = r"D:\\Face\\lips\\images\\temp.jpg"

    img = cv2.imread(imgPath)
    overlay = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # loop over the face detections
    a=[]
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        a=list(face_utils.FACIAL_LANDMARKS_IDXS.items())

    paletteImageList = paletteImages.split(',')
    index =1
    print(paletteImageList)
    height = overlay.shape[0]
    width = overlay.shape[1]

    empty_mask = np.zeros((height, width), dtype=np.uint8)

    for paletteImage in paletteImageList:
        print(paletteImage)
        bgr_p = get_dominant_colour(paletteImage)
        #print('palette' ,bgr)
        applied_lips, lips_polygon = apply_lipstick(img,bgr_p)
#         cv2.imwrite(imgOutputPath,applied_lips)

        cv2.fillPoly(empty_mask, lips_polygon, (255))
        res = cv2.bitwise_and(overlay,overlay,mask = empty_mask)

        background = cv2.imread(imgPath)
        overlay = cv2.imread(paletteImage)
        overlay= cv2.resize(overlay, (background.shape[1], background.shape[0]))
        
        rect = cv2.boundingRect(lips_polygon) # returns (x,y,w,h) of the rect
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
            rect = cv2.boundingRect(lips_polygon) # returns (x,y,w,h) of the rect
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
        cv2.fillConvexPoly(empty_mask, lips_polygon, (255,255,255))
        cv2.imwrite(imgOutputPath3, empty_mask)


        mask = Image.open(imgOutputPath3).convert('L').resize(im1.size)
        mask_blur = mask.filter(ImageFilter.GaussianBlur(3))
        im_final = Image.composite(im1, im2, mask_blur)
        im_final.save(f"{imgOutputPath2}_{color_control1[color_Labspace1.index(min(color_Labspace1))]}_{paletteImage}")
        
        index = index +1
    os.remove(imgOutputPath3) 
    os.remove('temp1.jpg')
    os.remove('temp2.jpg')