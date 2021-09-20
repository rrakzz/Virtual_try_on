from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys
import binascii
import struct
from PIL import Image, ImageDraw
import scipy
import scipy.misc
import scipy.cluster
from sys import argv
import matplotlib.pyplot as plt


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
    return mask

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


Model_PATH =Model_PATH = "C:\\Vivekanandan\\ServiceTransformation\\Web\DigitalAssurance\\Content\\ImageComparison\\shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(Model_PATH)

#imgPath = r"D:\Solutions\VTO\3.jpg"
#outputFolder = r"D:\Solutions\VTO\Output"
#paletteImages = "D:\\Solutions\\ImageProcessing\\Comparison\\Test1\\LakmAbsoluteArganOilLipColor_BurntBrown.jpg,D:\\Solutions\\ImageProcessing\\Comparison\\Test1\\LakmAbsoluteArganOilLipColor_CrimsonSilk.jpg"
#outputFolder = outputFolder + '\\'

#imgPath = argv[1]
#outputFolder = argv[2]
#paletteImages = argv[3]
#outputFolder = outputFolder + '\\'

imgPath = r"D:\Solutions\VTO\2.jpg"
imgOutputPath = r"D:\Solutions\VTO\2_out.jpg"
imgOutputPath2 = r"D:\Solutions\VTO\2_out_transparency.jpg"
paletteImages = "D:\\Solutions\\ImageProcessing\\Comparison\\Test1\\LakmAbsoluteArganOilLipColor_BurntBrown.jpg,D:\\Solutions\\ImageProcessing\\Comparison\\Test1\\LakmAbsoluteArganOilLipColor_CrimsonSilk.jpg"
imgOutputPath3 = r"D:\Solutions\VTO\2_out_mask.jpg"

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

height = overlay.shape[0]
width = overlay.shape[1]

empty_mask = np.zeros((height, width), dtype=np.uint8)

for paletteImage in paletteImageList:
    #imgOutPath = outputFolder + str(index) + '.jpg'
    bgr = get_dominant_colour(paletteImage)
    print(bgr)
    applied_lips, lips_polygon = apply_lipstick(img,bgr)
    cv2.imwrite(imgOutputPath,applied_lips)

    cv2.fillPoly(empty_mask, lips_polygon, (255))
    res = cv2.bitwise_and(overlay,overlay,mask = empty_mask)

    rect = cv2.boundingRect(lips_polygon) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    #bg = np.ones_like(cropped, np.uint8)*255
    #cv2.bitwise_not(bg,bg, mask=empty_mask)
    #dst2 = bg+ res

    mask = prepare_mask(lips_polygon,overlay)
    cv2.imwrite(imgOutputPath3,cropped)

    histograms = compute_histogram(mask, overlay)

    # Let's plot our test results
    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.imshow(overlay)
    plt.title('Image')

    plt.subplot(222)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')


    plt.subplot(223)
    plot_histogram(histograms)
    plt.title('Histogram')

    plt.show()


    alpha = 0.4  # Transparency factor.
    # Following line overlays transparent rectangle over the image
    image_new = cv2.addWeighted(overlay, alpha, applied_lips, 1 - alpha, 0)
    cv2.imwrite(imgOutputPath2,image_new)
    index = index +1
    break