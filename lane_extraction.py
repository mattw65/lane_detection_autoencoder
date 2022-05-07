from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image

def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(0, height), (width//2, 100), (width, height)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def average(image, lines):
    left = []
    right = []
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
            
def make_points(image, average): 
    slope, y_int = average 
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


i=1
basewidth = 300
img = Image.open("/Users/MattWalsh/Desktop/Files/School/CS6434/Project/frames/frame%d.jpg" % i)
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
img = np.array(img)
plt.imshow(img)
plt.show()

def extract_lines(img):
    # filter out white and yellow for lane lines
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 170, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)

    gr = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
    yw = cv2.bitwise_or(mask_yw, mask_yellow)
    plt.imshow(yw)
    plt.show()

    gaus = cv2.GaussianBlur(yw,(1,1), 0)

    edges = cv2.Canny(gaus,50,150)

    isolated = region(edges)
    plt.imshow(isolated)
    plt.show()

    lines = cv2.HoughLinesP(isolated,rho = 2,theta = 1*np.pi/180,threshold = 10,minLineLength = 10,maxLineGap = 10)

    return lines

def add_lines(img):
    lines = extract_lines(img)
    for l in lines:
        l = l[0]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)
    return img

def get_MSE(img1, img2):
    im1Lines = add_lines(img1)
    im2Lines = add_lines(img2)

    err = np.sum((im1Lines.astype("float") - im2Lines.astype("float")) ** 2)
    err /= float(im1Lines.shape[0] * im1Lines.shape[1])

    return err