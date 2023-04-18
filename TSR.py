import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math
from keras.models import model_from_json
from PIL import Image
import cv2
import numpy as np
import xlrd
from scipy import ndimage
import tensorflow as tf
import csv
import pyttsx3

  
filename = "./signnames.csv"
  

fields = [] 
rows = [] 


# reading csv file 
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = next(csvreader) 
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 

import pandas as pd
#reading csv file
df = pd.read_csv('C:/Users/HASWIKA/Desktop/TrafficSign/signnames.csv')


SIGNS=list(df.SignName)
SIZE =32

#function to contrast limit -preprocessing
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

#smoothing of noise
def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image

#binarisation of image    
def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    #print(nb_components, output, stats, centroids)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE    )
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts

def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    t=0
    l=0
    r=0
    b=0
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            sign,t,l,r,b = cropSign(image,coordinate)
    return sign, coordinate,t,l,r,b
        

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2
    
def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    
    return image[top:bottom,left:right],top,left,right,bottom

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)


def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))

    #binary_image = remove_line(binary_image)

    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)
    #signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    sign, coordinate,top,left,right,bottom = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    
    return coordinate, original_image,sign,top,left,right,bottom


    
def clean_images():
	file_list = os.listdir('./')
	for file_name in file_list:
		if '.png' in file_name:
			os.remove(file_name)

def main(args,file):
	#Clean previous image    
    clean_images()
    #Training phase
    json_file = open('model1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
    model.load_weights("model_weights1.h5")
    print("Loaded model from disk")
    model.summary()

    vidcap = cv2.VideoCapture(file)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.mp4',fourcc,fps, (640,480))
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None
    success = True
    #similitary_contour_with_circle = 0.75   # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w")
    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        frame = cv2.resize(frame, (640,480))

        print("Frame:{}".format(count))
        cv2.imwrite("frame%d.jpg" % count, frame)
        #image = cv2.cvtColor(image, cv2
        coordinate, image,sign,top,left,right,bottom = localization(frame, args.min_size_components, args.similitary_contour_with_circle, model, count, current_sign)
        
        im=Image.open("frame%d.jpg"%count)
        x = im.crop((left,top,right,bottom))
    
        if coordinate is not None:
            cv2.rectangle(image,coordinate[0],coordinate[1],(0, 255, 0), 1)
        im1= x.convert('RGB')
        im1 = im1.resize((32,32), Image.ANTIALIAS)
        im1 = np.array(im1)
        x1 = np.expand_dims(im1, axis=0)
        
        #print(x1.shape)
        x1=x1.reshape(1,32,32,3)
        out1 = model.predict(x1)
        y = np.argmax(out1)
        dir_test = os.listdir('E:/traffic-sign-recognition/gtsrb-german-traffic-sign/Train')
        dir_test = sorted(dir_test)
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        fontcolor = (0, 0, 255)
        for i in range(0,43):
            b=rows[i][0]
            if(int(b)==y): 
                ans=rows[i][1]
                engine=pyttsx3.init()
                engine.runAndWait()
                if coordinate is not None:
                    engine.say(ans)
                    cv2.putText(image,rows[i][1], (coordinate[0][0], coordinate[0][1] -15), fontface, fontscale, fontcolor)
                    
                    
        
        cv2.imshow('Result', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(image)
        count=count+1
      
def test(x):
    parser = argparse.ArgumentParser(description="NLP Assignment Command Line")
    
    parser.add_argument(
      '--file_name',
      default= x,
      help= "Video to be analyzed"
      )
    
    parser.add_argument(
      '--min_size_components',
      type = int,
      default= 300,
      help= "Min size component to be reserved"
      )

    
    parser.add_argument(
      '--similitary_contour_with_circle',
      type = float,
      default= 0.70,
      help= "Similitary to a circle"
      )
    
    args = parser.parse_args()
    main(args,x)
