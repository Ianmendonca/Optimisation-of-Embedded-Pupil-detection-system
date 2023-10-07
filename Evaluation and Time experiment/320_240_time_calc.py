import cv2
import numpy as np
import random as rng
import time
import pandas as pd
from datetime import datetime

''' Create a dataframe to store the time details for all the processess'''

columns = ('Total Time','Pre Process Time','Post Process','Blurring','Thresholding','Find Contours','Filter Contour')
df_experiment =pd.DataFrame(columns = columns)

total_time = []
pre_process_time = []
post_process = []
blurring = []
thresholding = []
find_contours = []
fil_contour = []

''' filter the contours based on the area and circularity'''

def filter_contour(_contours):
    _contours_filtered = []
    for i, c in enumerate(_contours):
        try:
            convex_hull = cv2.convexHull(c)
            area_hull = cv2.contourArea(convex_hull)
            # print("{} area convex hull {}".format(i, area_hull))
            if 100< area_hull<400:  # filtering based on area std: 100,400
                circumference_hull = cv2.arcLength(convex_hull, True)
                circularity_hull = (4 * np.pi * area_hull) / circumference_hull ** 2
                if 0.8 < circularity_hull:  # filtering based on circularity std : 0.8
                    _contours_filtered.append(convex_hull)
        except ZeroDivisionError:
            print("Division by zero for contour {}".format(i))
    return _contours_filtered


def draw_ellipse(_drawing, _contours_filtered):
    minEllipse = [None] * len(_contours_filtered)
    area_ellipse = [None] * len(_contours_filtered)
    for i, c in enumerate(_contours_filtered):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        minEllipse[i] = cv2.fitEllipse(c)
        (x, y), (MA, ma), angle = minEllipse[i]
        area_ellipse[i] = (np.pi / 4) * MA * ma
        try:
            small_area = area_ellipse.index(min(area_ellipse))
            (x_pred, y_pred), (MA, ma), angle = minEllipse[small_area]
            cv2.ellipse(_drawing, minEllipse[small_area], color=color, thickness=2)
        except TypeError:
            pass
    return _drawing

srcPiCam = 'libcamerasrc ! video/x-raw,width=320,height=240 ! videoflip method=clockwise ! videoconvert ! appsink drop=True'
pcap = cv2.VideoCapture(srcPiCam)
if pcap.isOpened():
        print(f'Pupil camera available:')

while True:
        t1 = time.process_time()
        pret, pframe = pcap.read()
        if pret:        
                pframe = pframe[0:150, 0:240]
                output = pframe.copy()
                gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                
                blurr_time_start = time.process_time()
                blurred = cv2.GaussianBlur(gray, (3,3), 0)
                blurr_time_end = time.process_time()
                blurring.append(blurr_time_end - blurr_time_start)
                
                threshold_time_start = time.process_time()
                _, thresholded = cv2.threshold(blurred, 124, 255, cv2.THRESH_BINARY_INV)
                threshold_time_end = time.process_time()
                thresholding.append(threshold_time_end - threshold_time_start)
                
                pre_process_time_end = time.process_time()
                pre_process_time.append(pre_process_time_end - t1)
                
                findContours_time_start = time.process_time()
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                findContours_time_end = time.process_time()
                find_contours.append(findContours_time_end - findContours_time_start)
                
                filtercontours_time_start = time.process_time()
                contours_filtered = filter_contour(contours)
                filtercontours_time_end = time.process_time()
                fil_contour.append(filtercontours_time_end-filtercontours_time_start)
                
                post_processing_time_start = time.process_time()                
                drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
                drawing = draw_ellipse(blurred, contours_filtered)
                t2 = time.process_time()
                
                post_process.append(t2-post_processing_time_start)
                total_time.append(t2-t1)
                    
                cv2.imshow('dframe', drawing  )
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
pcap.release()
cv2.destroyAllWindows()

df_experiment['Total Time'] = total_time
df_experiment['Pre Process Time'] = pre_process_time 
df_experiment['Post Process'] = post_process
df_experiment['Blurring'] = blurring
df_experiment['Thresholding'] = thresholding
df_experiment['Find Contours'] = find_contours
df_experiment['Filter Contour'] = fil_contour


time_right_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
file_name = f'//home//ubicomp//Desktop//eye-tracking//Raspberry-Pi//Time Experiment//320x240 resolution//320_240_total_time_{time_right_now}.xlsx'
df_experiment.to_excel(file_name, index=False)


