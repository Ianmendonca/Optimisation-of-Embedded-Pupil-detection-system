import cv2
import numpy as np
import random as rng
import time
import pandas as pd
from datetime import datetime

'''Change the parameters to conduct a real time detection'''
CANNY_THRESHOLD = 25

columns = ('Thresholding','Canny Detection',)
df_experiment =pd.DataFrame(columns = columns)

Canny = []
thresholding = []


def filter_contour(_contours):
    _contours_filtered = []
    for i, c in enumerate(_contours):
        try:
            convex_hull = cv2.convexHull(c)
            area_hull = cv2.contourArea(convex_hull)
            if 600 < area_hull <2500:  # filtering based on area
                circumference_hull = cv2.arcLength(convex_hull, True)
                circularity_hull = (4 * np.pi * area_hull) / circumference_hull ** 2
                if 0.8 < circularity_hull:  # filtering based on circularity
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

srcPiCam = 'libcamerasrc ! video/x-raw,width=640,height=480 ! videoflip method=clockwise ! videoconvert ! appsink drop=True'
pcap = cv2.VideoCapture(srcPiCam)
if pcap.isOpened():
        print(f'Puil camera available:')
        
while True:
        t1 = time.process_time()#pre process time start
        pret, pframe = pcap.read()
        
        if pret:                
                pframe = pframe[0:480, 0:480]
                output = pframe.copy()
                gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3,3), 0)
                
                threshold_time_start = time.process_time()
                _, thresholded = cv2.threshold(blurred, 113, 255, cv2.THRESH_BINARY_INV)
                threshold_time_end = time.process_time()
                thresholding.append(threshold_time_end - threshold_time_start)
                
                canny_start_time = time.process_time()
                canny = cv2.Canny(blurred, CANNY_THRESHOLD, CANNY_THRESHOLD * 2)
                canny_end_time = time.process_time()
                Canny.append(canny_end_time-canny_start_time)
                
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_filtered = filter_contour(contours)
                drawing = np.zeros((blurred.shape[0], blurred.shape[1], 3), dtype=np.uint8)
                drawing = draw_ellipse(blurred, contours_filtered)
                
                t2 = time.process_time()
                    
                cv2.imshow('dframe', drawing  )
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

pcap.release()
cv2.destroyAllWindows()

df_experiment['Canny Detection'] = Canny
df_experiment['Thresholding'] = thresholding

time_right_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
file_name = f'//home//ubicomp//Desktop//eye-tracking//Raspberry-Pi//Time Experiment//ThresholdingVsCannyEdgeDetection//ThresholdingVsCanny_640_480_{time_right_now}.csv'
df_experiment.to_csv(file_name, index=False)
