import cv2
import numpy as np
import random as rng

def filter_contour(_contours):
    _contours_filtered = []
    for i, c in enumerate(_contours):
        try:
            convex_hull = cv2.convexHull(c)
            area_hull = cv2.contourArea(convex_hull)
            if 600 < area_hull<2500:  # filtering based on area
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
        pret, pframe = pcap.read()
        if pret: 
		# Crop the image to required resolution
                pframe = pframe[0:480, 0:480]
                output = pframe.copy()
                gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3,3), 0)
		
		''' 
  		    Image binarisation/thresholding divides the eye image into two parts,
		    black(pupil) and white(rest of the eye).
      		    _, thresholded = cv2.threshold(X, Y, Z, cv2.THRESH_BINARY_INV)
		    X is the output of previous operation, Y= thresholding pixel value which has to be 
  		    adjusted according the condition, Y is the maximum value of the pixel
       		'''
		
                _, thresholded = cv2.threshold(blurred, 124, 255, cv2.THRESH_BINARY_INV) #new method added
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours_filtered = filter_contour(contours)
                drawing = np.zeros((blurred.shape[0], blurred.shape[1], 3), dtype=np.uint8)
                drawing = draw_ellipse(blurred, contours_filtered)
                cv2.imshow('dframe', drawing  )
		
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
		
pcap.release()
cv2.destroyAllWindows()
