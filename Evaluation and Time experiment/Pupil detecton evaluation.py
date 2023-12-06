import os
import cv2
import numpy as np
import random as rng
import os
import pandas as pd

'''
Here accuracy of the model is calculated using the LPW dataset with resolution 640x480.
Accuracy is calculated for different pixel errors.
'''

# Define the directory where your folders are located
base_directory = "Path to read the data\LPW"

#initialization
Px_1_error = 0
Px_5_error = 0
Px_10_error = 0
Px_15_error = 0
Px_20_error = 0
Px_25_error = 0

# Get a list of folder names within the base directory
folder_names = os.listdir(base_directory)
total_images = 0

# Iterate through each folder
for folder_name in folder_names:
    _, file_extension = os.path.splitext(folder_name)

    # Skip if the file doesn't have a recognized extension
    if file_extension in ['.ods','.txt','.ods#']:
        continue
    else:
        folder_path = os.path.join(base_directory, folder_name)

        file_names =  os.listdir(folder_path)

    avi_files = [filename for filename in file_names if filename.endswith('.avi')]
    txt_files = [filename for filename in file_names if filename.endswith('.txt')]

    for avi_file in avi_files:
        avi_name = os.path.splitext(avi_file)[0]  # Remove the .avi extension
        corresponding_txt = f"{avi_name}.txt"

        if corresponding_txt in txt_files:
            avi_path = os.path.join(folder_path, avi_file)
            txt_path = os.path.join(folder_path, corresponding_txt)

        #reading the ground truth file
        ground_truth = pd.read_csv(txt_path,sep =' ',header=None)
        ground_truth.columns = ['x_truth','y_truth']
        ground_truth['Sl_No'] = ground_truth.index

        frame_number= 0
        frame_dict = {}

        pred_cordinate =pd.DataFrame()

        def filter_contour(_contours):
            _contours_filtered = []
            for i, c in enumerate(_contours):
                try:
                    convex_hull = cv2.convexHull(c)
                    area_hull = cv2.contourArea(convex_hull)
                    if 600 < area_hull:  # filtering based on area
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
                #cv2.drawContours(_drawing, _contours_filtered, i, color)
                (x, y), (MA, ma), angle = minEllipse[i]
                #area_contour_hull = cv2.contourArea(c)
                area_ellipse[i] = (np.pi / 4) * MA * ma
                try:
                    small_area = area_ellipse.index(min(area_ellipse))
                    (x_pred, y_pred), (MA, ma), angle = minEllipse[small_area]
                    cv2.ellipse(_drawing, minEllipse[small_area], color=color, thickness=2)
                    frame_dict[frame_number] = {
                        'Sl_No': frame_number,
                        'x_pred': x_pred,
                        'y_pred': y_pred,
                        'Area' : min(area_ellipse)
                    }
                except TypeError:
                    pass
            return _drawing

        video_path=avi_path
        cap = cv2.VideoCapture(video_path)


        while True:
            ret, frame = cap.read()

            if not ret:  # If no frame is read, break the loop
                break
            total_images +=1
            copy = frame[0:480,0:480]            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3,3), 0)
            _, thresholded = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_filtered = filter_contour(contours)
            drawing = np.zeros((blurred.shape[0], blurred.shape[1], 3), dtype=np.uint8)
            drawing = draw_ellipse(frame, contours_filtered)

            cv2.imshow(f'{avi_name}', drawing)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        df_predicted = pd.DataFrame.from_dict(frame_dict, orient='index')

        # merge two dataFrames
        merged_df = pd.merge(df_predicted, ground_truth, on='Sl_No')

        # finding the absolute error between the ground truth and predicted pupil centers
        merged_df['x_error'] = (merged_df['x_truth']-merged_df['x_pred']).abs()
        merged_df['y_error'] = (merged_df['y_truth']-merged_df['y_pred']).abs() 

        #counting the total number of predictions in each category of errors
        count_1px = ((merged_df['x_error']<1) & (merged_df['y_error']<1)).sum()
        count_5px = ((merged_df['x_error']<5) & (merged_df['y_error']<5)).sum()
        count_10px = ((merged_df['x_error']<10) & (merged_df['y_error']<10)).sum()
        count_15px = ((merged_df['x_error']<15) & (merged_df['y_error']<15)).sum()
        count_20px = ((merged_df['x_error']<20) & (merged_df['y_error']<20)).sum()
        count_25px = ((merged_df['x_error']<25) & (merged_df['y_error']<25)).sum()

        Px_1_error = Px_1_error + count_1px
        Px_5_error = Px_5_error + count_5px
        Px_10_error = Px_10_error + count_10px
        Px_15_error = Px_15_error + count_15px
        Px_20_error = Px_20_error + count_20px
        Px_25_error = Px_25_error + count_25px

# Print the total error for different pixel sizes
print('1 pixel error' , Px_1_error/total_images,
            '5 pixel error' , Px_5_error/total_images,
            '10 pixel error' , Px_10_error/total_images,
            '15 pixel error' , Px_15_error/total_images,
            '20 pixel error' , Px_20_error/total_images,
            '25 pixel error' , Px_25_error/total_images,)

# storing the accuracy values for different pixel values 
Accuracy =[ {'Error':'1 pixel error','Accuracy' : Px_1_error/total_images},
            {'Error':'5 pixel error' ,'Accuracy': Px_5_error/total_images},
            {'Error':'10 pixel error','Accuracy' : Px_10_error/total_images},
            {'Error':'15 pixel error' ,'Accuracy': Px_15_error/total_images},
            {'Error':'20 pixel error','Accuracy' : Px_20_error/total_images},
            {'Error':'25 pixel error' ,'Accuracy': Px_25_error/total_images}]

df = pd.DataFrame(Accuracy)

#Storing the values the in an excel format in the local directory
df.to_excel('Path to save excel file\Accuracy.xlsx', index=False)
