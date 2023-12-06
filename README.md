# Optimisation-of-Embedded-Pupil-detection-system
Here, an attempt is made to improve the processing time of the embedded system in detecting the pupil from a live feed data of the eye

This project inspired by the work done in **["An Embedded and Real-Time Pupil Detection Pipeline"](https://github.com/ankurrajw/Pi-Pupil-Detection.git) (https://arxiv.org/abs/2302.14098)** .

The setup used for the this project is same as used in the project mentioned above.

The main difference is in the pipeline used to detect the pupil, where conept of Binarization is used instead of Canny edge detection.More about this is mentioned in the report attached and these changes can be found in the detect_pupil.py file. Please refer the file **detect.py** in **Rasberry-Pi directory** in the above mentioned repository for comparing the old code with the new.

![image](https://github.com/Ianmendonca/Optimisation-of-Embedded-Pupil-detection-system/assets/97366497/b1443456-74d3-4a6c-929f-259f40a78b71)

Above you can see an example of binarised image using thresholding process in OpenCV

![image](https://github.com/Ianmendonca/Optimisation-of-Embedded-Pupil-detection-system/assets/97366497/c80bcd90-d32a-43db-bbbf-1f0aedfb4e29)
The above image shows a pipeline of detecting the pupil in this project.

The new pipeline is evaluated using a publicly available dataset**(https://doi.org/10.48550/arXiv.1511.05768)**
