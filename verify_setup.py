import cv2
import numpy as np
net = cv2.dnn.readNetFromTensorflow(
'models/ssd_mobilenet_v3_large_coco.pb',
'models/ssd_mobilenet_v3_large_coco.pbtxt'
)
print('OpenCV', cv2.__version__)
print('Model loaded OK')
