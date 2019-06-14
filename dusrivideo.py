
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

options = {
        
        'model':'C:/Users/LENOVO/Anaconda3/darkflow-master/cfg/yolo.cfg',
        'load':'C:/Users/LENOVO/Desktop/project/darkflow-master/darkflow-master/bin/yolov2.weights',
        'threshold':0.3,
        'gpu': 1.0
        }

tfnet = TFNet(options)  #print modlw arch
count=0
capture = cv2.VideoCapture("C:\\Users\\LENOVO\\Desktop\\project\\mansi\\video.mp4")
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
#it returns a tuple containing 5 (R,G,B) sets
print(colors)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out11 = cv2.VideoWriter('C:\\Users\\LENOVO\\Desktop\\project\\mansi\\output2.mp4',fourcc, 20.0, (1500,800))
#This time we create a VideoWriter object. We should specify the output file name (eg: output.avi). 
#Then we should specify the FourCC code (details in next paragraph). Then number of frames per 
#second (fps) and frame size should be passed. And last one is isColor flag. If it is True, 
#encoder expect color frame, otherwise it works with grayscale frame.
count=0
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        resize=cv2.resize(frame,(640,480))
        cv2.imshow('video',resize)
        out11.write(resize)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
print("finished")
out11.release()
cv2.destroyAllWindows()