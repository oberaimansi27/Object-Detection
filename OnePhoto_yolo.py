

 
import sys

from darkflow.net.build import TFNet 
#tFNET IS USED TO IMPORT DARKNET (NEURAL N/W)
import cv2#COMPUTER VISION 
import matplotlib.pyplot as plt

#%config.Inlinebackend.figure_format = 'svg' 
#Could not create cuDNN handle when convnets are 
import tensorflow as tf
#TensorFlow default mode is to initialize all available GPUs
config = tf.ConfigProto()#config the session
config.gpu_options.allow_growth = True
#GPU Growth
#the first is the allow_growth option, which attempts to allocate only as much GPU memory based on 
#runtime allocations: it starts out allocating very little memory, and as Sessions get run and more
# GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process
sess = tf.Session(config=config)#session creation


options = {
        
        'model':'C:/Users/LENOVO/Anaconda3/darkflow-master/cfg/yolo.cfg',
        'load':'C:/Users/LENOVO/Desktop/project/darkflow-master/darkflow-master/bin/yolov2.weights',
        #We load the configuration file and pre trained weights into variables
        'threshold':0.3,
        'gpu': 1.0
        }

tfnet = TFNet(options)  #WE INITIATE THE OBJECT OF TFNET CLASS network creation

img = cv2.imread('C:\\Users\\LENOVO\\Desktop\\project\\mansi\\catdog.jpg')
img.shape  #Shape of image is accessed by img.shape. It returns a tuple of number of rows, 
#columns and channels (if image is color)

result = tfnet.return_predict(img)

t1 = (result[0]['topleft']['x'],result[0]['topleft']['y']) #top lefta
b1 = (result[0]['bottomright']['x'],result[0]['bottomright']['y']) # bottomright

labl = result[0]['label']
img = cv2.rectangle(img, t1, b1,(0.,255,0),7)
img = cv2.putText(img,labl, t1,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)











