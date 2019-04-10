#from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
#from imutils import face_utils
import numpy as np
from PIL import Image
#import pdb   

#def pre_process(path):
for i in range(2):    
    detector = dlib.get_frontal_face_detector()

    cap = cv2.imread('/home/esa/Pictures/some.jpg') # add your image here
    
    image= cv2.resize(cap, (400, 400))

    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    c=1235
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    rects = detector(RGB, 1)
    faces = dlib.full_object_detections()

    for rect in rects:
                    c1=rect.dcenter()
                    (x, y, w, h) = rect_to_bb(rect)
                    w=np.int(w*1.6) 
                    h=np.int(h*1.6) 
                    x=c1.x-np.int(w/2.0)
                    y=c1.y-np.int(h/2.0)
                    if y<0:
                       y=0
                    if x<0:
                       x=0
                       
                    faceOrig = imutils.resize(RGB[y:y+h, x:x+w],height=256) #y=10,h+60,W+40
  
                    d_num = np.asarray(faceOrig)

                    f_im = Image.fromarray(d_num)

                    f_im.save('./new_crop/'+str(c)+'.png')


