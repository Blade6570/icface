#from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
#from imutils import face_utils
import numpy as np
from PIL import Image
#import pdb   
from imutils.face_utils import FaceAligner
#def pre_process(path):
for i in range(2):    
    
    predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')

    fa = FaceAligner(predictor, desiredFaceWidth=256)

  #  detector = dlib.get_frontal_face_detector()
    detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

    image = cv2.imread('/home/nvme/soumya/test_ICface2/new_crop/pd.jpg') # add your image here
    
    #image= cv2.resize(cap, (256, 256))

    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    c=1235
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    rects = detector(RGB, 1)
    faces = dlib.full_object_detections()

#    faceAligned = fa.align(RGB, gray, rects[0].rect)
 #   rects_a = detector(faceAligned, 1)

    for rect in rects:
                    c1=rect.rect.dcenter()
                    #(x, y, w, h) = rect_to_bb(rect)
                    x=rect.rect.left()
                    y=rect.rect.top()
                    w=rect.rect.right()-x
                    h=rect.rect.bottom()-y
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


