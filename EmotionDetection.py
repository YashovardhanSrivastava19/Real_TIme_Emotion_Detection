#Facial Emotion Detection. 
""" [Detect Emotions in real time:This program is a demonstration on how to use FER(Facial Emotion Recognition),with MTCNN(Multi-task
Cascaded Convolution Nueral Network) to detect emotions in real time.] """
"""More Information:

FER:https://github.com/justinshenk/fer
OPENCV:https://opencv.org
WIKI:https://en.m.wikipedia.org/wiki/Affective_computing#Facial_affect_detection
"""
import cv2  
from fer import FER

class EmotionDetection:
    CAMERA_PORT=0
    def __init__(self) -> None:
        pass
    def DetectEmotion(self,ifBoundingBox=True,boundingBoxColor:tuple=(0,120,0),boundingBoxThickness:int=2,)->None:
        """ [Call this function for detecting emotions(pass optional arguments if necessary)] """                
        camera_object=cv2.VideoCapture(EmotionDetection.CAMERA_PORT,cv2.CAP_DSHOW)  #Change CAMERA_PORT to 1 if not using the primary camera.  
        detector=FER(mtcnn=True)
        while(True):
            _,frame=camera_object.read()
            result=detector.detect_emotions(frame)
            key=cv2.waitKey(1) 
            if result!=[]:
                for person in result:
                    bounding_box=person["box"]  
                    if ifBoundingBox:
                        #rectangle requires opposite vertices to draw.
                        cv2.rectangle(frame,(bounding_box[0],bounding_box[1]),
                        (bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]),
                        boundingBoxColor,
                        boundingBoxThickness)          
                    print(detector.top_emotion(frame))  #One of many uses!            
            if key==ord('q'):  
                """ [Press q to break out of the loop] """
                break
            cv2.imshow("Emotion Detection",frame)
        camera_object.release()
        cv2.destroyAllWindows()

#Object Instantiation
detEmo=EmotionDetection()
detEmo.DetectEmotion()
