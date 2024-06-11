
import cv2
import mediapipe as mp
import time 

class pose_detector():
    def __init__(self,mode=True,modelcomplexity=2,smoothlandmarks=True,enablesegmentation=False,
                 smoothsegmentation=True,mindetectionconfidence=0.5,mintrackingconfidence=0.5):
        self.mode=mode
        self.modelcomplexity=modelcomplexity
        self.smoothlandmarks=smoothlandmarks
        self.enablesegmentation=enablesegmentation
        self.smoothsegmentation=smoothsegmentation
        self.mindetectionconfidence=mindetectionconfidence
        self.mintrackingconfidence=mintrackingconfidence
        
        self.pose = mp.solutions.pose
        self.mppose = self.pose.Pose(self.mode,self.modelcomplexity,self.smoothlandmarks,self.enablesegmentation
                                     ,self.smoothsegmentation,self.mindetectionconfidence,self.mintrackingconfidence)
        self.mp_draw = mp.solutions.drawing_utils

    def findpose(self,img,draw=True):
        self.results = self.mppose.process(img)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img,self.results.pose_landmarks,self.pose.POSE_CONNECTIONS)
            
        return img
    
    def findlocation(self,img,draw=True):
        lstlm=[]
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(img,self.results.pose_landmarks,self.pose.POSE_CONNECTIONS)
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lstlm.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lstlm
        
    
    
   
    
def main():
    cap = cv2.VideoCapture(0)
    lstlm=[]
    ctime=0
    ptime=0
    detector = pose_detector()
    while True:
        status , img = cap.read()
        # img = cv2.resize(img,(416,416))
        img = detector.findpose(img)
        lstlm=detector.findlocation(img)
        if len(lstlm)!=0:
            print(lstlm[10])
            cv2.circle(img,(lstlm[9][1],lstlm[9][2]),15,(255,0,255),cv2.FILLED)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
    
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
if __name__=="__main__":
    main()