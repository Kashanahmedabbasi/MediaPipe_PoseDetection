import cv2
import time
import PoseDetection as pd

cap = cv2.VideoCapture(0)
lstlm=[]
ctime=0
ptime=0
detector = pd.pose_detector()
while True:
    status , img = cap.read()
    img = detector.findpose(img,draw=False)
    lstlm=detector.findlocation(img)
    if len(lstlm)!=0:
        print(lstlm[0])
        # cv2.circle(img,(lstlm[0][1],lstlm[0][2]),15,(255,0,255),cv2.FILLED)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break