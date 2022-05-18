import numpy as np
import time
import PoseModule as pm
from expression import *
import cv2




cap = cv2.VideoCapture('E:/Pre-Thesis 2/ver 1.6/facial expression & Posture detection/videos/chest pain/4.mp4')
    
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0
flag=0
flag1=0
while True:
    success, img = cap.read()
    #img = cv2.resize(img, (852, 480))
    img = cv2.resize(img, (480, 852))
        #img = cv2.imread("G:/Project acanto/Post Estimation/videos/demo.jpg")
        #img = cv2.imread("E:/Pre-Thesis 2/ver 1.4/Post Estimation/newImg/stomach_ache/3.jpg")
        #img = cv2.imread("E:/Pre-Thesis 2/ver 1.4/Post Estimation/newImg/chest_fall/2.jpg")
        #img = cv2.imread("E:/Pre-Thesis 2/ver 1.4/Post Estimation/newImg/Bed_fall/2.jpg")
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    img.flags.writeable = False
    emote=findemote(img)
        # print(lmList)
    if len(lmList) != 0:
        handangleLeft = detector.findHandAngleLeft(img, 11, 13, 15)
        handangleRight = detector.findHandAngleRight(img, 12, 14, 16)
            
        HipAngleRight = detector.findHipAngleRight(img, 12, 24, 26)
        HipAngleLeft = detector.findHipAngleLeft(img, 11, 23, 25)
            
        HeadAngleRight = detector.findHeadAngleRight(img, 8, 12, 24)
        HeadAngleLeft = detector.findHeadAngleLeft(img, 7, 11, 23)
            
        LeftLeg = detector.findLeftLegAngle(img, 23, 25, 27)
        RightLeg = detector.findRightLegAngle(img, 24, 26, 28)
            
            
        CAngle = detector.findAngle(img, 23, 25, 27)
            
        Ha2Sh_DistanceLeft= detector.dis(img, 11, 15)
        Ha2Hi_DistanceLeft= detector.dis(img, 23, 15)
        
        He2ShR= detector.dis(img, 8, 12)
        He2ShL= detector.dis(img, 7, 11)
        
        fi2Ls= detector.dis(img, 21, 11)
        fi2Rs= detector.dis(img, 21, 12)
        
        per = np.interp(handangleLeft, (210, 310), (0, 100))
        bar = np.interp(handangleLeft, (220, 310), (650, 100))
            
        per = np.interp(handangleRight, (210, 310), (0, 100))
        bar = np.interp(handangleRight, (220, 310), (650, 100))
        
        per = np.interp(HipAngleRight, (210, 310), (0, 100))
        bar = np.interp(HipAngleRight, (220, 310), (650, 100))
            
        per = np.interp(HipAngleLeft, (210, 310), (0, 100))
        bar = np.interp(HipAngleLeft, (220, 310), (650, 100))
        
        per = np.interp(HeadAngleRight, (210, 310), (0, 100))
        bar = np.interp(HeadAngleRight, (220, 310), (650, 100))
            
        per = np.interp(HeadAngleLeft, (210, 310), (0, 100))
        bar = np.interp(HeadAngleLeft, (220, 310), (650, 100))
            
       
            
        per = np.interp(LeftLeg, (210, 310), (0, 100))
        bar = np.interp(LeftLeg, (220, 310), (650, 100))
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        LeftLeg = LeftLeg - 180
        value = He2ShR-He2ShL
        value = abs(value)
        print(LeftLeg)
#        print(HipAngleLeft)
           
    #video ready certified   
        if 5<handangleLeft<100 and 40<HipAngleLeft<220 and Ha2Sh_DistanceLeft<Ha2Hi_DistanceLeft:
            cv2.putText(img, str('Chest Pain'), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
        if 5<handangleLeft<300 and 40<HipAngleLeft<210:
            if 120<HeadAngleLeft<=170 and 75<LeftLeg<115 and 85<HipAngleLeft<120:
                cv2.putText(img, str('sitting'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif 110<HeadAngleLeft<=140 and 80<LeftLeg<115 and 70<HipAngleLeft<105:
                cv2.putText(img, str('Head Leaning'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif 115<HeadAngleLeft<180 and 80<LeftLeg<140 and 55<HipAngleLeft<100:
                cv2.putText(img, str('Shoulder Bending'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif 130<HeadAngleLeft<=200 and 110<LeftLeg<150 and 85<HipAngleLeft<110:
                cv2.putText(img, str('Fall'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
                                    
                    
    #video ready certified                
        elif 20<handangleRight<60 and 310<handangleLeft<340:
            cv2.putText(img, str('Headache'), (150, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255),3)
        
    #video ready
        if 60<handangleLeft<110 and 170<handangleRight<320 and 110<HipAngleLeft<310 and 40<Ha2Hi_DistanceLeft<100:
            cv2.putText(img, str('stomach ache'), (150, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)                    
        if 60<handangleLeft<210 and 60<handangleRight<310 and 110<HipAngleLeft<310 and Ha2Sh_DistanceLeft>Ha2Hi_DistanceLeft:
            if 60<handangleLeft<180 and 170<handangleRight<300 and 180<HipAngleLeft<300 and value<20 and flag==0:
                cv2.putText(img, str('Sitting'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif 60<handangleLeft<180 and 240<handangleRight<300 and 140<HipAngleLeft<225 and flag1==0:
                flag=1
                cv2.putText(img, str('Side Bending'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif 60<handangleLeft<170 and 130<handangleRight<260 and HipAngleLeft<190 and emote == None:
                flag1=1
                cv2.putText(img, str('Fall'), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
        
        
                  
    #video ready certified
        if 160<HipAngleRight<190 and 160<RightLeg<200:
            cv2.putText(img, str('Sleeping'), (600, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
        elif 190<HipAngleRight<230 and 40<RightLeg<160:
            cv2.putText(img, str('Fall'), (600, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
    #                (255, 0, 0), 5)
    cv2.putText(img, str(emote), (300, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)
    cv2.imshow("Posture & Facial Expression Detection", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()