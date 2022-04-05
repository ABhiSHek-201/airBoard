import cv2
import numpy as np
import os
import HandTrackingModule as htm

class VideoCamera(object):
    def __init__(self, video):
        self.video = video
        self.video.set(3,1280)
        self.video.set(4,720)

        folderPath = "headerImg"
        myList = os.listdir(folderPath)

        self.overlayList = []
        self.brushThickness = 15
        self.eraserThickness = 2*self.brushThickness 
        
        self.mode_disp = "Mode: "
        self.mode = ""
        self.mode_coord = (0, 600)

        self.font_disp = "Size: "
        self.font_d = str(self.brushThickness)
        self.font_coord = (1150 ,600)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255,0,255)
        self.thickness = 2

        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            self.overlayList.append(image)

        
        self.header = self.overlayList[1]
        self.drawColor = (255,0,255)

        self.detector = htm.handDetector(detectionCon=0.85)
        self.xp, self.yp = 0, 0
        self.imgCanvas = np.zeros((720,1280,3), np.uint8)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):

        # 1. import the image
        success,img = self.video.read()
        img = cv2.flip(img,1)

        if not success:
            # print("Success print: ",success)
            # print("Camera Opened : ",self.video.isOpened())
            return b'0'

        # 2.find Landmarks
        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img,draw=False)

        if len(lmList)!=0:
            # print(lmList)
            # tip of index and middle finger
            x1,y1 = lmList[8][1:]
            x2,y2 = lmList[12][1:]

            # 3.check which fingers are up
            fingers = self.detector.fingersUp()
            # print(fingers)

            # 4.if selection mode - two fingers are up
            if not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                self.mode = "Size Down"
                if self.brushThickness > 5:
                    self.brushThickness -= 1
                    self.eraserThickness = 2*self.brushThickness 
                self.font_d = str(self.brushThickness)

            elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
                self.mode = "Size Up"
                if self.brushThickness < 60:
                    self.brushThickness += 1
                    self.eraserThickness = 2*self.brushThickness 
                self.font_d = str(self.brushThickness)

            elif fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                self.mode = "Select Mode"
                # print("Select Mode")
                # Checking the click
                if y1<125:
                    if 320<x1<420:
                        self.header = self.overlayList[1]
                        self.drawColor = (255,0,255)

                    elif 580<x1<680:
                        self.header = self.overlayList[2]
                        self.drawColor = (255,0,0)

                    elif 830<x1<930:
                        self.header = self.overlayList[3]
                        self.drawColor = (0,255,0)

                    elif x1>1100:
                        self.header = self.overlayList[0]
                        self.drawColor = (0,0,0)
                        # print("Eraser to be added!")

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)

            # 5. Drawing mode - Index finger is up
            elif fingers[1]:
                cv2.circle(img,(x1,y1),15,self.drawColor,cv2.FILLED)
                # print("Write Mode")
                self.mode = "Write Mode"
                if self.xp ==0 and self.yp==0:
                    self.xp,self.yp = x1,y1
                if self.drawColor == (0,0,0):
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
                else:
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)
                self.xp, self.yp = x1, y1


        imgGray = cv2.cvtColor(self.imgCanvas,cv2.COLOR_BGR2GRAY)
        _,imgInv = cv2.threshold(imgGray,100,255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,self.imgCanvas)

        # setting the header image
        img[:125,:1280] = self.header
        # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
        # cv2.imshow('AirBoard',img)
        # cv2.imshow('Black Console', imgCanvas)

        img = cv2.putText(img, self.mode_disp + self.mode, self.mode_coord, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        img = cv2.putText(img, self.font_disp + self.font_d, self.font_coord, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)

        # ret, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()