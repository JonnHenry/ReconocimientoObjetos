import cv2 
import numpy as np
import imutils
import time

from collections import deque

#creamos el objeto de video (camara)
captura=cv2.VideoCapture(0)

#captura = cv2.VideoCapture("figuras.mp4")
total=0
pts = deque()
(dx, dy) = (0, 0)

while True:

    ret, frame = captura.read()

    if ret == False:
        break

    frameAux = frame.copy()
    
    gray = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)

    gray=cv2.GaussianBlur(frameAux,(5,5),0)
    
    cv2.imshow("SinEfectos 5656", gray)

    bordes = cv2.Canny(gray, 10, 225)
    
    bordes = cv2.dilate(bordes, None, iterations=1)
    bordes = cv2.erode(bordes, None, iterations=1)

    cv2.imshow("SinEfectos", bordes)

    cnts,_ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4

    #cnts = imutils.grab_contours(cnts)
    center = None
    for c in cnts:
        area=cv2.contourArea(c)
        if area>2500:

            #aproximacion de contorno
            epsilon = 0.01*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            #print(len(approx))
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #print(center)
            #Si la aproximacion tiene 4 vertices correspondera a un rectangulo (Libro)

            if len(approx)==4:
                cv2.drawContours(frame,[approx],-1,(255,0,0),3,cv2.LINE_AA)
                cv2.circle(frame,center, 3, (0,0,255), -1)
                pts.append(center)
                #print(len(pts))
                #cv2.putText(frame,"(x: " + str(cx) + ", y: " + str(cy) + ")",(cx+10,cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)
                

                for i in range(1,len(pts)):
                    if pts[i-1] is None or pts[i] is None:
                        continue

                    if (len(pts)>=10):
                        dx=pts[-10][0]-pts[i][0]
                        dy=pts[-10][1]-pts[i][1]
                
                    thickness = int(np.sqrt(1000 / float(i + 1)) * 2.5)
                    #cv2.line(frame,pts[i-1],pts[i],(0,0,255),thickness=thickness) 
                
            cv2.putText(frame,"dx: "+str(dx)+"   dy: "+str(dy),(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)

                    
    #Mostramos imagen
    cv2.imshow("video", frame)
    
    k = cv2.waitKey(20)
    if (k == 27):
        break

#Liberamos Objeto
captura.release()

#Destruimos Ventanas
cv2.destroyAllWindows()



