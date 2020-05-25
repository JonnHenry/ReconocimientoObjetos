import cv2
import numpy as np 
###FUNCIONES
def getSides(contorno):
    epsilon = 0.01* cv2.arcLength(contorno,True)
    approx = cv2.approxPolyDP(contorno,epsilon,True)
    #print(approx0
    return len(approx)>2 #and len(approx) <20

####################
video = cv2.VideoCapture(0)

black_img = np.zeros((960,960,3),np.uint8)
lower_red = np.array([0,0, 0])#60, 66, 134]
upper_red = np.array([100,100,100 ]) #180, 255, 24350
kernelBorde = np.ones((7,7),np.uint8)

while 1:
    _,frame = video.read()
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(7,7),0)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #bordes = cv2.Canny(gray,10, 200)
    #cv2.imshow('Gausiana',gray)
    #mascara = cv2.inRange(hsv,lower_red,upper_red,gray)
    mascara = cv2.medianBlur(gray,5)
    #bordes = cv2.medianBlur(bordes,11)
    #bordes = cv2.dilate(bordes, kernelBorde, iterations=1)
    #bordes = cv2.erode(bordes, kernelBorde, iterations=1)
    #print(mascara )
    #T= abs(mascara -255)
    #mascara = cv2.medianBlur(mascara,51)
    #cv2.imshow('Original', frame)
    #cv2.imshow('mask', mascara)
    #contorno,_ = cv2.findContours(bordes.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    circles = cv2.HoughCircles(mascara,cv2.HOUGH_GRADIENT,1,10, param1=50,param2=12,minRadius=50,maxRadius=100)

    circles = np.uint16(np.around(circles))


    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('Borde',mascara)
    cv2.imshow('Deteccion',frame)
    #cv2.waitKey(0)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
video.release()