import cv2
import numpy as np

#rutaArchivo: Si no se desea el reconocimiento de objetos con un video ya hecho, se pone la ruta en ""
#reconocimientoCamara: Si es True se reconoce con la camara
#lados: Este es un parametro que funciona de la siguiente manera:
#lados==3 -> Se reconoce un triangulo
#lados==40 -> Se reconoce un rectangulo
#lados==41 -> Se reconoce un cuadrado
#lados==5 -> Se reconoce un circulo
#lados==6 -> Se reconoce todo lo anterior



def reconoceObjetos(rutaArchivo,lados,reconocimientoCamara):
    capturaVideo = None
    contTriangulo = 0
    contRectangulo = 0
    contCuadrado = 0
    contCirculo =0
    centroTriangulo = []
    centroRectangulo = []
    centroCuadrado = []
    centroRectangulo = []
    if (reconocimientoCamara==True):
        capturaVideo=cv2.VideoCapture(0)
    else:
        capturaVideo = cv2.VideoCapture("TomadoVideo.mp4")
    
    while(True):
        continuaVideo, videoOriginal = capturaVideo.read()
        if continuaVideo == False:
            break

        frameCopia = videoOriginal.copy()
        frameAnalisis=cv2.GaussianBlur(frameCopia,(7,7),0)
        frameEscalaGrises = cv2.cvtColor(frameAnalisis,cv2.COLOR_BGR2GRAY)
        #Se muestra el video a escala de grises
        cv2.imshow("Escala de Grises", frameEscalaGrises)

        #Se detecta los bordes
        bordesFrame = cv2.Canny(frameEscalaGrises, 50, 225)
        bordesFrame = cv2.dilate(bordesFrame, None, iterations=2)
        bordesFrame = cv2.erode(bordesFrame, None, iterations=2)
        bordesFrame = cv2.morphologyEx(bordesFrame, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cv2.imshow("Bordes del Video", bordesFrame)

        #Encuentra los contornos de los bordes en la imagen
        contornos,_ = cv2.findContours(bordesFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            areaContorno = cv2.contourArea(contorno)
            if areaContorno>2500:
                epsilon = 0.01*cv2.arcLength(c,True)
                aproximacion = cv2.approxPolyDP(contorno,epsilon,True)
                x,y,w,h = cv2.boundingRect(aproximacion)
                momento = cv2.moments(contorno)
                centro = (int(momento["m10"] / momento["m00"]), int(momento["m01"] / momento["m00"]))
                if (len(aproximacion)==lados or lados==6):
                    cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),2)    
                    cv2.putText(frameCopia,'Triangulo', (x,y-5),1,1.5,(0,255,0),2)      
                    cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                    
                    if (contTriangulo==0):
                        centroTriangulo.append(centro)
                    
                    if (contTriangulo==5):
                        centroTriangulo.append(centro)
                        cv2.putText(frameCopia,'Dx:'+str(centroTriangulo[1][0]-centroTriangulo[0][0])+' Dy:'+str(centroTriangulo[1][1]-centroTriangulo[0][1]), (x,y+5),1,1.5,(0,255,0),2)
                        contTriangulo = 0
                        centroTriangulo = []

                    contTriangulo+=1






