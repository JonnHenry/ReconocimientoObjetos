import cv2
import numpy as np
import time
import math
import os.path
import cvui
#rutaArchivo: Si no se desea el reconocimiento de objetos con un video ya hecho, se pone la ruta en ""

def reconoceObjetos(rutaArchivo):
    capturaVideo = None
    centroTriangulo = [(0,0)]
    centroRectangulo = [(0,0)]
    centroCuadrado = [(0,0)]
    centroRectangulo = [(0,0)]
    centroCirculo = [(0,0)]
    contTriangulo = 0
    contRectangulo = 0
    contCirculo = 0
    contCuadrado = 0
    pasoDiferencial = 10
    codec = cv2.VideoWriter_fourcc(*'XVID')
    
    
    nameRC = ['Grabar','Detener']
    inameRC=0
    iname=0
    triangle = [True]
    circle = [True]
    rectangle=[True]
    square=[True]
    WINDOWS_NAME='Reconocimiento de Figuras Geometricas'
    error=''
    if not os.path.isfile(rutaArchivo):
        error='El video no se encuentra en el directorio'
    record = False
    salir =0
    tecla=0
   
    while 1:
        name = 'WebCam' if iname%2 else 'Video'
        if (name=='WebCam' and os.path.isfile(rutaArchivo)):
            capturaVideo = cv2.VideoCapture(rutaArchivo)
        else:
            capturaVideo=cv2.VideoCapture(0)
        capturaVideo.set(3,640)
        capturaVideo.set(4, 480)

        time.sleep(2.0)
        
       
        cvui.init(WINDOWS_NAME)
        bandera=True
       
        while(bandera):
            continuaVideo, videoOriginal = capturaVideo.read()
            if continuaVideo == False:
                break

            frameCopia = videoOriginal.copy()

            frameEscalaGrises = cv2.cvtColor(frameCopia,cv2.COLOR_BGR2GRAY)
            frameEscalaGrises=cv2.GaussianBlur(frameEscalaGrises,(7,7),0)
   
            bordesFrame = cv2.Canny(frameEscalaGrises, 50, 225)
            bordesFrame = cv2.morphologyEx(bordesFrame, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            bordesFrame = cv2.dilate(bordesFrame, None, iterations=2)
            bordesFrame = cv2.erode(bordesFrame, None, iterations=2)
                     
            contornos,_ = cv2.findContours(bordesFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            cannyscaler = cv2.resize(bordesFrame,None,fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)
            grayscaler = cv2.resize(frameEscalaGrises,None,fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)
            scaler = cv2.vconcat([grayscaler,cannyscaler])

            cvui.text(frameCopia, 15,30,error,0.6,0xff0000)
            cvui.window(frameCopia, 10, 50, 0, 180, 'Escoja la figura geometrica')
            cvui.checkbox(frameCopia, 15, 80, 'Triangulo: ', triangle)
            cvui.checkbox(frameCopia,15,110,'Cuadrado: ',square)
            cvui.checkbox(frameCopia,15,140,'Rectangulo: ',rectangle)
            cvui.checkbox(frameCopia,15,170,'Circulo',circle)
            cvui.text(frameCopia,15,190,'Medio:')
            
            
            imgc = cv2.cvtColor(scaler,cv2.COLOR_GRAY2BGR)
           
            for contorno in contornos:
                areaContorno = cv2.contourArea(contorno)

                if areaContorno>2500:
                    epsilon = 0.01*cv2.arcLength(contorno,True)
                    aproximacion = cv2.approxPolyDP(contorno,epsilon,True)
                    x,y,w,h = cv2.boundingRect(aproximacion)
                    momento = cv2.moments(contorno)
                    centro = (int(momento["m10"] / momento["m00"]), int(momento["m01"] / momento["m00"]))
                    
                    if (len(aproximacion)==3 and triangle[0]):
                        cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                        cv2.putText(frameCopia,'Triangulo', (x,y-5),1,1.5,(0,255,0),2)      
                        cv2.circle(frameCopia,centro, 3, (0,0,255), -1)

                        if (centroTriangulo[0][0]-centro[0]!=0 or centroTriangulo[0][1]-centro[1]!=0 ):
                            cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)
                        
                        if contTriangulo % pasoDiferencial == 0:
                            centroTriangulo[0]=centro

                        contTriangulo += 1
                        
                    if (len(aproximacion)==4 and len(aproximacion)!=3):
                        relacionAspecto = 1 if .95<(w/h) and (w/h)<1.05 else 0
                        if (relacionAspecto==1 and square[0]):
                            cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                            cv2.putText(frameCopia,'Cuadrado', (x,y-5),1,1.5,(0,255,0),2)      
                            cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                            
                            if (centroCuadrado[0][0]-centro[0]!=0 or centroCuadrado[0][1]-centro[1]!=0):
                                cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)
                                
                            if contCuadrado % pasoDiferencial == 0:
                                centroCuadrado[0]=centro

                            contCuadrado += 1

                        if (relacionAspecto!=1 and rectangle[0]):

                            cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                            cv2.putText(frameCopia,'Rectangulo', (x,y-5),1,1.5,(0,255,0),2)      
                            cv2.circle(frameCopia,centro, 3, (0,0,255), -1)

                            if (centroRectangulo[0][0]-centro[0]!=0 or centroRectangulo[0][1]-centro[1]!=0):
                                cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)

                            if contRectangulo % pasoDiferencial == 0:
                                centroRectangulo[0]=centro

                            contRectangulo += 1   
 
                    if (len(aproximacion)>10 and circle[0]):                  
                        perimetro = cv2.arcLength(contorno, True)
                        if perimetro != 0:
                            circularidad = 4*math.pi*(areaContorno/(perimetro*perimetro))
                            if 0.85 < circularidad < 1.2:
                                cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                                cv2.putText(frameCopia,'Circulo', (x,y-5),1,1.5,(0,255,0),2)      
                                cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                                if (centroCirculo[0][0]-centro[0]!=0 or centroCirculo[0][1]-centro[1]!=0):
                                    cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)
                                if contCirculo % pasoDiferencial == 0:
                                    centroCirculo[0]=centro
                                contCirculo += 1
                
                
                
                both = np.hstack((frameCopia,imgc))
                if cvui.button(frameCopia, 15, 210, name):
                    iname+=1
                    bandera=False
                if cvui.button(frameCopia, 15, 242, nameRC[inameRC%2]):
                    if not inameRC%2:
                        print('Grabando....')
                        grabar = cv2.VideoWriter('videoGrabado'+str(inameRC)+'.avi', codec, 220, (960, 480))
                        record=True
                    else:
                        grabar.release()
                    inameRC+=1
                if record:
                    grabar.write(both)
                if cvui.button(frameCopia, 15, 274, "&Salir"):
                    salir=1
                cvui.update()
                cv2.imshow(WINDOWS_NAME,both)
            
            tecla = cv2.waitKey(10)

            if (tecla == 27 or salir):
                break
        if (tecla == 27 or salir):
                break

   
    capturaVideo.release()
    grabar.release()

    cv2.destroyAllWindows()

video =  input('Ingrese el nombre del video a reconocer: ')
reconoceObjetos(video)

