import cv2
import numpy as np
import time
import math

#rutaArchivo: Si no se desea el reconocimiento de objetos con un video ya hecho, se pone la ruta en ""
#reconocimientoCamara: Si es True se reconoce con la camara
#grabarVideo: Es un parametro de tipo Boolean
#listaReconocer: Este es un parametro que es una lista los valores que se utilizan son 
# triangulo
# cuadrado
# rectangulo
# circulo
# Ejemplo: ['cuadrado','triangulo','circulo']
# Si se desea reconocer todo se debe de utilizar todos los valores que se desea recononcer 

def reconoceObjetos(rutaArchivo,listaReconocer,reconocimientoCamara, grabarVideo):
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
    cantCentros = 0
    pasoDiferencial = 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    if (reconocimientoCamara):
        capturaVideo=cv2.VideoCapture(0)
    else:
        capturaVideo = cv2.VideoCapture(rutaArchivo)
        

    time.sleep(2.0)
    grabar = cv2.VideoWriter('videoGrabado.avi', fourcc,40, (int(capturaVideo.get(3)),int(capturaVideo.get(4))))
    
    while(True):
        continuaVideo, videoOriginal = capturaVideo.read()
        if continuaVideo == False:
            break

        frameCopia = videoOriginal.copy()
        frameAux = frameCopia
        
        frameEscalaGrises = cv2.cvtColor(frameCopia,cv2.COLOR_BGR2GRAY)
        frameEscalaGrises=cv2.GaussianBlur(frameEscalaGrises,(7,7),0)
        #frameEscalaGrises = cv2.medianBlur(frameEscalaGrises,7)
        #Se muestra el video a escala de grises
        #cv2.imshow("Escala de Grises", frameEscalaGrises)

        #Se detecta los bordes
        bordesFrame = cv2.Canny(frameEscalaGrises, 50, 225)
        bordesFrame = cv2.morphologyEx(bordesFrame, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        bordesFrame = cv2.dilate(bordesFrame, None, iterations=2)
        bordesFrame = cv2.erode(bordesFrame, None, iterations=2)
        
        #cv2.imshow("Bordes del Video", bordesFrame)

        #Encuentra los contornos de los bordes en la imagen
        contornos,_ = cv2.findContours(bordesFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #contornos,_ = cv2.findContours(bordesFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            areaContorno = cv2.contourArea(contorno)

            if areaContorno>2500:
                epsilon = 0.01*cv2.arcLength(contorno,True)
                aproximacion = cv2.approxPolyDP(contorno,epsilon,True)
                x,y,w,h = cv2.boundingRect(aproximacion)
                momento = cv2.moments(contorno)
                centro = (int(momento["m10"] / momento["m00"]), int(momento["m01"] / momento["m00"]))
                #Para el triangulo
                if (len(aproximacion)==3 and 'triangulo' in listaReconocer):
                    cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                    cv2.putText(frameCopia,'Triangulo', (x,y-5),1,1.5,(0,255,0),2)      
                    cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                    cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)

                #Para un cuadrado 
                if (len(aproximacion)==4):
                    relacionAspecto = int(w/h)
                    if (relacionAspecto==1 and 'cuadrado' in listaReconocer):
                        cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                        cv2.putText(frameCopia,'Cuadrado', (x,y-5),1,1.5,(0,255,0),2)      
                        cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                        cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)
                        

                    if (relacionAspecto!=1 and 'rectangulo' in listaReconocer):

                        cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                        cv2.putText(frameCopia,'Rectangulo', (x,y-5),1,1.5,(0,255,0),2)      
                        cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                        cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)


                if (len(aproximacion)>10 and 'circulo' in listaReconocer):                  
                    perimetro = cv2.arcLength(contorno, True)
                    if perimetro != 0:
                        circularidad = 4*math.pi*(areaContorno/(perimetro*perimetro))
                        if 0.7 < circularidad < 1.2:
                            cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                            cv2.putText(frameCopia,'Circulo', (x,y-5),1,1.5,(0,255,0),2)      
                            cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                            cv2.putText(frameCopia,'x:'+str(centro[0])+' y:'+str(centro[1]), (x,y+15),1,1.5,(0,0,255),2)

            

            cv2.imshow("Reconocimiento", frameCopia)
            if (grabarVideo):
                grabar.write(frameAux)

        
        tecla = cv2.waitKey(5)
        #Esc key para detenerse Esc==27
        if (tecla == 27):
            break

    #Se libera el objeto que estaba capturando el video
    capturaVideo.release()
    grabar.release()

    #Se cierra todas las ventanas
    cv2.destroyAllWindows()


reconoceObjetos('figuras.mp4',['triangulo','rectangulo', 'cuadrado','circulo'],True,True)

