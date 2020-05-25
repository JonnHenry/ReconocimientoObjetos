import cv2
import numpy as np
import time
import math

#rutaArchivo: Si no se desea el reconocimiento de objetos con un video ya hecho, se pone la ruta en ""
#reconocimientoCamara: Si es True se reconoce con la camara
#listaReconocer: Este es un parametro que es una lista los valores que se utilizan son 
# triangulo
# cuadrado
# rectangulo
# circulo
# Ejemplo: ['cuadrado','triangulo','circulo']
# Si se desea reconocer todo se debe de utilizar todos los valores que se desea recononcer 

def reconoceObjetos(rutaArchivo,listaReconocer,reconocimientoCamara, grabarVideo):
    capturaVideo = None
    centroTriangulo = []
    centroRectangulo = []
    centroCuadrado = []
    centroRectangulo = []
    centroCirculo = []
    cantCentros = 0
    pasoDiferencial = 15
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    
    if (reconocimientoCamara):
        capturaVideo=cv2.VideoCapture(0)
    else:
        capturaVideo = cv2.VideoCapture(rutaArchivo)
        

    time.sleep(2.0)
    out = cv2.VideoWriter('videoGrabado.avi', fourcc, 20.0, (int(capturaVideo.get(3)),int(capturaVideo.get(4))))
    
    while(True):
        continuaVideo, videoOriginal = capturaVideo.read()
        if continuaVideo == False:
            break

        frameCopia = videoOriginal.copy()

        
        frameEscalaGrises = cv2.cvtColor(frameCopia,cv2.COLOR_BGR2GRAY)
        frameEscalaGrises=cv2.GaussianBlur(frameEscalaGrises,(7,7),0)
        #frameEscalaGrises = cv2.medianBlur(frameEscalaGrises,7)
        #Se muestra el video a escala de grises
        cv2.imshow("Escala de Grises", frameEscalaGrises)

        #Se detecta los bordes
        bordesFrame = cv2.Canny(frameEscalaGrises, 50, 225)
        bordesFrame = cv2.morphologyEx(bordesFrame, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        bordesFrame = cv2.dilate(bordesFrame, None, iterations=2)
        bordesFrame = cv2.erode(bordesFrame, None, iterations=2)
        
        cv2.imshow("Bordes del Video", bordesFrame)

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
                    
                    centroTriangulo.append(centro)
                    cantCentros = len(centroTriangulo)-1
                    if (cantCentros>pasoDiferencial):
                        cv2.putText(frameCopia,'Dx:'+str(centroTriangulo[cantCentros][0]-centroTriangulo[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroTriangulo[cantCentros][1]-centroTriangulo[cantCentros-pasoDiferencial][1]), (x,y+5),1,1.5,(0,0,255),2)

                #Para un cuadrado 
                if (len(aproximacion)==4):
                    relacionAspecto = int(w/h)
                    if (relacionAspecto==1 and 'cuadrado' in listaReconocer):
                        cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                        cv2.putText(frameCopia,'Cuadrado', (x-5,y-10),1,1.5,(0,255,0),2)      
                        cv2.circle(frameCopia,centro, 3, (0,0,255), -1)

                        centroCuadrado.append(centro)
                        cantCentros = len(centroCuadrado) -1 
                        if (cantCentros>pasoDiferencial):
                            cv2.putText(frameCopia,'Dx:'+str(centroCuadrado[cantCentros][0]-centroCuadrado[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroCuadrado[cantCentros][1]-centroCuadrado[cantCentros-pasoDiferencial][1]), (x,y+5),1,1.5,(0,0,255),2)
                           
                    if (relacionAspecto!=1 and 'rectangulo' in listaReconocer):

                        cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                        cv2.putText(frameCopia,'Rectangulo', (x,y-5),1,1.5,(0,255,0),2)      
                        cv2.circle(frameCopia,centro, 3, (0,0,255), -1)

                        centroRectangulo.append(centro)
                        cantCentros = len(centroRectangulo)-1
                        if (cantCentros>pasoDiferencial):
                            cv2.putText(frameCopia,'Dx:'+str(centroRectangulo[cantCentros][0]-centroRectangulo[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroRectangulo[cantCentros][1]-centroRectangulo[cantCentros-pasoDiferencial][1]), (x,y+5),1,1.5,(0,0,255),2)
                            

                if (len(aproximacion)>5 and 'circulo' in listaReconocer):                  
                    perimetro = cv2.arcLength(contorno, True)
                    if perimetro != 0:
                        circularidad = 4*math.pi*(areaContorno/(perimetro*perimetro))
                        if 0.7 < circularidad < 1.2:
                            cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),cv2.LINE_AA)    
                            cv2.putText(frameCopia,'Circulo', (x,y-5),1,1.5,(0,255,0),2)      
                            cv2.circle(frameCopia,centro, 3, (0,0,255), -1)

                            centroCirculo.append(centro)
                            cantCentros = len(centroCirculo)-1
                            if (cantCentros>pasoDiferencial):
                                cv2.putText(frameCopia,'Dx:'+str(centroCirculo[cantCentros][0]-centroCirculo[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroCirculo[cantCentros][1]-centroCirculo[cantCentros-pasoDiferencial][1]), (x,y+5),1,1.5,(0,255,0),2)

            cv2.imshow("Reconocimiento", frameCopia)

        tecla = cv2.waitKey(20)
        #Esc key para detenerse Esc==27
        if (tecla == 27):
            if (grabarVideo):
                out.write(frameCopia)

            break

    #Se libera el objeto que estaba capturando el video
    capturaVideo.release()

    #Se cierra todas las ventanas
    cv2.destroyAllWindows()


reconoceObjetos('figuras.mp4',['rectangulo', 'cuadrado','circulo'],True,True)

