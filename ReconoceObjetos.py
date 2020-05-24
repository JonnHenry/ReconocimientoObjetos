import cv2
import numpy as np

#rutaArchivo: Si no se desea el reconocimiento de objetos con un video ya hecho, se pone la ruta en ""
#reconocimientoCamara: Si es True se reconoce con la camara
#lados: Este es un parametro que funciona de la siguiente manera:
#lados==3 -> Se reconoce un triangulo
#lados==40 -> Se reconoce un rectangulo
#lados==41 -> Se reconoce un cuadrado
#lados==10 -> Se reconoce un circulo
#lados==-1 -> Se reconoce todo lo anterior

def reconoceObjetos(rutaArchivo,lados,reconocimientoCamara):
    capturaVideo = None
    centroTriangulo = []
    centroRectangulo = []
    centroCuadrado = []
    centroRectangulo = []
    centroCirculo = []
    cantCentros = 0
    pasoDiferencial = 10
    if (reconocimientoCamara):
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
        frameEscalaGrises = cv2.medianBlur(frameEscalaGrises,5)
        #Se muestra el video a escala de grises
        cv2.imshow("Escala de Grises", frameEscalaGrises)

        #Se detecta los bordes
        bordesFrame = cv2.Canny(frameEscalaGrises, 50, 225)
        bordesFrame = cv2.dilate(bordesFrame, None, iterations=2)
        bordesFrame = cv2.erode(bordesFrame, None, iterations=2)
        bordesFrame = cv2.morphologyEx(bordesFrame, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cv2.imshow("Bordes del Video", bordesFrame)

        #Encuentra los contornos de los bordes en la imagen
        #contornos,_ = cv2.findContours(bordesFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contornos,_ = cv2.findContours(bordesFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            areaContorno = cv2.contourArea(contorno)
            if areaContorno>2500:
                epsilon = 0.01*cv2.arcLength(contorno,True)
                aproximacion = cv2.approxPolyDP(contorno,epsilon,True)
                x,y,w,h = cv2.boundingRect(aproximacion)
                momento = cv2.moments(contorno)
                centro = (int(momento["m10"] / momento["m00"]), int(momento["m01"] / momento["m00"]))
                #Para el triangulo
                if (len(aproximacion)==lados or lados==-1):
                    cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),2)    
                    cv2.putText(frameCopia,'Triangulo', (x,y-5),1,1.5,(0,255,0),2)      
                    cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                    
                    centroTriangulo.append(centro)
                    cantCentros = len(centroTriangulo)-1
                    if (cantCentros>pasoDiferencial):
                        cv2.putText(frameCopia,'Dx:'+str(centroTriangulo[cantCentros][0]-centroTriangulo[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroTriangulo[cantCentros][1]-centroTriangulo[cantCentros-pasoDiferencial][1]), (x,y-30),1,1.5,(0,0,255),2)

                #Para un cuadrado 
                if (len(aproximacion)==4 or lados==-1):
                    relacionAspecto = int(w/h)
                    if (relacionAspecto==1 and lados==41):
                        cantCentros = len(centroCuadrado)
                        cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),2)    
                        cv2.putText(frameCopia,'Cuadrado', (x-5,y-10),1,1.5,(0,255,0),2)      
                        cv2.circle(frameCopia,centro, 3, (0,0,255), -1)

                        centroCuadrado.append(centro)
                        cantCentros = len(centroCuadrado) -1 
                        if (cantCentros>pasoDiferencial):
                            cv2.putText(frameCopia,'Dx:'+str(centroCuadrado[cantCentros][0]-centroCuadrado[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroCuadrado[cantCentros][1]-centroCuadrado[cantCentros-pasoDiferencial][1]), (x,y+20),1,1.5,(0,0,255),2)
                           
                    else:

                        cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),2)    
                        cv2.putText(frameCopia,'Rectangulo', (x,y-5),1,1.5,(0,255,0),2)      
                        cv2.circle(frameCopia,centro, 3, (0,0,255), -1)

                        centroRectangulo.append(centro)
                        cantCentros = len(centroRectangulo)-1
                        if (cantCentros>pasoDiferencial):
                            cv2.putText(frameCopia,'Dx:'+str(centroRectangulo[cantCentros][0]-centroRectangulo[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroRectangulo[cantCentros][1]-centroRectangulo[cantCentros-pasoDiferencial][1]), (x+30,y),1,1.5,(0,0,255),2)
                            

                #if (len(aproximacion)>15 or lados==6):
                #    cv2.drawContours(frameCopia,[aproximacion],-1,(255,0,0),2)    
                #    cv2.putText(frameCopia,'Circulo', (x,y-5),1,1.5,(0,255,0),2)      
                #    cv2.circle(frameCopia,centro, 3, (0,0,255), -1)
                #
                #    centroCirculo.append(centro)
                #    cantCentros = len(centroCirculo)-1
                #    if (cantCentros>pasoDiferencial):
                #        #cv2.putText(frameCopia,'Dx:'+str(centroCirculo[cantCentros][0]-centroCirculo[cantCentros -pasoDiferencial][0])+' Dy:'+str(centroCirculo[cantCentros][1]-centroCirculo[cantCentros-pasoDiferencial][1]), (x,y+5),1,1.5,(0,255,0),2)
                #        print(cantCentros)

            cv2.imshow("Reconocimiento", frameCopia)


        tecla = cv2.waitKey(10)
        if (tecla == 27):
            break

    #Se libera el objeto que estaba capturando el video
    capturaVideo.release()

    #Se cierra todas las ventanas
    cv2.destroyAllWindows()


reconoceObjetos('',-1,True)

