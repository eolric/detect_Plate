# Librerías
import cv2
import numpy as np
import pytesseract
import glob
import re
import imutils

#Para windows, hay que especificar la ruta de tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#open files images jpg
def load_images(path_name):
    """
    - Name: load_images
    - Description: Carga las imagenes jpg del Path seleccionado
    - Parameters: 
        - path_name: nombre del path o la ruta de la carpeta
    - Returns:
        - names: Lista con los archivos jpg que contenga el path 
        - iterations: número de imágenes, tamaño de names
    """
    names = glob.glob(path_name+"*.jpg")
    print("Imágenes detectadas: ", len(names))
    iterations = range(len(names))
    return names, iterations

def read_and_filter_images(name):
    """
    - Name: read_and_filter_images
    - Description: Lee y filtra las imagenes.
        - Se aplica primero un filtro para reducir el ruido de la imagen mientras
            se preservan los bordes nítidos (bilateralFilter).
        - Luego se aplica un filtro HSV para segmentar y manipular imágenes en
            función de los colores. Hue (tono), Saturation (saturación) y Value (valor o brillo)
    - Parameters:
        - names: nombre de la imagen
    - Returns:
        - img: imagen original 
        - img_filterHSV: imagen filtrada
    """
    img = cv2.imread(name)  # Se lee la imagen
    img_clean = cv2.bilateralFilter(img.copy(), -1, 5, 5)  # Se reduce el ruido
    # Definir los valores para el filtro HSV
    lower = np.array([35/2,90,90])
    upper = np.array([90/2,255,255])
    hsv = cv2.cvtColor(img_clean.copy(), cv2.COLOR_BGR2HSV)
    # Aplicación del umbral para obtener colores específicos
    hsv_mask = cv2.inRange(hsv.copy(), lower, upper)
    # Aplicar las operaciones morfológicas
    struct_elem_mask = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
    closing_mask = cv2.morphologyEx(hsv_mask.copy(), 
                                    cv2.MORPH_CLOSE,struct_elem_mask)  # para cerrar pequeños agujeros dentro de los objetos
    opening_mask = cv2.morphologyEx(closing_mask,
                                    cv2.MORPH_OPEN,struct_elem_mask)  # para eliminar pequeños objetos dentro del área de interés
    img_filterHSV =  cv2.bitwise_and(img_clean,img_clean, mask=opening_mask)
    return img, img_filterHSV

def border_detect_and_count(image, sigma= 0.99):
    """
    - NAME: border_detect_and_count
    - DESCRIPTION: Detección de bordes utilizando el algoritmo de Canny 
    con umbrales superior e inferior generados estadísticamente.
    - PARAMETERS: 
        - image: imagen de entrada para detectar bordes
        - sigma: valor de sigma para calcular los umbrales según el valor mediano
    - RETURNS:    
        - bordes: imagen filtrada de salida del mismo tamaño que la entrada
        con los bordes 
    """
    v = np.median(image)  # Cálculo de la mediana de las intensidades de píxeles
    # Aplicación automática de la detección de bordes de Canny usando la mediana
    lower = int(max(0, (1.0 - sigma) * v)) 
    upper = int(min(255, (1.0 + sigma) * v)) 
    edged = cv2.Canny(image, lower, upper)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encontrar contornos. RETR_EXTERNAL recupera solo los contornos externos y descarta los contornos anidados.
    return contours

def get_boxes_and_filter5(img_original, contours):
    """
    - NAME: get_boxes_and_filter5
    - DESCRIPTION: Obtención de los recuadros de los bordes encontrados 
    y se escogen los 5 bordes con áreas más grandes. También se grafica los 5 recuadros sobre la imagen
    - PARAMETERS: 
        - img_original: imagen original
        - contours: contronos encontrados en la imagen
    - RETURNS:    
        - contours5: contiene 5 contornos de los recuadros con las mayores áreas
    """
    # Se obtienen los recuadros para cada contorno encontrado en la imagen 
    rectangles_pre = []
    for j in range(len(contours)):
        rect = cv2.minAreaRect(contours[j])
        approx = cv2.boxPoints(rect)
        approx = approx.astype(int)  # MOD EDU
        rectangles_pre.append(approx.reshape(4,1,2))
    # Filtrado de los 5 recuadros con mayor área de cuadro, "ojito, puede cambiar dependiendo de la distacia de las placas en las imágenes"
    contours5 = sorted(rectangles_pre,key=cv2.contourArea, reverse = True)[:5]
    # Se dibuja la imagen de los 5 recuadros
    area5_contours = cv2.drawContours(img_original.copy(), contours5,
                                       -1, (0, 255, 0), 2)
    cv2.imshow('5_recuadros', area5_contours)
    return contours5

def filter_aspectratio(contours):
    """
    - NAME: filter_aspectratio
    - DESCRIPTION: Se filtran los recuadros por el por su aspect ratio
    - PARAMETERS: 
        - contours: contronos encontrados en la imagen
    - RETURNS:    
        - the_rectangle: contiene el recuadro de la placa
    """
    rectangles = []
    error = 100
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])  # Calcula el rectángulo de área mínima que puede encerrar el contorno actual (con rotación permitida).
        approx = cv2.boxPoints(rect)  # Obtiene los cuatro puntos del rectángulo rotado.
        approx = approx.astype(int)  # Convierte las coordenadas a enteros.
        [x, y, w, h] = cv2.boundingRect(approx)  # Calcula las medidas del cuadro delimitador sin rotación.
        aspect_ratio = w/h  # Se calcula el aspect ratio
        #### Para colombia las medidas de las placas son 330 mm de ancho y 160 mm de alto
        aspect_error = abs(2-aspect_ratio)  # es la diferencia absoluta entre la relación de aspecto calculada y la relación de aspecto teórica deseada que es 2   
        # Se deja el recuadro con el menor error
        if(aspect_error < error):  # Si el aspect_error actual es menor que el error anterior, se actualiza el error y se guarda el cuadro delimitador             
            error = aspect_error
            the_rectangle = approx.reshape(4,1,2)
        # MOD EDU
        # if(1.9 < aspect_ratio and aspect_ratio < 2.5):            
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     placa = gray[y:y+h, x:x+w]
        #     placa_text = pytesseract.image_to_string(placa, config='--psm 11')
        #     cv2.imshow('Placa', placa)
        #     print('Placa= ', placa_text)
        return the_rectangle

def transformada_perspectiva(img, bound):
    """
    - NAME: transformada_perspectiva
    - DESCRIPTION: Esta función realiza una transformación de perspectiva en una placa utilizando
    una imagen original y un cuadro delimitador de 4 puntos.
    - PARAMETERS: 
        - img: imagen original
        - bound: 4 puntos del recuadro de la placa                        
    - RETURNS:
        - warped: contiene la transformada de perspectiva
        - corner_plate: Esquinas de la placa
    """
    # Se halla la forma de la imagen 
    h, w, _ = img.shape
    # Se halla las coordenadas de la imagen
    img_corners= np.float32(np.array([[0,0],
                           [w-1,0],
                           [w-1, h-1],
                           [0, h-1]]))
    
    # Ordenar los puntos en sentido de las agujas del reloj  
    rect = np.zeros((4, 2), dtype="float32")  # Crea una lista de coordenadas que estará ordenada en el sentido de las agujas del reloj: la primera entrada será la esquina superior izquierda
    s = bound.sum(axis=1)
    rect[0] = bound[np.argmin(s)]
    rect[2] = bound[np.argmax(s)]
    diff = np.diff(bound, axis=1)
    rect[1] = bound[np.argmin(diff)]
    rect[3] = bound[np.argmax(diff)]
    # Ya se han ordenado los puntos en sentido de las agujas del reloj
    bound2 = rect

    # Redimensionar el tamaño del recuadro
    plate_2 = img.copy()
    xt = 0
    yt = 0
    bound3 = np.array([[bound2[0,0]-xt, bound2[0,1]-yt], 
                       [bound2[1,0]+xt, bound2[1,1]-yt],
                       [bound2[2,0]+xt, bound2[2,1]+yt],
                       [bound2[3,0]-xt, bound2[3,1]+yt]], dtype=np.uint16)
         
    bound3_f = np.float32(bound3)
    
    # Obtener la matriz de transformación de perspectiva
    T = cv2.getPerspectiveTransform(bound3_f, img_corners) 
    
    # Retornar la imagen transformada y la imagen original con las esquinas del cuadro delimitador
    warped = cv2.warpPerspective(img.copy(), T, (w,h))
    
    # RETURN warped image and original image+corners of bounding box
    return warped

def plate_processing(plate):
    """
    - NAME: plate_processing
    - DESCRIPTION: Esta función procesa la imagen de la placa detectada.
    La convierte a escala de grises, la filtra para eliminar ruido y resaltar los caracteres 
    - PARAMETERS: 
        - plate: imagen de la placa                    
    - RETURNS:
        - warped: contiene la transformada de perspectiva
    """
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) # Convertimos la imagen a escala de grises

    # Aplicamos thresholding automático con el algoritmo de Otsu. Esto hará que el texto se vea blanco, y los elementos
    # del fondo sean menos prominentes.
    thresholded = cv2.threshold (gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    #Calculamos y normalizamos la transformada de distancia.
    dist = cv2.distanceTransform (thresholded, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = (dist*255).astype('uint8')
    # Aplicamos thresholding al resultado de la operación anterior, y mostramos el resultado en pantalla.
    dist = cv2.threshold (dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Aplicamos apertura para desconectar manchas y blobs de los elementos que nos interesan (los números)
    kernel = cv2.getStructuringElement (cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)

    # Hallamos los contornos de los números en la imagen.
    contours = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Solo los contornos grandes perdurarán, ya que corresponden a los números que nos interesan.
        if w >= 35 and h >= 100:
            chars.append(contour)

#main
#path imgaes
path_name = 'images_Placas/'
names, iterations = load_images(path_name)  # Se cargan las imágenes del path

for i in iterations:  # Se recorre cada imagen
    print("Iteración: ", i+1)
    img_original, img_filter = read_and_filter_images(names[i])  # Se filtra la imagen
    contours = border_detect_and_count(img_filter)  # Se hallan los contornos
    area5_contours = get_boxes_and_filter5(img_original, contours)
    rectangle_placa = filter_aspectratio(area5_contours)
    trans_prespect = transformada_perspectiva(img_original, np.int32(rectangle_placa.reshape(4,2)))
    
    cv2.waitKey(0)
    