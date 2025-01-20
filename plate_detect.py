# Librerías
import cv2
import numpy as np
import pytesseract
import glob
import re

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

def filter_aspectratio(img, contours):
    """
    - NAME: filter_aspectratio
    - DESCRIPTION: Se filtran los recuadros por el por su aspect ratio
    - PARAMETERS: 
        - contours: contronos encontrados en la imagen
    - RETURNS:    
        - placa: contiene el recuadro de la placa
    """
    rectangles = []
    error = 100000000
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])  # Calcula el rectángulo de área mínima que puede encerrar el contorno actual (con rotación permitida).
        approx = cv2.boxPoints(rect)  # Obtiene los cuatro puntos del rectángulo rotado.
        approx = approx.astype(int)  # Convierte las coordenadas a enteros.
        [x, y, w, h] = cv2.boundingRect(approx)  # Calcula las medidas del cuadro delimitador sin rotación.
        aspect_ratio = w/h  # Se calcula el aspect ratio
        #### Para colombia las medidas de las placas son 330 mm de ancho y 160 mm de alto
        if(1.9 < aspect_ratio and aspect_ratio < 2.5):            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            placa = gray[y:y+h, x:x+w]
            placa_text = pytesseract.image_to_string(placa, config='--psm 11')
            cv2.imshow('Placa', placa)
            print('Placa= ', placa_text)

#main
#path imgaes
path_name = 'images_Placas/'
names, iterations = load_images(path_name)  # Se cargan las imágenes del path

for i in iterations:  # Se recorre cada imagen
    print("Iteración: ", i+1)
    img_original, img_filter = read_and_filter_images(names[i])  # Se filtra la imagen
    contours = border_detect_and_count(img_filter)  # Se hallan los contornos
    area5_contours = get_boxes_and_filter5(img_original, contours)
    placa = filter_aspectratio(img_filter, area5_contours)
    # 

    cv2.waitKey(0)
    