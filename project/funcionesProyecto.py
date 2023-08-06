import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from skimage.feature import hog
from skimage import exposure
from skimage import segmentation
from sklearn.cluster import KMeans
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import naive_bayes
import csv
import os
from sklearn.model_selection import train_test_split


def funcionLeerClase(pathDirectorio):
    pathsCompletos = glob(pathDirectorio+'/*')
    imagenes = []
    for nombreImagen in pathsCompletos:
        imagen = cv2.imread(nombreImagen)
        imagenes.append(imagen)
    return imagenes


def funcionLeerTodasClases(pathDirectorio):
    etiquetasClases = ['10c','1c','1e','20c','2c','2e','50c','5c']
    lista1c = []
    lista2c = []
    lista5c = []
    lista10c = []
    lista20c = []
    lista50c = []
    lista1e = []
    lista2e = []
    vec_listas = [lista1c,lista2c,lista5c,lista10c,lista20c,lista50c,lista1e,lista2e]
    for i in range(len(etiquetasClases)):
        print('En',pathDirectorio+'/'+etiquetasClases[i], end = "")
        vec_listas[i] = funcionLeerClase(pathDirectorio+'/'+etiquetasClases[i])
        print(', hay', len(vec_listas[i]), 'ejemplos')
    return vec_listas


def ecualizacionAdaptativa(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(3,3))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    ecualizada = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype('uint8')
    return ecualizada


def imagen_media_color(imagen, nClusters):
    alg = KMeans(n_clusters=nClusters, n_init = 20)
    alg.fit(np.hstack((imagen[:,:,0].reshape(-1,1), imagen[:,:,1].reshape(-1,1), imagen[:,:,2].reshape(-1,1))))
    etiquetas = alg.labels_
    centros = alg.cluster_centers_
    nuevaImagen = np.zeros(imagen.shape)
    nuevaImagen = centros[etiquetas].reshape(imagen.shape)
    puntoCentro = (imagen.shape[0]//2,imagen.shape[1]//2)
    colorMoneda = nuevaImagen[puntoCentro]
    nuevaImagen = nuevaImagen/255
    return nuevaImagen, colorMoneda


def quitarFondo(img):
    imgBlurreada = cv2.blur(img,(3,3))
    imagenMedia, colorMoneda = imagen_media_color(imgBlurreada, nClusters = 2)
    puntoCentro = (img.shape[0]//2,img.shape[1]//2)
    imagenSinFondo = img.copy()
    posFondo = np.any(imagenMedia != imagenMedia[puntoCentro], axis = 2)
    #Ponemos el fondo del mismo color que la moneda
    imagenSinFondo[posFondo] = colorMoneda
    return imagenSinFondo


def generarHogClase(clase):
    claseHog = []
    for img in clase:
        img = cv2.medianBlur(img,3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagenSinFondo = quitarFondo(img)
        imagenSinFondo = ecualizacionAdaptativa(imagenSinFondo)
        fd = hog(imagenSinFondo, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(7, 7))
        claseHog.append(fd)
    array = np.array(claseHog)
    return array


def generarCsv(directorio, test_labels):
    contenido = os.listdir(directorio)
    test_image_ids = []
    for fichero in contenido:
        if os.path.isfile(os.path.join(directorio, fichero)) and fichero.endswith('.jpg'):
            test_image_ids.append(fichero)
    fields = ["Id", "Expected"] 
    filename = "results.csv"
    with open(filename, 'w', newline="") as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter=',') 
        csvwriter.writerow(fields) 
        for i in range(len(test_labels)):
            csvwriter.writerow([test_image_ids[i], int(test_labels[i])])
    return