from email.mime import image
import cv2
import os
import numpy as np

dataPath = "C:/Users/VERGARA/Desktop/Deteccion de mascarillas/Dataset_faces"
dir_list = os.listdir(dataPath)
print("Lista de archivos:", dir_list)

labels = []
facesData = []
label = 0

for name_dir in dir_list:
    dirpath = dataPath + "/" + name_dir
    
    for file_name in os.listdir(dirpath):
        image_path = dirpath + "/" + file_name
        print(image_path)
        image = cv2.imread(image_path, 0)
        
        facesData.append(image)
        labels.append(label)
    label += 1 
    
print("Etiqueta 0: ", np.count_nonzero(np.array(labels) == 0))
print("Etiqueta 1: ", np.count_nonzero(np.array(labels) == 1))
      
#LBPH FaceRecognizen
face_mask = cv2.face.LBPHFaceRecognizer_create()

#entrenar el programa para que sepa distinguir 
print("Entrenando...")
face_mask.train(facesData, np.array(labels))

face_mask.write("face_mask_model.xml")
print("Modelo almacenado")
        