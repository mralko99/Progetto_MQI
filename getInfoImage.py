import csv
import subprocess
import sys
import numpy as np
import cv2
from Xml_formatter import Xml_formatter


Dataset = "Dataset2/"


# Crea un dizionario per ogni classe con il codice per il download
with open('./csv_folder/class-descriptions-boxable.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dict_list = {rows[0]: rows[1] for rows in reader}

Tipo = ["train", "validation", "test"]
annotation = []
current_type = ""

Foto = sys.argv
if len(Foto) > 2 or len(Foto) == 1:
    print("Errore nell'utilizzo del comando")
else:
    Nome = Foto[1]

    for i in Tipo:
        # Mi faccio una regex per caricare tutte le linee che contengono l'immagine corrente
        commandStr2 = "grep " + Nome + " ./csv_folder/" + i + "-annotations-bbox.csv"
        # Applico la regex e ottengo current_annotations array di tutte le righe che mi interessano
        current_annotations = subprocess.run(commandStr2.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        current_annotations = current_annotations.splitlines()
        if len(current_annotations) != 0:
            current_type = i
        for x in current_annotations:
            annotation.append(x)

    current_annotations2 = []
    # Scarto dalle current_annotations tutte le classi non di interesse
    for i in range(len(annotation)):
        lineParts = annotation[i].split(',')
        Classe = dict_list[lineParts[2]]
        if Classe not in current_annotations2:
            current_annotations2.append(Classe)

    # Scarico l'immagine richiesta
    subprocess.run(['aws', 's3', '--no-sign-request', '--only-show-errors', 'cp', 's3://open-images-dataset/' + current_type + '/' + Nome + ".jpg", Nome + ".jpg"])

    for i in current_annotations2:
        print(i)

    image = cv2.imread(Nome + ".jpg")

    xml, image, image2 = Xml_formatter(annotation, current_type, image, current_annotations2)

    cv2.imwrite(Nome + ".jpg", image2)
