import csv
import subprocess
import os
import numpy as np
import cv2
import os
import random
from Xml_formatter import Xml_formatter, image_resize

# Modes = ["train", "validation"]
Modes = ["train", "validation"]
Dataset = "Dataset3/"
Final_Size = 512
validation_ratio = 0.25

# Carico i dataset a cui sono interessato e le quantità di immagini che voglio scaricare
f = open("classes.names", "r", encoding="UTF-8")
classes = []
quantity = []
for x in f:
    if x[0] != "#":
        classes.append(x.strip().split("-")[0])
        quantity.append(int(x.strip().split("-")[1]))
f.close()

# Crea un dizionario per ogni classe con il codice per il download
with open('./csv_folder/class-descriptions-boxable.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]: rows[0] for rows in reader}

# Creo la cartella dataset
subprocess.run(['rm', '-rf', Dataset])
subprocess.run(['mkdir', Dataset])

for runMode in Modes:

    print("-Scarichiamo il", runMode, "set")

    if runMode == "validation":
        for x in range(len(quantity)):
            quantity[x] = int(float(quantity[x]) * validation_ratio)

    for ind in range(len(classes)):

        # Creo delle variabili locali per richiamarle più facilmente
        className = classes[ind]
        Data_Quantity = quantity[ind]
        if runMode == Modes[0]:
            subprocess.run(['mkdir', Dataset + className + "/"])
        directory_image = Dataset + className + "/" + runMode + '/image/'
        directory_annotation = Dataset + className + "/" + runMode + '/annotation/'
        directory_review = Dataset + className + "/" + runMode + '/review/'

        # Creo una sotto directory per ogni classe di interesse
        subprocess.run(['mkdir', Dataset + className + "/" + runMode + '/'])
        subprocess.run(['mkdir', directory_image])
        subprocess.run(['mkdir', directory_annotation])
        subprocess.run(['mkdir', directory_review])

        # Mi faccio una regex per caricare tutte le linee che contengono una bounding box della classe che mi interessa
        commandStr = "grep " + dict_list[className] + " ./csv_folder/" + runMode + "-annotations-bbox.csv"
        # Applico la regex e ottengo class_annotations array di tutte le righe che mi interessano
        class_annotations = subprocess.run(commandStr.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        class_annotations = class_annotations.splitlines()

        # Mi creo un array con tutte le diverse foto di una certa classe
        image_id = []
        image_id.append(class_annotations[0].split(',')[0])
        for line in class_annotations:
            lineParts = line.split(',')
            if lineParts[0] != image_id[len(image_id) - 1]:
                image_id.append(lineParts[0])

        # Printing immagini di interesse
        debug_0 = False
        if debug_0 is True:
            for x in image_id:
                print(x)

        # Printo quante immagini stiamo scaricando
        print("     Stiamo scaricando", str(Data_Quantity), "immagini su", str(len(image_id)), "della classe", className)

        # Faccio lo shuffle delle risorse
        # random.seed(20)
        # random.shuffle(image_id)

        # Riduco le immagini che devo scaricare
        image_id = image_id[0:Data_Quantity]

        conteggio = 0
        print('      ', '-' * 10, "%.2f" % float(0), "%", end="")
        for im in image_id:
            conteggio += 1
            full_progbar = 10
            filled_progbar = round(conteggio / Data_Quantity * full_progbar)
            print('\r       ', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), "%.2f" % float(conteggio / Data_Quantity * 100), "%", end="")

            # Mi faccio una regex per caricare tutte le linee che contengono l'immagine corrente
            commandStr2 = "grep " + im + " ./csv_folder/" + runMode + "-annotations-bbox.csv"
            # Applico la regex e ottengo current_annotations array di tutte le righe che mi interessano
            current_annotations = subprocess.run(commandStr2.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
            current_annotations = current_annotations.splitlines()

            current_annotations2 = []
            # Scarto dalle current_annotations tutte le classi non di interesse
            for i in range(len(current_annotations)):
                lineParts = current_annotations[i].split(',')
                for intrest in classes:
                    # print(lineParts[2], dict_list[intrest])
                    if lineParts[2] == dict_list[intrest]:
                        current_annotations2.append(current_annotations[i])

            # Sovrascrivo le current_annotations con la versione di solo interesse
            current_annotations = current_annotations2

            # Printing delle varie immagini con relative label
            debug_1 = False
            if debug_1 is True:
                print("-Codice Immagine:" + im)
                for pr_annotation in current_annotations:
                    print(pr_annotation)
                print()

            # Scarico l'immagine richiesta
            subprocess.run(['aws', 's3', '--no-sign-request', '--only-show-errors', 'cp', 's3://open-images-dataset/' + runMode + '/' + im + ".jpg", directory_image + '/' + im + ".jpg"])

            image = cv2.imread(directory_image + '/' + im + ".jpg")

            xml, image, image2 = Xml_formatter(current_annotations, runMode, image, classes)

            cv2.imwrite(directory_image + '/' + im + ".jpg", image)
            cv2.imwrite(directory_review + '/' + im + ".jpg", image2)

            g = open(directory_annotation + '/' + im + ".xml", "w", encoding="UTF-8")
            for x in xml:
                g.write(x)
                g.write("\n")
            g.close()
        print()
    print()
