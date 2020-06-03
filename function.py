import csv
import cv2
import subprocess
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    # Mi calcolo altezza e larghezza dalla foto
    (h, w) = image.shape[:2]

    # Vedo quale delle due dimensioni devo modificare
    if height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        return

    # Faccio il primo resize alla dimensione più grande
    resized = cv2.resize(image, dim, interpolation=inter)

    # Mi calcolo i padding
    condizione = h < w
    if condizione:
        yRidimensionata = 512 / w * h
        xOffset = 0
        yOffset = int((512 - yRidimensionata) / 2)
    else:
        xRidimensionata = 512 / h * w
        xOffset = int((512 - xRidimensionata) / 2)
        yOffset = 0
    # La trasformo in un immagine quadrata
    square = np.zeros((512, 512, 3), np.uint8)
    square[yOffset:yOffset + resized.shape[0], xOffset:xOffset + resized.shape[1]] = resized

    return square


# Funzione che carica le subclasses dal file
def get_subclass():
    subclasses = []
    f = open("subclasses.names", "r", encoding="UTF-8")
    cnt = 0
    for x in f:
        if x[0] != "#":
            subclasses.append([x.strip().split("-")[0], x.strip().split("-")[1].split(";")])
            cnt += 1
    f.close()
    return subclasses


# Funzione che carica i filter dal file
def get_classfilter():
    classfilter = []
    f = open("filter.names", "r", encoding="UTF-8")
    for x in f:
        if x[0] != "#":
            classfilter.append(x.strip())
    f.close()
    return classfilter


# Funzione che carica le quantità e le classi da scaricare
def get_classqnt():
    classes = np.asarray([])
    load_qnt = np.asarray([])
    f = open("classes.names", "r", encoding="UTF-8")
    for x in f:
        if x[0] != "#":
            classes = np.append(classes, x.strip().split("-")[0])
            load_qnt = np.append(load_qnt, int(x.strip().split("-")[1]))
    f.close()
    return classes, load_qnt


# Funzione che carico il dizionario con tutte le associazioni class:id(Predefinito) se inverted=False id:class
def get_dict(inverted=True):
    f = open("csv_folder/class-descriptions-boxable.csv", "r", encoding="UTF-8")
    reader = csv.reader(f)
    if inverted:
        dict_list = {rows[1]: rows[0] for rows in reader}
    else:
        dict_list = {rows[0]: rows[1] for rows in reader}
    return dict_list


# Funzione ottimizzata per fare le regex
def grep(query, current_mode):
    commandStr = "grep " + query + " ./csv_folder/" + current_mode + "-annotations-bbox.csv"
    finding = subprocess.run(commandStr.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    finding = finding.splitlines()
    return finding


# Funzione per scaricare i file
def download(runMode, image, directory):
    subprocess.run(['aws', 's3', '--no-sign-request', '--only-show-errors', 'cp', 's3://open-images-dataset/' + runMode + '/' + image + ".jpg", directory])


# Funzione che disegna le bounding box sulle immagini, fa il resize e formatta l'xml
def Xml_formatter(current_bbox, mode, image, classes):
    size = image.shape[:2]

    # Faccio il resize, mando la dimensione più grande per essere trasformata in 512
    condizione = size[0] < size[1]
    if condizione:
        image = image_resize(image, width=512)
    else:
        image = image_resize(image, height=512)

    dict_list = get_dict(inverted=False)

    # Mi copio l'immagine che verrà poi editata
    image2 = np.copy(image)

    # Formatto il file in modo carino
    lines = []
    lines.append("<annotation>")
    lines.append("  " + "<folder>" + mode + "</folder>")
    lines.append("  " + "<filename>" + current_bbox[0].split(",")[0] + ".jpg</filename>")
    lines.append("  " + "<path /><source>")
    lines.append("      " + "<database>Unknown</database>")
    lines.append("  " + "</source>")
    lines.append("  " + "<size>")
    lines.append("      " + "<width>512</width>")
    lines.append("      " + "<height>512</height>")
    lines.append("      " + "<depth>3</depth>")
    lines.append("  " + "</size>")
    lines.append("  " + "<segmented>0</segmented>")
    # Itero per le varie bbox
    for line in current_bbox:
        lineParts = line.split(',')
        lines.append("  " + "<object>")

        # Mi salvo la classe per il colore
        current_class = dict_list[lineParts[2]]

        lines.append("      " + "<name>" + current_class + "</name>")
        lines.append("      " + "<pose>Unspecified</pose>")
        lines.append("      " + "<truncated>" + lineParts[9] + "</truncated>")
        lines.append("      " + "<difficult>0</difficult>")
        lines.append("      " + "<bndbox>")

        # Mi calcolo le nuove bbox
        if condizione:
            yRidimensionata = 512 / size[1] * size[0]
            yOffset = int((512 - yRidimensionata) / 2)
            xmin = int(float(lineParts[4]) * 512)
            xmax = int(float(lineParts[5]) * 512)
            ymin = int(float(lineParts[6]) * yRidimensionata + yOffset)
            ymax = int(float(lineParts[7]) * yRidimensionata + yOffset)
        else:
            xRidimensionata = 512 / size[0] * size[1]
            xOffset = int((512 - xRidimensionata) / 2)
            xmin = int(float(lineParts[4]) * xRidimensionata + xOffset)
            xmax = int(float(lineParts[5]) * xRidimensionata + xOffset)
            ymin = int(float(lineParts[6]) * 512)
            ymax = int(float(lineParts[7]) * 512)

        lines.append("          " + "<xmin>" + str(xmin) + "</xmin>")
        lines.append("          " + "<ymin>" + str(ymin) + "</ymin>")
        lines.append("          " + "<xmax>" + str(xmax) + "</xmax>")
        lines.append("          " + "<ymax>" + str(ymax) + "</ymax>")

        lines.append("      " + "</bndbox>")
        lines.append("  " + "</object>")

        # Scelgo il colore a seconda della classe
        if current_class == classes[0]:
            color = (255, 0, 0)
        if len(classes) > 1:
            if current_class == classes[1]:
                color = (0, 255, 0)
        if len(classes) > 2:
            if current_class == classes[2]:
                color = (0, 0, 255)
        if len(classes) > 3:
            if current_class == classes[3]:
                color = (255, 255, 0)
        if len(classes) > 4:
            if current_class == classes[4]:
                color = (0, 255, 255)
        if len(classes) > 5:
            if current_class == classes[5]:
                color = (255, 0, 255)
        if len(classes) > 6:
            color = (255, 255, 255)
            print("attenzione potrebbero esserci più classi colorate allo stesso modo")
        # Aggiungo la bbox alla foto
        image2 = cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), color, 2)

    lines.append("</annotation>")
    return lines, image, image2
