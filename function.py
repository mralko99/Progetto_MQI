import cv2
import subprocess
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config

global s3
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def download(id, directory):
    modes = ["train", "validation", "test"]
    # Incremento il counter
    im = id

    for iter_mode in modes:
        current_bbox = grep(im, iter_mode)
        if len(current_bbox) != 0:
            curr_mode = iter_mode
            break
    # Scarico l'immagine richiesta
    s3.download_file('open-images-dataset', curr_mode + '/' + im + '.jpg', directory + '/' + im + ".jpg")


def processing(id, condizione, dict_list, classes, subclasses, directory_annotation, directory_image, directory_review, directory_download, Final_Size):
    modes = ["train", "validation", "test"]

    for iter_mode in modes:
        current_bbox = grep(id, iter_mode)
        if len(current_bbox) != 0:
            runMode = iter_mode
            break

    # Faccio la grep (trovo le righe contenenti ...) con il nome della current_foto
    current_bbox = grep(id, runMode)

    # Creo una variabile vuota per memorizzare le bbox
    current_bbox2 = []

    for i in range(len(current_bbox)):
        # Apro la linea per sapere che classe è
        lineParts = current_bbox[i].split(',')

        # Itero sulle classi di interesse
        for intrest in classes:
            # Se è una classe di interesse la copio
            if lineParts[2] == dict_list[intrest]:
                current_bbox2.append(current_bbox[i])
                break

        # Se non è scattata nessuna classe allora controllo le sottoclassi
        if condizione:
            # Itero sulle sottoclassi
            for x in subclasses:
                # Se una classe di interesse ha sottoclassi
                if x[0] in classes:
                    # Itero le sottoclassi di una classe di interesse
                    for intrest in x[1]:
                        # Se è presente
                        if lineParts[2] == dict_list[intrest]:
                            # Modifico la linea riscrivendo la classe di interesse al posto della sotto classe
                            modified_bbox = [lineParts[0], lineParts[1], dict_list[x[0]]]
                            for modified in lineParts[3:]:
                                modified_bbox.append(modified)
                            modified_bbox = ",".join(modified_bbox)
                            # La copio sulle righe che mi interessano
                            current_bbox2.append(modified_bbox)
                            break

    current_bbox = current_bbox2

    # Leggo l'immagine che ho appena scaricato
    image = cv2.imread(directory_download + '/' + id + ".jpg")

    size = image.shape[:2]

    xml_generator(runMode, Final_Size, size, current_bbox, directory_annotation, id, True)

    image = image_resize(image, Final_Size, directory_image, id, True)

    box_drawer(image, Final_Size, size, current_bbox, classes, directory_review, id, True)


def regex_map(class_mode):
    current_class = class_mode[0]
    current_mode = class_mode[1]
    current_id = []
    current_bbox = grep(current_class, current_mode)
    [current_id.append(line[:16]) for line in current_bbox]
    return current_id


def intrest(class_mode):
    current_class = class_mode[0]
    current_mode = class_mode[1]
    current_bbox = grep(current_class, current_mode)
    return current_bbox


# Funzione che carica le subclasses dal file
def get_subclass():
    subclasses = []
    f = open("./names/subclasses.names", "r", encoding="UTF-8")
    for x in f:
        if x[0] != "#":
            subclasses.append([x.strip().split("-")[0], x.strip().split("-")[1].split(";")])
    f.close()
    return subclasses


# Funzione che carica i filter dal file
def get_classfilter():
    classfilter = []
    f = open("./names/filter.names", "r", encoding="UTF-8")
    for x in f:
        if x[0] != "#":
            classfilter.append(x.strip())
    f.close()
    return classfilter


# Funzione che carica le quantità e le classi da scaricare
def get_classqnt():
    classes = np.asarray([])
    load_qnt = np.asarray([])
    f = open("./names/classes.names", "r", encoding="UTF-8")
    for x in f:
        if x[0] != "#":
            classes = np.append(classes, x.strip().split("-")[0])
            load_qnt = np.append(load_qnt, int(x.strip().split("-")[1]))
    f.close()
    return classes, load_qnt


# Funzione che carico il dizionario con tutte le associazioni class:id(Predefinito) se inverted=False id:class
def get_dict(inverted=True):
    f = open("./csv_folder/class-descriptions-boxable.csv", "r", encoding="UTF-8")
    dict_list = {}
    for x in f:
        if inverted:
            dict_list[x.strip().split(",")[1]] = x.strip().split(",")[0]
        else:
            dict_list[x.strip().split(",")[0]] = x.strip().split(",")[1]
    return dict_list


# Funzione che carica le subclasses dal file
def get_settings():
    f = open("./names/settings.names", "r", encoding="UTF-8")
    Dataset = ''
    Filtri = []
    Filtri2 = []
    for x in f:
        if x[0] != "#":
            if len(x.strip().split(",")) == 1:
                Dataset = x.strip()
            if len(x.strip().split(",")) == 5:
                [Filtri2.append(y) for y in x.strip().split(",")]
    f.close()
    for x in Filtri2:
        if x == '1':
            Filtri.append(True)
        else:
            Filtri.append(False)
    return Dataset, Filtri


# Funzione ottimizzata per fare le regex
def grep(query, current_mode):
    commandStr = "grep " + query + " ./csv_folder/" + current_mode + "-annotations-bbox.csv"
    finding = subprocess.run(commandStr.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    finding = finding.splitlines()
    return finding


def xml_generator(mode, dimension, original_size, current_bbox, directory=None, id=None, save=False):
    dict_list = get_dict(inverted=False)
    # Formatto il file in modo carino
    lines = []
    lines.append("<annotation>")
    lines.append("  " + "<folder>" + mode + "</folder>")
    lines.append("  " + "<filename>" + current_bbox[0].split(",")[0] + ".jpg</filename>")
    lines.append("  " + "<path /><source>")
    lines.append("      " + "<database>Unknown</database>")
    lines.append("  " + "</source>")
    lines.append("  " + "<size>")
    lines.append("      " + "<width>" + str(dimension) + "</width>")
    lines.append("      " + "<height>" + str(dimension) + "</height>")
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
        if original_size[0] < original_size[1]:
            yRidimensionata = dimension / original_size[1] * original_size[0]
            yOffset = int((dimension - yRidimensionata) / 2)
            xmin = int(float(lineParts[4]) * dimension)
            xmax = int(float(lineParts[5]) * dimension)
            ymin = int(float(lineParts[6]) * yRidimensionata + yOffset)
            ymax = int(float(lineParts[7]) * yRidimensionata + yOffset)
        else:
            xRidimensionata = dimension / original_size[0] * original_size[1]
            xOffset = int((dimension - xRidimensionata) / 2)
            xmin = int(float(lineParts[4]) * xRidimensionata + xOffset)
            xmax = int(float(lineParts[5]) * xRidimensionata + xOffset)
            ymin = int(float(lineParts[6]) * dimension)
            ymax = int(float(lineParts[7]) * dimension)

        lines.append("          " + "<xmin>" + str(xmin) + "</xmin>")
        lines.append("          " + "<ymin>" + str(ymin) + "</ymin>")
        lines.append("          " + "<xmax>" + str(xmax) + "</xmax>")
        lines.append("          " + "<ymax>" + str(ymax) + "</ymax>")

        lines.append("      " + "</bndbox>")
        lines.append("  " + "</object>")

    lines.append("</annotation>")

    if save:
        # Salvo il file xml
        g = open(directory + '/' + id + ".xml", "w", encoding="UTF-8")
        for x in lines:
            g.write(x)
            g.write("\n")
        g.close()


def image_resize(image, dimension, directory=None, id=None, save=False):

    # Mi calcolo altezza e larghezza dalla foto
    (h, w) = image.shape[:2]

    # Vedo quale delle due dimensioni devo modificare
    if h < w:
        yRidimensionata = dimension / w * h
        dim = (dimension, int(yRidimensionata))

        xOffset = 0
        yOffset = int((dimension - yRidimensionata) / 2)
    else:
        xRidimensionata = dimension / h * w
        dim = (int(xRidimensionata), dimension)

        xOffset = int((dimension - xRidimensionata) / 2)
        yOffset = 0

    # Faccio il primo resize alla dimensione più grande
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # La trasformo in un immagine quadrata
    square = np.zeros((dimension, dimension, 3), np.uint8)
    square[yOffset:yOffset + resized.shape[0], xOffset:xOffset + resized.shape[1]] = resized

    if save:
        cv2.imwrite(directory + '/' + id + ".jpg", square)

    return square


def box_drawer(image, dimension, original_size, current_bbox, classes, directory=None, id=None, save=False):
    dict_list = get_dict(inverted=False)
    info = []
    for line in current_bbox:
        lineParts = line.split(',')

        # Mi salvo la classe per il colore
        current_class = dict_list[lineParts[2]]

        # Mi calcolo le nuove bbox
        if original_size[0] < original_size[1]:
            xRidimensionata = dimension
            yRidimensionata = dimension / original_size[1] * original_size[0]
            yOffset = int((dimension - yRidimensionata) / 2)
            xmin = int(float(lineParts[4]) * dimension)
            xmax = int(float(lineParts[5]) * dimension)
            ymin = int(float(lineParts[6]) * yRidimensionata + yOffset)
            ymax = int(float(lineParts[7]) * yRidimensionata + yOffset)
        else:
            xRidimensionata = dimension / original_size[0] * original_size[1]
            yRidimensionata = dimension
            xOffset = int((dimension - xRidimensionata) / 2)
            xmin = int(float(lineParts[4]) * xRidimensionata + xOffset)
            xmax = int(float(lineParts[5]) * xRidimensionata + xOffset)
            ymin = int(float(lineParts[6]) * dimension)
            ymax = int(float(lineParts[7]) * dimension)

        color = (0, 0, 0)
        color_name = "Nero"

        # Scelgo il colore a seconda della classe
        if current_class == classes[0]:
            color = (255, 0, 0)
            color_name = "Blu"
        if len(classes) > 1:
            if current_class == classes[1]:
                color = (0, 255, 0)
                color_name = "Verde"
        if len(classes) > 2:
            if current_class == classes[2]:
                color = (0, 0, 255)
                color_name = "Rosso"
        if len(classes) > 3:
            if current_class == classes[3]:
                color = (255, 255, 0)
                color_name = "Ciano"
        if len(classes) > 4:
            if current_class == classes[4]:
                color = (0, 255, 255)
                color_name = "Giallo"
        if len(classes) > 5:
            if current_class == classes[5]:
                color = (255, 0, 255)
                color_name = "Magenta"
        if len(classes) > 6:
            color = (255, 255, 255)
            color_name = "Bianco"

        # Aggiungo la bbox alla foto
        percentage = (xmax - xmin) * (ymax - ymin) / (xRidimensionata * yRidimensionata)
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        if percentage < 0.02:
            if int(lineParts[8]) or int(lineParts[9]):
                color = (0, 0, 0)
            image = cv2.putText(image, str("%.3f" % percentage) + "%", (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.e-2 * (ymax - ymin), color, 2)

        info.append([current_class, percentage, [xmin, xmax, ymin, ymax], color_name, lineParts[8:13]])

    if save:
        cv2.imwrite(directory + '/' + id + ".jpg", image)

    return info
