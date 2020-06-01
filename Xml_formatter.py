import csv
import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    square = np.zeros((512, 512, 3), np.uint8)

    condizione = h < w

    if condizione:
        yRidimensionata = 512 / w * h
        xOffset = 0
        yOffset = int((512 - yRidimensionata) / 2)
        square[yOffset:yOffset + resized.shape[0], xOffset:xOffset + resized.shape[1]] = resized
    else:
        xRidimensionata = 512 / h * w
        xOffset = int((512 - xRidimensionata) / 2)
        yOffset = 0
        square[yOffset:yOffset + resized.shape[0], xOffset:xOffset + resized.shape[1]] = resized

    # return the resized image
    return square


def Xml_formatter(current_annotations, mode, image, classes):

    size = image.shape[:2]

    condizione = size[0] < size[1]

    if condizione:
        image = image_resize(image, width=512)
    else:
        image = image_resize(image, height=512)

    # Crea un dizionario per ogni classe con il codice per il download
    with open('./csv_folder/class-descriptions-boxable.csv', mode='r') as infile:
        reader = csv.reader(infile)
        dict_list = {rows[0]: rows[1] for rows in reader}

    image2 = np.copy(image)

    lines = []
    lines.append("<annotation>")
    lines.append("  " + "<folder>" + mode + "</folder>")
    lines.append("  " + "<filename>" + current_annotations[0].split(",")[0] + ".jpg</filename>")
    lines.append("  " + "<path /><source>")
    lines.append("      " + "<database>Unknown</database>")
    lines.append("  " + "</source>")
    lines.append("  " + "<size>")
    lines.append("      " + "<width>512</width>")
    lines.append("      " + "<height>512</height>")
    lines.append("      " + "<depth>3</depth>")
    lines.append("  " + "</size>")
    lines.append("  " + "<segmented>0</segmented>")
    for line in current_annotations:
        lineParts = line.split(',')
        lines.append("  " + "<object>")
        lines.append("      " + "<name>" + dict_list[lineParts[2]] + "</name>")
        lines.append("      " + "<pose>Unspecified</pose>")
        lines.append("      " + "<truncated>" + lineParts[9] + "</truncated>")
        lines.append("      " + "<difficult>0</difficult>")
        lines.append("      " + "<bndbox>")

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

        if dict_list[lineParts[2]] == classes[0]:
            color = (255, 0, 0)
        if len(classes) > 1:
            if dict_list[lineParts[2]] == classes[1]:
                color = (0, 255, 0)
        if len(classes) > 2:
            if dict_list[lineParts[2]] == classes[2]:
                color = (0, 0, 255)
        if len(classes) > 3:
            if dict_list[lineParts[2]] == classes[3]:
                color = (255, 255, 0)

        image2 = cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), color, 2)

        lines.append("          " + "<xmin>" + str(xmin) + "</xmin>")
        lines.append("          " + "<ymin>" + str(ymin) + "</ymin>")
        lines.append("          " + "<xmax>" + str(xmax) + "</xmax>")
        lines.append("          " + "<ymax>" + str(ymax) + "</ymax>")

        lines.append("      " + "</bndbox>")
        lines.append("  " + "</object>")
    lines.append("</annotation>")
    return lines, image, image2
