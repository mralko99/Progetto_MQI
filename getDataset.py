import sys
from os import listdir
from os.path import isfile, join
from shutil import copyfile

Modes = ["train", "validation"]
Dataset = "Dataset2/"

# Carico i dataset a cui sono interessato e le quantità di immagini che voglio scaricare
f = open("classes.names", "r", encoding="UTF-8")
classes = []
for x in f:
    if x[0] != "#":
        classes.append(x.strip().split("-")[0])
f.close()

for current_class in classes:
    for current_mode in Modes:
        mypath = Dataset + current_class + "/" + current_mode + "/review/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for photo in onlyfiles:
            id = photo[:16]
            copyfile(Dataset + current_class + "/" + current_mode + "/annotation/" + id + "xml", Dataset + current_mode + "/annotation/" + id + "xml")
            copyfile(Dataset + current_class + "/" + current_mode + "/image/" + id + "jpg", Dataset + current_mode + "/image/" + id + "jpg")
