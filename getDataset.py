from os import listdir, makedirs
from os.path import isfile, isdir, join
from shutil import copyfile
from scandir import scandir
import random as rnd

Modes = ["train", "validation"]
directory_dataset = []
dataset_reviewed = []

if not isdir("Verificato/"):
    makedirs("Verificato/")
if not isdir("Dataset/"):
    makedirs("Dataset/")

for directory in scandir("Verificato/"):
    if isdir("Verificato/" + directory.name) and "Verificato" in directory.name:
        dataset_reviewed.append(directory.name.split("_")[1])

for directory in scandir("Dataset/"):
    if isdir("Dataset/" + directory.name) and "Dataset" in directory.name and directory.name.split("_")[1] not in dataset_reviewed:
        directory_dataset.append(directory.name.split("_")[1])

for dirs in directory_dataset:
    Dataset_origine = "Dataset/Dataset_" + dirs + "/"
    Dataset_destinazione = "Verificato/Verificato_" + dirs + "/"

    # Carico le classi dalle cartelle nel Dataset_origine
    classes = []
    for entry in scandir(Dataset_origine):
        classes.append(entry.name)

    # Creo le directory per Dataset_destinazione image e annotation
    for current_mode in Modes:
        directory_image = Dataset_destinazione + current_mode + '/image/'
        directory_annotation = Dataset_destinazione + current_mode + '/annotation/'
        if not isdir(directory_image):
            makedirs(directory_image)
        if not isdir(directory_annotation):
            makedirs(directory_annotation)

    # Copio i file che sono presenti in tutte le review in image e annotation
    for current_class in classes:
        path = Dataset_origine + current_class + "/review/"
        nomi = [f for f in listdir(path) if isfile(join(path, f))]

        rnd.seed(80)
        rnd.shuffle(nomi)

        train = nomi[:int(len(nomi) * 0.8)]
        validation = nomi[int(len(nomi) * 0.8):]
        for photo in train:
            id = photo[:16]

            copyfile(Dataset_origine + current_class + "/annotation/" + id + ".xml",
                     Dataset_destinazione + "train/annotation/" + id + ".xml")

            copyfile(Dataset_origine + current_class + "/image/" + id + ".jpg",
                     Dataset_destinazione + "train/image/" + id + ".jpg")
        for photo in validation:
            id = photo[:16]

            copyfile(Dataset_origine + current_class + "/annotation/" + id + ".xml",
                     Dataset_destinazione + "validation/annotation/" + id + ".xml")

            copyfile(Dataset_origine + current_class + "/image/" + id + ".jpg",
                     Dataset_destinazione + "validation/image/" + id + ".jpg")
