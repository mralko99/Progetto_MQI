from os import listdir, makedirs
from os.path import isfile, isdir, join
from shutil import copyfile
from scandir import scandir

Modes = ["train", "validation"]
Dataset_origine = "Dataset2/"
Dataset_destinazione = "BigData/"

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
    for current_mode in Modes:
        mypath = Dataset_origine + current_class + "/" + current_mode + "/review/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for photo in onlyfiles:
            id = photo[:16]
            copyfile(Dataset_origine + current_class + "/" + current_mode + "/annotation/" + id + ".xml", Dataset_destinazione + current_mode + "/annotation/" + id + ".xml")
            copyfile(Dataset_origine + current_class + "/" + current_mode + "/image/" + id + ".jpg", Dataset_destinazione + current_mode + "/image/" + id + ".jpg")
