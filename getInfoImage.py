import sys
from cv2 import imread
from os import listdir, remove
from os.path import isfile, join
from function import get_dict, get_subclass, grep, download, image_resize, box_drawer


dict_list = get_dict(inverted=False)  # Carico il dizionario con tutte le associazioni id:class
dict_list2 = get_dict()               # Carico il dizionario con tutte le associazioni class:id
subclasses = get_subclass()           # Carico se ci sono le classi con le loro sottoclassi
change_label = False                  # Decide se cambiare le label delle sottoclassi
Final_Size = 512

modes = ["train", "validation", "test"]
current_mode = ""

# Elimino, se sono presenti immagini .jpg
onlyfiles = [f for f in listdir("./") if isfile(join("./", f))]
for x in onlyfiles:
    if x[len(x) - 3:] == "jpg":
        remove(x)
# Se l'utente non dà un input valido al comando
Foto = sys.argv
if len(Foto) > 2 or len(Foto) == 1 or len(Foto[1].strip()) != 16:
    print("Errore nell'utilizzo del comando")
else:
    photo_id = Foto[1].strip()
    # Individuo la modalita e mi salvo il risultato della grep
    for iter_mode in modes:
        current_bbox = grep(photo_id, iter_mode)
        if len(current_bbox) != 0:
            current_mode = iter_mode
            break

#    for i in current_bbox:
#        print(i)

    # Mi salvo le classi delle immagini per poi stamparle
    current_classes = []
    for i in range(len(current_bbox)):
        lineParts = current_bbox[i].split(',')
        iter_class = dict_list[lineParts[2]]
        # Se attivo la modifica delle sottoclassi
        if change_label:
            for x in subclasses:
                for intrest in x[1]:
                    if lineParts[2] == dict_list2[intrest]:
                        # Modifico il nome della classe
                        iter_class = x[0]
                        # Modifico la riga della bbox
                        modified_bbox = [lineParts[0], lineParts[1], dict_list2[x[0]]]
                        for modified in lineParts[3:]:
                            modified_bbox.append(modified)
                        current_bbox[i] = ",".join(modified_bbox)
        # Mi salvo il nome della classe se è nuovo
        if iter_class not in current_classes:
            current_classes.append(iter_class)

    # Stampo le classi presenti
#    for i in current_classes:
#        print(i)

    # Scarico l'immagine richiesta
    download(photo_id, "./")

    # Carico la foto da file
    image = imread(photo_id + ".jpg")

    size = image.shape[:2]

    image = image_resize(image, Final_Size)

    info = box_drawer(image, Final_Size, size, current_bbox, current_classes, [], ".", photo_id, True)
    # [current_class, percentage, xmin, xmax, ymin, ymax, color_name, lineParts[8:13]]
    max_0 = 0
    max_1 = 0
    for y in info:
        if len(y[0]) > max_0:
            max_0 = len(y[0])
        if len(y[3]) > max_1:
            max_1 = len(y[3])
    for y in info:
        print("Classe:", y[0] + " " * (max_0 - len(y[0])), "  Colore:", y[3] + " " * (max_1 - len(y[3])), "  Percentuale:", str("%.4f" % y[1]), sep="", end="")

        print("  BBox:[", sep="", end="")
        for x in range(len(y[2])):
            print("0" * (3 - len(str(y[2][x]))), sep="", end="")
            if x != len(y[2]) - 1:
                print(y[2][x], sep="", end=",")
            else:
                print(y[2][x], sep="", end="")

        print("]  Data:[", sep="", end="")
        for x in range(len(y[4])):
            if x != len(y[4]) - 1:
                print(y[4][x], sep="", end=",")
            else:
                print(y[4][x], sep="", end="")
        print("]")
