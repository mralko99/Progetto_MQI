import sys
from cv2 import imread, imwrite
from os import listdir, remove
from os.path import isfile, join
from function import Xml_formatter, get_dict, get_subclass, grep, download


dict_list = get_dict(inverted=False)  # Carico il dizionario con tutte le associazioni id:class
dict_list2 = get_dict()               # Carico il dizionario con tutte le associazioni class:id
subclasses = get_subclass()           # Carico se ci sono le classi con le loro sottoclassi
change_label = True                  # Decide se cambiare le label delle sottoclassi

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
    for i in current_classes:
        print(i)

    # Scarico l'immagine richiesta
    download(current_mode, photo_id, "./" + photo_id + ".jpg")

    # Carico la foto da file
    image = imread(photo_id + ".jpg")

    # La passo a Xml_formatter
    xml, image, image2 = Xml_formatter(current_bbox, current_mode, image, current_classes)

    # Salvo la foto in stile review
    imwrite(photo_id + ".jpg", image2)
