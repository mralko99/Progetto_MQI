import cv2
import numpy as np
import random as rnd
from shutil import rmtree
from os import mkdir, makedirs, listdir
from os.path import join, isfile, isdir
from function import get_subclass, get_classfilter, get_classqnt, grep, get_dict, download, xml_generator, image_resize, box_drawer

# VARIABILI GLOBALI
modes = ["train", "validation", "test"]     # Imposto le modes che andrà a scaricare
Dataset = "Dataset/"                # Imposto la directory dove scaricare le foto per la review
Final_Size = 512                    # Imposto le dimensioni finali dell'immagine di output
validation_ratio = 0.25             # Imposto la percentuale di dati scaricati di validation rispetto a train
Dataset_filter = "BigData/"         # Imposto la directory dove andare a cercare i file già scaricati
classes, qnt = get_classqnt()  # Carico le classi e le quantità associate
class_filter = get_classfilter()    # Carico le classi che verranno filtrate dal download
subclasses = get_subclass()         # Carico se ci sono le classi con le loro sottoclassi
dict_list = get_dict()               # Carico il dizionario con tutte le associazioni class:id
rnd.seed(80)                        # Imposto il seed

# 0 Già Scaricate, 1 Classi filtro, 2 isGroup, 3 isDepiction, 4 Sottoclassi
Filtri = [True, True, True, True, True]


# Cancello la cartella del dataset
if isdir(Dataset):
    rmtree(Dataset, ignore_errors=True)
# Creo di nuovo la cartella del dataset che adesso sarà vuota
if not isdir(Dataset):
    mkdir(Dataset)
# #


# Carico tutti i filtri
downloaded = []
filter_id = []
if Filtri[0] or Filtri[1]:
    print('-Carichiamo i filtri per le classi')
    for runMode in modes:

        print('  Nel', runMode, 'set')

        downloaded_path = Dataset_filter + runMode + "/annotation/"
        if isdir(downloaded_path) and Filtri[0]:
            print('   Carico le immagini che ho già scaricato')
            downloaded_temp = [f[:16] for f in listdir(downloaded_path) if isfile(join(downloaded_path, f))]
            downloaded.extend(downloaded_temp)

        if Filtri[1]:
            print('   Carico le classi di filtro')
            # Itero su le classi di filtro
            for x in range(len(class_filter)):
                full_progbar = 10
                filled_progbar = round((x + 1) / (len(class_filter)) * full_progbar)
                print('\r       ', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), 'Classi:', x + 1, '/', len(class_filter), end="", flush=True)
                # Faccio la grep (trovo le righe contenenti ...) con il nome della classe filtro
                filter_bbox = grep(dict_list[class_filter[x]], runMode)

                # Itero sulle linee che ho trovato
                for line in filter_bbox:
                    filter_id.append(line[:16])
            print()

npFile = np.asarray(downloaded, dtype=str)
# Trasformo tutto in un np_array
npFilter = np.array(filter_id, dtype=str)
# Interseco il np_array con se stesso per evitare le ripetizioni
npFilter = np.intersect1d(npFilter, npFilter)


# Itero questa parte di codice per tutte le classi di interesse
for ind in range(len(classes)):

    # Creo delle variabili locali per richiamarle più facilmente
    curr_class = classes[ind]
    curr_qnt = int(qnt[ind])

# CREO LE DIRECTORY DI LAVORO
    # Definisco le directory
    directory_class = Dataset + curr_class + '/'
    directory_image = directory_class + '/image/'
    directory_annotation = directory_class + '/annotation/'
    directory_review = directory_class + '/review/'

    # Creo le directory mancanti
    if not isdir(directory_image):
        makedirs(directory_image)
    if not isdir(directory_annotation):
        makedirs(directory_annotation)
    if not isdir(directory_review):
        makedirs(directory_review)
# #

    # MI CALCOLO GLI ID DELLE FOTO DELLE CLASSI DI INTERESSE
    # Creo un array vuoto
    image_id = []
    print('-Caricho le immagini della classe', curr_class, 'e della sua sottoclasse')
    for runMode in modes:
        print('  Nel', runMode, 'set')
        print('   Carico dalla classe')
        # Su classe corrente
        if not Filtri[2] and not Filtri[3]:
            regex = dict_list[curr_class]
        # Se non deve essere group
        if Filtri[2] and not Filtri[3]:
            regex = dict_list[curr_class] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],0,[0-1],[0-1]"
        # Se non deve essere depiction
        if not Filtri[2] and Filtri[3]:
            regex = dict_list[curr_class] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],[0-1],0,[0-1]"
        # Se non deve essere ne group ne depiction
        if Filtri[2] and Filtri[3]:
            regex = dict_list[curr_class] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],0,0,[0-1]"

        image_bbox = grep(regex, runMode)
        for line in image_bbox:
            image_id.append(line[:16])

        if Filtri[4]:
            # Su sottoclasse di classe corrente
            for x in subclasses:
                if x[0] == curr_class:
                    print('   Carico dalle sottoclassi')
                    # Itero le sottoclassi di una classe di interesse
                    for y in range(len(x[1])):
                        intrest = x[1][y]
                        full_progbar = 10
                        filled_progbar = round((y + 1) / (len(x[1])) * full_progbar)
                        print('\r       ', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), 'Sottoclassi:', y + 1, '/', len(x[1]), end="", flush=True)
                        if not Filtri[2] and not Filtri[3]:
                            regex = dict_list[intrest]
                        # Se non deve essere group
                        if Filtri[2] and not Filtri[3]:
                            regex = dict_list[intrest] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],0,[0-1],[0-1]"
                        # Se non deve essere depiction
                        if not Filtri[2] and Filtri[3]:
                            regex = dict_list[intrest] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],[0-1],0,[0-1]"
                        # Se non deve essere ne group ne depiction
                        if Filtri[2] and Filtri[3]:
                            regex = dict_list[intrest] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],0,0,[0-1]"
                        image_bbox = grep(regex, runMode)
                        for line in image_bbox:
                            image_id.append(line[:16])
            print()

    # Trasformo tutto in un np_array
    npImage = np.array(image_id, dtype=str)
    # Elimino i duplicati in npImage
    npImage = np.intersect1d(npImage, npImage)

    # Elimino le immagini già scaricate e revisionate
    npImage = np.setdiff1d(npImage, npFile)
    # Elimino le immagini filtrate
    npImage = np.setdiff1d(npImage, npFilter)

    print()

    if curr_qnt > len(npImage):
        # Printo quante immagini stiamo scaricando
        print("-Stiamo scaricando tutte le ", str(len(npImage)), " immagini della classe", curr_class, "anche se ne erano state richieste", str(curr_qnt))
        curr_qnt = len(npImage)

    else:
        # Printo quante immagini stiamo scaricando
        print("-Stiamo scaricando", str(curr_qnt), "immagini su", str(len(npImage)), "della classe", curr_class)

    # Faccio lo shuffle delle risorse
    # rnd.shuffle(npImage)

# #### RIDUCO L'ARRAY E SCARICO

    # Riduco l'array delle immagini che devo scaricare
    npImage = npImage[0:curr_qnt]

    npFile = np.union1d(npFile, npImage)

    # Itero sull'array finale di codici di immagini filtrate
    for cnt in range(len(npImage)):

        # Incremento il counter
        im = npImage[cnt]

        for iter_mode in modes:
            current_bbox = grep(im, iter_mode)
            if len(current_bbox) != 0:
                runMode = iter_mode
                break

        # Print progress bar delle foto che sto scaricando
        full_progbar = 10
        filled_progbar = round((cnt + 1) / (curr_qnt - 1) * full_progbar)
        print('\r   ', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), "%.2f" % float((cnt + 1) / (curr_qnt) * 100), "%", end="", flush=True)

        # Faccio la grep (trovo le righe contenenti ...) con il nome della current_foto
        current_bbox = grep(im, runMode)

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
            if Filtri[4]:
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

        # Sovrascrivo le current_bbox con la versione di solo interesse
        current_bbox = current_bbox2

        # Scarico l'immagine richiesta
        download(runMode, im, directory_image + '/' + im + ".jpg")

        # Leggo l'immagine che ho appena scaricato
        image = cv2.imread(directory_image + '/' + im + ".jpg")

        size = image.shape[:2]

        xml_generator(runMode, Final_Size, size, current_bbox, directory_annotation, im, True)

        image = image_resize(image, Final_Size, directory_image, im, True)

        box_drawer(image, Final_Size, size, current_bbox, classes, directory_review, im, True)

    print()

print()
