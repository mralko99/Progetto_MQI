import cv2
import numpy as np
import random as rnd
from shutil import rmtree
from os import mkdir, makedirs, listdir
from os.path import join, isfile, isdir
from function import get_subclass, get_classfilter, get_classqnt, grep, get_dict, download, xml_generator, image_resize, box_drawer

# VARIABILI GLOBALI
Modes = ["train", "validation"]     # Imposto le modes che andrà a scaricare
Dataset = "Dataset/"                # Imposto la directory dove scaricare le foto per la review
Final_Size = 512                    # Imposto le dimensioni finali dell'immagine di output
validation_ratio = 0.25             # Imposto la percentuale di dati scaricati di validation rispetto a train
Dataset_filter = "BigData/"         # Imposto la directory dove andare a cercare i file già scaricati
classes, load_qnt = get_classqnt()  # Carico le classi e le quantità associate
class_filter = get_classfilter()    # Carico le classi che verranno filtrate dal download
subclasses = get_subclass()         # Carico se ci sono le classi con le loro sottoclassi
dict_list = get_dict()               # Carico il dizionario con tutte le associazioni class:id
rnd.seed(80)                        # Imposto il seed
Filtri = [True, True, True]


# REINIZZIALIZZO LA CARTELLA DATASET
# Cancello la cartella del dataset
if isdir(Dataset):
    rmtree(Dataset, ignore_errors=True)
# Creo di nuovo la cartella del dataset che adesso sarà vuota
if not isdir(Dataset):
    mkdir(Dataset)
# #

# Itero tutto il codice per le modes
for runMode in Modes:

    print('-Scarichiamo il', runMode, 'set')

    print('   Carico le immagini che ho già scaricato')
    # CARICO TUTTE LE FOTO CHE HO GIA' SCARICATO PER NON RISCARICARLE
    # Mi scrivo il path su cui devo cercare le foto già scaricate
    downloaded_path = Dataset_filter + runMode + "/annotation/"
    # Se esiste carico le immagini già scaricate
    if isdir(downloaded_path):
        # Mi scrivo tutti gli id nelle foto già scaricate sull'array downloaded
        downloaded = [f[:16] for f in listdir(downloaded_path) if isfile(join(downloaded_path, f))]
        # Trasformo l'array downloaded in un np.array
        npFile = np.array(downloaded, dtype=str)
    # Sennò creo un array vuoto in quanto ci devo fare delle operazioni
    else:
        npFile = np.asarray([], dtype=str)
    # #

# CARICO TUTTE LE FOTO IN CUI SONO PRESENTI FILTER CLASS E NE FACCIO UN NP-ARRAY
    # Creo un array vuoto
    filter_id = []

    if Filtri[0]:
        print('   Carico le classi di filtro')
        # Itero su le classi di filtro
        for x in range(len(class_filter)):
            full_progbar = 10
            filled_progbar = round(x / len(class_filter) * full_progbar)
            print('\r       ', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), 'Classi:', x + 1, '/', len(class_filter), end="", flush=True)
            # Faccio la grep (trovo le righe contenenti ...) con il nome della classe filtro
            filter_bbox = grep(dict_list[class_filter[x]], runMode)

            # Itero sulle linee che ho trovato
            for line in filter_bbox:
                filter_id.append(line[:16])

    print()
    if Filtri[1]:
        print('   Carico tutte le immagini che sono depiction')
        # Faccio la grep (trovo le righe contenenti ...) con il nome della classe filtro
        filter_bbox = grep("[0-1],[0-1],[0-1],1,[0-1]", runMode)
        # Itero sulle linee che ho trovato
        for line in filter_bbox:
            filter_id.append(line[:16])

    if Filtri[2]:
        print('   Carico tutte le immagini che sono group')
        # Faccio la grep (trovo le righe contenenti ...) con il nome della classe filtro
        filter_bbox = grep("[0-1],[0-1],1,[0-1],[0-1]", runMode)
        # Itero sulle linee che ho trovato
        for line in filter_bbox:
            filter_id.append(line[:16])

    # Trasformo tutto in un np_array
    npFilter = np.array(filter_id, dtype=str)
    # Interseco il np_array con se stesso per evitare le ripetizioni
    npFilter = np.intersect1d(npFilter, npFilter)

    

# INIZIALIZZO LE QUANTITA DI DOWNLOAD PER LE DIVERSE MODALITA'
    # Ricarico ogni volta le quantità dall'array del file
    qnt = np.asarray(load_qnt)
    # Se è validation lo riduco per il validation_ratio
    if runMode == 'validation':
        qnt = qnt * validation_ratio
    # Lo trasformo in intero per evitare quantità non tonde per il download
    qnt = qnt.astype(int)
# #

    # Itero questa parte di codice per tutte le classi di interesse
    for ind in range(len(classes)):

        # Creo delle variabili locali per richiamarle più facilmente
        curr_class = classes[ind]
        curr_qnt = int(qnt[ind])

# CREO LE DIRECTORY DI LAVORO
        # Definisco le directory
        directory_class = Dataset + curr_class + '/'
        directory_image = directory_class + runMode + '/image/'
        directory_annotation = directory_class + runMode + '/annotation/'
        directory_review = directory_class + runMode + '/review/'

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
        # Faccio la grep (trovo le righe contenenti ...) con il nome della classe filtro
        image_bbox = grep(dict_list[curr_class], runMode)
        # Itero sulle linee che ho trovato
        for line in image_bbox:
            lineParts = line.split(',')
            Id_immagine = lineParts[0]
            # Se filter id è vuoto ci carico il primo elemento
            if len(image_id) == 0:
                image_id.append(Id_immagine)
            # Altrimenti dopo aver controllato che non è presente nell'ultima posizione lo aggiungo
            elif Id_immagine != image_id[len(image_id) - 1]:
                image_id.append(Id_immagine)

        # Trasformo tutto in un np_array
        npImage = np.array(image_id, dtype=str)
        # Elimino i duplicati in npImage
        npImage = np.intersect1d(npImage, npImage)
        # Elimino i file già scaricati e revisionati
        npImage = np.setdiff1d(npImage, npFile)
        # Elimino i file già scaricati e revisionati
        npImage = np.setdiff1d(npImage, npFilter)

        # Faccio lo shuffle delle risorse
        # rnd.shuffle(npImage)
# #
        # Printo quante immagini stiamo scaricando
        print("     Stiamo scaricando", str(curr_qnt), "immagini su", str(len(npImage)), "della classe", curr_class)

        # Riduco l'array delle immagini che devo scaricare
        npImage = npImage[0:curr_qnt]

        npFile = np.union1d(npFile, npImage)

        # Creo una variabile d'appoggio che mi serve per fare le print di quanti elementi ho scaricato
        cnt = 0

        # Stampo il cnt a 0%
        print('      ', '-' * 10, "%.2f" % float(0), "%", end="")

        # Itero sull'array finale di codici di immagini filtrate
        for im in npImage:
            # Incremento il counter
            cnt += 1

# FACCIO LA PRINT IN MODO ADEGUATO PER CAPIRE A CHE PUNTO SONO
            full_progbar = 10
            filled_progbar = round(cnt / curr_qnt * full_progbar)
            print('\r       ', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), "%.2f" % float(cnt / curr_qnt * 100), "%", end="", flush=True)
# #
            # Faccio la grep (trovo le righe contenenti ...) con il nome della current_foto
            current_bbox = grep(im, runMode)

            # Creo una variabile vuota per memorizzare le bbox
            current_bbox2 = []

# SCARTO DALLE CURRENT_BBOX QUELLE NON DI INTERESSE
            # Itero sul numero di current_bbox
            for i in range(len(current_bbox)):
                # Apro la linea per sapere che classe è
                lineParts = current_bbox[i].split(',')

                condizione = True

                # Itero sulle classi di interesse
                for intrest in classes:
                    # Se è una classe di interesse la copio
                    if lineParts[2] == dict_list[intrest]:
                        condizione = False
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

            # Sovrascrivo le current_bbox con la versione di solo interesse
            current_bbox = current_bbox2
# #

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
