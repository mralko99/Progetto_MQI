from __future__ import division
import os
import sys
import function
import multiprocessing
import numpy as np
import random as rnd
from shutil import rmtree
from functools import partial


# VARIABILI GLOBALI
modes = ["train", "validation", "test"]     # Imposto le modes che andrà a scaricare
Final_Size = 512                    # Imposto le dimensioni finali dell'immagine di output
Dataset_filter = "BigData/"         # Imposto la directory dove andare a cercare i file già scaricati
classes, qnt = function.get_classqnt()  # Carico le classi e le quantità associate
class_filter = function.get_classfilter()    # Carico le classi che verranno filtrate dal download
subclasses = function.get_subclass()         # Carico se ci sono le classi con le loro sottoclassi
dict_list = function.get_dict()               # Carico il dizionario con tutte le associazioni class:id
Dataset, Filtri = function.get_settings()                # Imposto la directory dove scaricare le foto per la review
rnd.seed(50000)                        # Imposto il seed


# Cancello la cartella del dataset
if os.path.isdir(Dataset):
    rmtree(Dataset, ignore_errors=True)
# Creo di nuovo la cartella del dataset che adesso sarà vuota
if not os.path.isdir(Dataset):
    os.mkdir(Dataset)
# #


# Carico tutti i filtri
downloaded = []
filter_id = []

if Filtri[0]:
    print('-Carico le immagini gia scaricate', flush=True)
    for runMode in modes:
        downloaded_path = Dataset_filter + runMode + "/annotation/"
        if os.path.isdir(downloaded_path) and Filtri[0]:
            [downloaded.append(f[:16]) for f in os.listdir(downloaded_path) if os.path.isfile(os.path.join(downloaded_path, f))]
npFile = np.asarray(downloaded, dtype=str)

if Filtri[1]:
    print('-Carico i vari id delle foto appartenenti alle classi di filtro', flush=True)
    argument = []
    for x in class_filter:
        for y in modes:
            argument.append([dict_list[x], y])
    filter_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    filter_bbox = filter_pool.map(function.regex_map, argument)
    [filter_id.append(y) for x in filter_bbox for y in x]
    filter_pool.close()
    filter_pool.join()
if Filtri[0] or Filtri[1]:
    print()
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
    directory_download = directory_class + '/download/'

    # Creo le directory mancanti
    if not os.path.isdir(directory_image):
        os.makedirs(directory_image)
    if not os.path.isdir(directory_annotation):
        os.makedirs(directory_annotation)
    if not os.path.isdir(directory_review):
        os.makedirs(directory_review)
    if not os.path.isdir(directory_download):
        os.makedirs(directory_download)
# #

    # MI CALCOLO GLI ID DELLE FOTO DELLE CLASSI DI INTERESSE
    # Creo un array vuoto
    argument = []
    intrest = []

    intrest.append(curr_class)

    if Filtri[4]:
        print('-Carico le immagini della classe', curr_class, 'e della sue sottoclassi', flush=True)
        for x in subclasses:
            if x[0] == curr_class:
                for y in x[1]:
                    intrest.append(y)
    else:
        print('-Carico le immagini della classe', curr_class, flush=True)

    for x in intrest:
        for y in modes:
            if not Filtri[2] and not Filtri[3]:
                regex = dict_list[x]
            # Se non deve essere group
            if Filtri[2] and not Filtri[3]:
                regex = dict_list[x] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],0,[0-1],[0-1]"
            # Se non deve essere depiction
            if not Filtri[2] and Filtri[3]:
                regex = dict_list[x] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],[0-1],0,[0-1]"
            # Se non deve essere ne group ne depiction
            if Filtri[2] and Filtri[3]:
                regex = dict_list[x] + "," + "[0-9][.]*[0-9]*," * 5 + "[0-1],[0-1],0,0,[0-1]"

            argument.append([regex, y])

    image_id = []
    image_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    image_bbox = image_pool.map(function.regex_map, argument)
    [image_id.append(y) for x in image_bbox for y in x]
    image_pool.close()
    image_pool.join()

    # Trasformo tutto in un np_array
    npImage = np.array(image_id, dtype=str)
    # Elimino i duplicati in npImage
    npImage = np.intersect1d(npImage, npImage)

    # Elimino le immagini già scaricate e revisionate
    npImage = np.setdiff1d(npImage, npFile)
    # Elimino le immagini filtrate
    npImage = np.setdiff1d(npImage, npFilter)

    # Printo quante immagini stiamo scaricando
    if curr_qnt > len(npImage):
        print("-Stiamo scaricando tutte le ", str(len(npImage)), " immagini della classe", curr_class, "anche se ne erano state richieste", str(curr_qnt), flush=True)
        curr_qnt = len(npImage)
    else:
        print("-Stiamo scaricando", str(curr_qnt), "immagini su", str(len(npImage)), "della classe", curr_class, flush=True)

    # Faccio lo shuffle delle risorse
    # rnd.shuffle(npImage)

    npImage = npImage[0:curr_qnt]

    npFile = np.union1d(npFile, npImage)

    # Scarico le immagini
    download_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    download_init = partial(function.download,
                            directory=directory_download)
    for i, _ in enumerate(download_pool.imap(download_init, npImage), 1):
        sys.stderr.write('\r    Scaricati {:.2f}%'.format(float(i) / len(npImage) * 100))
    print()
    download_pool.map(download_init, npImage)
    download_pool.close()
    download_pool.join()

    # Processo le immagini
    processing_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    processing_init = partial(function.processing,
                              condizione=Filtri[4],
                              dict_list=dict_list,
                              classes=classes,
                              subclasses=subclasses,
                              directory_annotation=directory_annotation,
                              directory_image=directory_image,
                              directory_review=directory_review,
                              directory_download=directory_download,
                              Final_Size=Final_Size)
    for i, _ in enumerate(processing_pool.imap(processing_init, npImage), 1):
        sys.stderr.write('\r    Processati {:.2f}%'.format(float(i) / len(npImage) * 100))
    processing_pool.map(processing_init, npImage)
    processing_pool.close()
    processing_pool.join()

    print("\n    Cancello la cartella dei file scaricati", flush=True)
    if os.path.isdir(directory_download):
        rmtree(directory_download, ignore_errors=True)

    print()
