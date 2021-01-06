import tkinter as tk
import os

def ects():
    arr = os.listdir('data/UCRArchive_2018')
    path = "results/ects"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    for item in arr:
        os.system("ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 -o results/ects/" + item + " ects")

def edsc():
    arr = os.listdir('data/UCRArchive_2018')
    path = "results/edsc"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    for item in arr:
        os.system("ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 --cplus -o results/edsc/" + item + " edsccplus")

def teaser():
    arr = os.listdir('data/UCRArchive_2018')
    path = "results/teaser"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    for item in arr:
        os.system("ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 --java -o results/teaser/" + item + " teaser -s 20")

def ecec():
    arr = os.listdir('data/UCRArchive_2018')
    path = "results/ecec"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    for item in arr:
        os.system("ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 --java -o results/ecec/" + item + " ecec")


def mlstm():
    arr = os.listdir('data/UCRArchive_2018')
    path = "results/mlstm"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    for item in arr:
        os.system("ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 -g normal -o results/mlstm/" + item + " mlstm")

path = "results/"
try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)
root = tk.Tk()
root.title("ETS Algorithms")

button = tk.Button(root, text='ECTS', width=25, command=ects)
button.pack()


button = tk.Button(root, text='EDSC', width=25, command=edsc)
button.pack()

button = tk.Button(root, text='Teaser', width=25, command=teaser)
button.pack()

button = tk.Button(root, text='ECEC', width=25, command=ecec)
button.pack()

button = tk.Button(root, text='MLSTM', width=25, command=mlstm)
button.pack()

button = tk.Button(root, text='Stop', width=25, command=root.destroy)
button.pack()

root.mainloop()
