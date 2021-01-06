import tkinter as tk
import os

import click


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
        os.system(
            "ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 -o results/ects/" + item + " ects")


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
        os.system(
            "ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 --cplus -o results/edsc/" + item + " edsccplus")


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
        os.system(
            "ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 --java -o results/teaser/" + item + " teaser -s 20")


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
        os.system(
            "ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 --java -o results/ecec/" + item + " ecec")


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
        os.system(
            "ets -t data/UCRArchive_2018/" + item + "/" + item + "_TRAIN.tsv -e data/UCRArchive_2018/" + item + "/" + item + "_TEST.tsv -s '\\t' -d 0 -c -1 -g normal -o results/mlstm/" + item + " mlstm")


@click.command()
@click.option('-a', '--algorithm', type=click.Choice(['ects', 'edsc', 'teaser', 'ecec', 'mlstm'], case_sensitive=False),
              help='Chosen algorithm.')
def run(algorithm):
    if algorithm == "ects":
        ects()
    elif algorithm == "edsc":
        edsc()
    elif algorithm == "teaser":
        teaser()
    elif algorithm == "ecec":
        ecec()
    elif algorithm == "mlstm":
        mlstm()


if __name__ == '__main__':
    path = "results/"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    run()
