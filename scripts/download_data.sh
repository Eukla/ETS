#!/usr/bin/env bash

DATA_FOLDER="../data"
BIO_REPO="https://owncloud.skel.iit.demokritos.gr/index.php/s/pBsULX86ttrF7YJ/download"
UCR_REPO="https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip"
MAR_REPO="https://owncloud.skel.iit.demokritos.gr/index.php/s/ol1UzrLCLhP6vKL/download"
UCR_MISSING_VALUES_FOLDER="Missing_value_and_variable_length_datasets_adjusted"
UCR_PASSWORD="someone"

if [[ ! -d "${DATA_FOLDER}" ]]; then

    # Create the data folder
    mkdir ${DATA_FOLDER}


    # Download the BIO archive
    wget --progress=bar:force --content-disposition -P ${DATA_FOLDER} ${BIO_REPO}

    # Unzip the data
    unzip "${DATA_FOLDER}/BioArchive.zip" -d ${DATA_FOLDER}

    # Remove the archive
    rm "${DATA_FOLDER}/BioArchive.zip"


    # Download the UCR archive
    wget --progress=bar:force -P ${DATA_FOLDER} ${UCR_REPO}

    # Unzip the data
    unzip -P ${UCR_PASSWORD} "${DATA_FOLDER}/UCRArchive_2018.zip" -d ${DATA_FOLDER}

    # Remove the archive
    rm "${DATA_FOLDER}/UCRArchive_2018.zip"

    # Remove data having missing values or variable length time-series
    for dataset in $(find "${DATA_FOLDER}/UCRArchive_2018/${UCR_MISSING_VALUES_FOLDER}" -type d -depth 1); do
        dataset_name=$(basename "${dataset}")
        rm -r "${DATA_FOLDER}/UCRArchive_2018/${dataset_name}"
        mv "${dataset}" "${DATA_FOLDER}/UCRArchive_2018/"
    done
    rm -r "${DATA_FOLDER}/UCRArchive_2018/${UCR_MISSING_VALUES_FOLDER}"

    wget --progress=bar:force --content-disposition -P ${DATA_FOLDER} ${MAR_REPO}

    # Unzip the data
    unzip "${DATA_FOLDER}/Maritime.zip" -d ${DATA_FOLDER}

    # Remove the archive
    rm "${DATA_FOLDER}/Maritime.zip"
fi
