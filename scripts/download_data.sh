#!/usr/bin/env bash

DATA_FOLDER="../data"
BIO_REPO="https://owncloud.skel.iit.demokritos.gr/index.php/s/5o0E7MZdaabFpjX/download"
UCR_REPO="https://owncloud.skel.iit.demokritos.gr/index.php/s/IVWeJ0UsdPbjhhU/download"
MAR_REPO="https://owncloud.skel.iit.demokritos.gr/index.php/s/Yp6QWTA9LZGEOGY/download"

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
    wget --progress=bar:force  --content-disposition -P ${DATA_FOLDER} ${UCR_REPO}

    # Unzip the data
    unzip "${DATA_FOLDER}/UCR_UEA.zip" -d ${DATA_FOLDER}

    # Remove the archive
    rm "${DATA_FOLDER}/UCR_UEA.zip"


    wget --progress=bar:force --content-disposition -P ${DATA_FOLDER} ${MAR_REPO}

    # Unzip the data
    unzip "${DATA_FOLDER}/Maritime_data.zip" -d ${DATA_FOLDER}

    # Remove the archive
    rm "${DATA_FOLDER}/Maritime_data.zip"
fi
