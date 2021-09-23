# ETSC: Early Time Series Classification

`ETSC` is a Python Early Classification of Time-Series library for public use, from the work **"Evaluation of Early Time-Series Classification
Algorithms"**, **Authors: Evgenios Kladis, Charilaos Akasiadis, Evangelos Michelioudakis, Elias Alevizos, Alexander Artikis**.

Aim of this work is to study and collect algorithms that conduct early time-series classification, in a user-friendly format, for researchers to use for their work.

Currently six algorithms are included in this directory. A python cli, simplifies the execution of each algorithm
The predictions are evalueated through metrics such as earliness, accuracy, f1-score(if wanted) and computation time for both training and testing.

## Acknowledgments

Special thanks to Evangelos Michelioudakis (vagmcs@iit.demokritos.gr) for the contribution to the development of this repository.

## License

This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions; See the [GNU General Public License v3 for more details](http://www.gnu.org/licenses/gpl-3.0.en.html).

## Requirements

Python3 is required to install the libraries stated in the `requirements.txt`.

JVM >= 1.8 is required to run the algorithms that are implemented using java.

## Installation
1. Install the `virtualenv` package:

```bash
pip3 install virtualenv
```

2. Create a new virtual environment:

```bash
virtualenv venv
```

3. Activate virtual environment:

```
. venv/bin/activate
```

4. Install required packages:

```bash
pip3 install -r requirements.txt
```

5. Locally install `timeline`:

```bash
pip install --editable .
```

## Downloading the data

For downloading the data run the script `download_data.sh` found in the script folder. The downloaded data can be found inside folder `data`.
10 datasets are available, derived from the [UCR_UEA library](https://www.timeseriesclassification.com/). Multivariate datasets from the Biological and Maritime field are also provided.

## Experimental Setup

Note that only ECTS was implemented by us, using the paper of the algorithm as a guide. The rest of the algorithms derive from sources we provide in the following table. All credit goes to the original creators of the algorithms papers. 

| Algorithm | Parameters |
|---|---|
| ECTS [\[paper\]](https://link.springer.com/article/10.1007/s10115-011-0400-x) | support = 0 |
| EDSC [\[paper\]](https://epubs.siam.org/doi/10.1137/1.9781611972818.22) [\[code\]](https://drive.google.com/file/d/0BxY8OirJ0-gdbnBYNnRNbW9xeTQ/view) | CHE k=3, min_length=5, max_length=len(time_series)/2 |
| TEASER [\[paper\]](https://link.springer.com/article/10.1007/s10618-020-00690-z) [\[code\]](https://github.com/patrickzib/SFA) | S=20 (for the UCR_UEA), S=10 (for the biological and maritime) |
| ECEC [\[paper\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8765556) [\[code\]](https://github.com/junweilvhfut/ECEC)| training_times=20, length = len(time_series)/20,a=0.8 |
| MLSTM [\[paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200?via%3Dihub) [\[code\]](https://github.com/titu1994/MLSTM-FCN) | LSTM cells = [8, 64, 128], tested_lengths = [0.4,0.5,0.6] %  |
| ECONOMY-K [\[paper\]](https://link.springer.com/chapter/10.1007/978-3-319-23528-8_27) [\[code\]](https://tslearn.readthedocs.io/en/stable/user_guide/early.html) | k = [1, 2, 3], λ = 100, cost = 0.001  |
## Menu Guide

After running the <em> Virtual Enviroment </em> commands stated above, by running `ets` a menu with all programming options appears.
A running command is constructed as follows:

`ets <program commands> <algorithm> <algorithm commands>`

If you want to see the algorithm's menu run:

`ets <program commands> <algorithm> --help`

### Quick commands rundown used for the experiments

`-i <file path>` : Only one file is given for cross validation with a given number of folds.

`-t <file-path>` : The training file used. A `-e` command is also required.

`-e <file-path>` : The testing file used. A `-t` command is also required.

`-o <file-path>` : The desired output stream file. Default output steam is the console.

`-s <char>`: The seperator of each collumn in the file/s.

`-d` & `-h`: Commands that indicate the collumn of the classes in the input file/s. It can be either the `<int>` of the collumn for `-d` or the `<name>` for `-h`.

`-v <int>`: In case of multivariate input, describes the number of variables and should always be followed by `-g`. All Multivariate input files, each time-series, should take up `-v` consequent lines for each univariate time-series variable, bearing the same labels

`-g <method>`: The methods used to deal with multivariate time-series. We used `vote` which conducts the voting as explained in the paper and `normal` which passes the whole multivariate input in the algorithm, currently possible only by MLSTM. Also MLSTM requires `-g normal` for univariate time-series as well.

`--java` & `--cplus`: Command that is required for non-python implementations. `--java ` for Teaser and ECEC,`--cplus` for EDSC.

`-c <number>`: The class for which the F1-score will be calculated. If -1 is passed then the F1-score of all classes is calculated (not supported for multivariate time-series yet).

`--make-cv`: Takes the training and testing file, merges them and conducts cross validation.

`--folds` : Used when there are premade folds available.

### Test Run for UCR_UEA

`ects` : `ets -t "training file name" -e "testing file name" --make-cv -h Class -c -1 -g vote ects -u 0.0`

`edsc` : `ets -t "training file name" -e "testing file name" --make-cv -h Class -c -1 --cplus -g vote edsccplus`

`ecec` : `ets -t "training file name" -e "testing file name" --make-cv -h Class -c -1 --java -g vote ecec`

`teaser` : `ets t "training file name" -e "testing file name" --make-cv -h Class -c -1 --java -g vote teaser -s 20`

`mlstm` : `ets t "training file name" -e "testing file name" --make-cv -h Class -c -1 -g normal mlstm`

`eco-k` : `ets t "training file name" -e "testing file name" --make-cv -h Class -c -1 -g vote economy-k`


### Test Run for Maritime and Biological

`ects` : `ets -i "file location" -g vote -v (3 for Biological or 5 Maritime) -d 0 -c -1 ects -u 0.0`

`edsc` : `ets -i "file location" -g vote -v (3 for Biological or 5 Maritime) -d 0 -c -1 --cplus edsccplus`

`ecec` : `ets -i "file location" -g vote -v (3 for Biological or 5 Maritime) -d 0 -c -1 --java ecec`

`teaser` : `ets -i "file location" -g vote -v (3 for Biological or 5 Maritime) -d 0 -c -1 --java teaser -s 10`

`mlstm` : `ets -i "file location" -v (3 for Biological or 5 Maritime) -d 0 -c -1 -g normal mlstm`

`eco-k` : `ets -i "file location"" -g vote -v (3 for Biological or 5 Maritime) -d 0 -c -1 economy-k`

### Disclaimer

Any false product and misuse of the used algorithms is on the authors of the paper. Please inform us if you detect any misconduct or misuse of the code/datasets used in this repository.
