# ETSC: Early Time Series Classification

`ETSC` is a Python Early Classification of Time-Series library for public use.

Aim of this work is to study and collect algorithms that conduct early time-series classification, in a user-friendly format, for researchers to use for their work.

Currently five algorithms are included in this directory. A python cli, simplifies the execution of each algorithm
The predictions are evalueated through metrics such as earliness, accuracy, f1-score(if wanted) and computation time for both training and testing.
Gui_run or terminal_run are demos, which run the desired algorithm for all the UCR datasets.

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
Over 100 datasets are available, derived from the [UCR library](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/). Multivariate datasets from the Biological and Maritime field are also provided.
## Experimental Setup

| Algorithm | Parameters |
|---|---|
| ECTS [\[paper\]](https://link.springer.com/article/10.1007/s10115-011-0400-x) | support = 0 |
| EDSC [\[paper\]](https://epubs.siam.org/doi/10.1137/1.9781611972818.22) | CHE, k=3, min_length=5, max_length=len(time_series)/2 |
| TEASER [\[paper\]](https://link.springer.com/article/10.1007/s10618-020-00690-z) [\[code\]](https://github.com/patrickzib/SFA) | S=20 (for the UCR), S=10 (for the biological and maritime) |
| ECEC [\[paper\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8765556) [\[code\]](https://github.com/junweilvhfut/ECEC)| training_times=20, length = len(time_series)/20,a=0.8 |
| MLSTM [\[paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200?via%3Dihub) [\[code\]](https://github.com/titu1994/MLSTM-FCN) | LSTM cells = 8, tested_lengths = [0.4,0.5,0.6] %  |

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


### Test Run

`ects` : `ets -t data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv -e data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv -d 0 -c -1 -s \\t ects -u 0.0`

`edsc` : `ets -t data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv -e data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv -d 0 -c -1 -s \\t --cplus edsccplus`

`ecec` : `ets -t data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv -e data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv -d 0 -c -1 -s \\t --java ecec`

`teaser` : `ets -t data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv -e data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv -d 0 -c -1 -s \\t --java teaser -s 20`

`mlstm` : `ets -t data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv -e data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv -d 0 -c -1 -s \\t -g normal mlstm`

There are also two demo programs for the UCR Dataset run with:

`python gui_run.py`, where algorithms are selected through a GUI.

`python terminal run_py -a <algorithm>`, where the algorithms are given as command line arguements.
