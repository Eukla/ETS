# Timeline

An Early Classification of Time-Series library.


## License

This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions; See the [GNU General Public License v3 for more details](http://www.gnu.org/licenses/gpl-3.0.en.html).

## Using a Virtual Environment

1. Install the `virtualenv` package:

```bash
pip install virtualenv
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

## Running tests

Run the following command:

```bash
pytest
``` 

## Downloading the data

For downloading the data run the script `download_data.sh` found in the script folder. The downloaded data can be found inside folder `data`.


## Experimental Results

Previously published experimental results for the UCR dataset can be found [here](docs/experimental_results.md). 