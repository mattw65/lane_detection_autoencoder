# Lane Detection Autoencoder

## Denoising Convolutional Autoencoder

File: `culane_autoencoder.py`

Includes a GaussianNoise layer for adding noise to input data during training. This trains the autoencoder on the task of image reconstruction. 

### Requirements

File: `culane_autoencoder_requirements.txt`

It is recommended to create a Python virtual environment. Then, the requirements can be installed with:

`pip install -r culane_autoencoder_requirements.txt`

To train on the [CULane dataset](https://xingangpan.github.io/projects/CULane.html), the data can be downloaded from [this Google Drive](https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu). The dataset is quite large. You may download a single part of the dataset, such as `driver_161_90frame`. The path to the extracted dataset directory (eg. `driver_161_90frame`) should be passed to the tool with the `--data_dir` flag. 

### Usage

```
python culane_autoencoder.py [-h] [-d DATA_DIR] [-e EPOCHS] [-b BATCH_SIZE]

optional arguments:
    -h, --help      show help message and exit
    -d DATA_DIR, --data_dir DATA_DIR
                    The directory containing the CULane data
    -e EPOCHS, --epochs EPOCHS
                    The number of epochs to train the Denoising Convolutional Autoencoder for
    -b BATCH_SIZE, --batch_size BATCH_size
                    The number of images to train on at a time
```