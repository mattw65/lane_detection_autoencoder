import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt


class DenoisingCAE(Model):
    def __init__(self, height, width, depth, filters=(32, 64), latentDim=16, seed=36):
        super(DenoisingCAE, self).__init__()

        self.latentDim = latentDim
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        # channels dimension itself
        self.inputShape = (height, width, depth)
        chanDim = -1
        
        encoder_list = []
        # Define the input to the encoder
        encoder_list.append(Input(shape=self.inputShape))
        
        # Define layer for adding noise to input
        encoder_list.append(GaussianNoise(stddev=1.0, seed=seed))
        # add_noise = keras.Sequential([
        #     keras.layers.GaussianNoise(stddev=1.0, seed=seed)
        # ])

        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            encoder_list.append(Conv2D(f, (3, 3), strides=2, padding="same"))
            encoder_list.append(LeakyReLU(alpha=0.2))
            encoder_list.append(BatchNormalization(axis=chanDim))
        
        encoder_list.append(Flatten())
        encoder_list.append(Dense(self.latentDim))
        
        # build the encoder model
        self.encoder = keras.Sequential(encoder_list, name = "encoder")
        print(self.encoder.summary())
        
        decoder_list = []
        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        self.latentInputs = Input(shape=(self.latentDim,))
        decoder_list.append(self.latentInputs)

        volumeSize = self.encoder.layers[-3].output_shape
        # print(volumeSize)  # (None, 64, 64, 64)
        
        decoder_list.append(Dense(np.prod(volumeSize[1:])))
        decoder_list.append(Reshape((volumeSize[1], volumeSize[2], volumeSize[3])))
        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            decoder_list.append(Conv2DTranspose(f, (3, 3), strides=2, padding="same"))
            decoder_list.append(LeakyReLU(alpha=0.2))
            decoder_list.append(BatchNormalization(axis=chanDim))
        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        decoder_list.append(Conv2DTranspose(depth, (3, 3), padding="same"))
        decoder_list.append(Activation("sigmoid"))
        # build the decoder model
        self.decoder = keras.Sequential(decoder_list, name = "decoder")
        print(self.decoder.summary())
    
    def call(self, X):
        # our autoencoder is the encoder + decoder
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='day', type=str, help='The directory containing the subset of CULane data')
    parser.add_argument('-e', '--epochs', default=20, type=int, help="Number of epochs to train model for")
    parser.add_argument('-b', '--batch_size', default=36, type=int, help="Number of images to train on at a time")
    args = parser.parse_args()

    seed = 36
    epochs = args.epochs
    batch_size = args.batch_size

    # Load subset of CULane data from directory
    # Original image size = 1640 x 590
    image_size = (128, 320)  # (height, width)
    data_dir = args.data_dir

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        validation_split = 0.2
    )
    
    train_ds = train_datagen.flow_from_directory(
        data_dir,
        subset = "training",
        target_size = image_size,
        color_mode = "grayscale",
        classes = None,
        class_mode = "input",  # labels are identical to input images
        batch_size = batch_size,
        shuffle = True,
        seed = seed
    )

    val_ds = train_datagen.flow_from_directory(
        data_dir,
        subset = "validation",
        target_size = image_size,
        color_mode = "grayscale",
        classes = None,
        class_mode = "input",  # labels are identical to input images
        batch_size = batch_size,
        shuffle = True,
        seed = seed
    )

    # Found 10386 files belonging to 1 class 
    # Using 8309 files for training
    # Using 2077 files for validation

    dcae = DenoisingCAE(height=image_size[0], width=image_size[1], depth=1)

    optim = Adam(learning_rate=1e-3)
    dcae.compile(loss="mse", optimizer=optim)

    H = dcae.fit(
        train_ds,
        validation_data = val_ds,
        epochs = epochs,
        batch_size = batch_size
    )

    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss (MSE)")
    plt.legend(loc="upper right")
    plt.savefig("dcae_training_metrics.png", format='png')

    dcae.save("dcae")

