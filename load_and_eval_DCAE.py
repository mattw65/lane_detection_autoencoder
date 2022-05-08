import numpy as np
import keras
import argparse

import os

from PIL import Image
import matplotlib.pyplot as plt

def model_load(model_dir):
    """
    Loads model from save directory
    """
    model = keras.models.load_model(model_dir)
    print(model.summary())

    return model

def data_load(data_dir, noise_level, resize_shape):
    test = []
    sub_dirs = [s for s in os.listdir(data_dir) if s.endswith('.MP4')]
    for sub_dir in sub_dirs:
        image_files = [f for f in os.listdir(os.path.join(data_dir,sub_dir)) if f.endswith('.jpg')]
        for image in image_files:
            img = Image.open(os.path.join(data_dir, sub_dir, image)).convert('L')
            img_data = img.resize(resize_shape)
            test.append(np.array(img_data))    
            
    testX = np.asarray(test)
    # print(testX.shape)

    testX = np.expand_dims(testX, axis=-1)
    testX = testX.astype("float32") / 255.0

    testNoise = np.random.normal(loc=0.0, scale=noise_level, size=testX.shape)
    testXNoisy = np.clip(testX + testNoise, 0, 1)
    
    return testX, testXNoisy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='day', type=str, help='The directory containing the subset of CULane data')
    parser.add_argument('-m', '--model_dir', default='dcae', type=str, help="The directory containing the saved model information")
    parser.add_argument('-n','--noise_level', default=0.1, type=float, help="The standard deviation of the gaussian noise distribution")
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    noise_level = args.noise_level

    model = model_load(model_dir)

    height = 128
    width = 320
    img_shape = (height, width) 
    test, test_noisy = data_load(data_dir, noise_level=noise_level, resize_shape=(width, height))
    
    print(test_noisy.shape)
    test_loss = model.evaluate(test_noisy, test)
    print("Test set loss:", test_loss)

    num_display = 3
    fig, ax = plt.subplots(nrows=3, ncols=num_display, sharey='row', sharex='all', figsize=(10,4.5))
    fig.suptitle("Noise Level=%.1f, Test Loss=%.5f" % (noise_level, test_loss), fontsize=14)
    for r in range(len(ax)):
        for c in range(len(ax[0])):
            ax[r,c].set_xticks([])      
            ax[r,c].set_yticks([])

    y_labels = ["Original", "Noisy", "Reconstructed"]
    for i in range(len(y_labels)):
        ax[i,0].set_ylabel(y_labels[i], fontsize=14)

    selector = 200
    for c in range(num_display):
        # Plot original
        ax[0,c].imshow(test[c*selector].reshape(img_shape), cmap=plt.cm.gray)

        # Plot noisy
        noisy_example = test_noisy[c*selector]
        # print(noisy_example.shape)
        ax[1,c].imshow(test_noisy[c*selector].reshape(img_shape), cmap=plt.cm.gray)

        # Plot reconstructed
        noisy_example = np.expand_dims(noisy_example, axis=0)
        # print(noisy_example.shape)
        reconstructed = model.predict(noisy_example, batch_size=1)
        ax[2,c].imshow(reconstructed.reshape(img_shape), cmap=plt.cm.gray)

    fig.tight_layout(pad=0.05)
    # fig.subplots_adjust(top=0.85)
    plt.savefig("sample_images.png")

