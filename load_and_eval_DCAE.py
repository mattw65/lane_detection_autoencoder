import numpy as np
import keras
import argparse

from PIL import Image
import matplotlib.pyplot as plt

def model_load(model_dir):
    """
    Loads model from save directory
    """
    model = keras.models.load_model(model_dir)
    print(model.summary())

    return model

def data_load(data_dir, num_examples, resize_shape):
    test = []
    for i in range(num_examples):
        img = Image.open("%s/frame%d_reduced.jpg" % (data_dir, i))
        img_data = img.resize(resize_shape)
        test.append(np.array(img_data))
            
    testX = np.asarray(test)
    # print(testX.shape)

    testX = np.expand_dims(testX, axis=-1)
    testX = testX.astype("float32") / 255.0

    testNoise = np.random.normal(loc=0.0, scale=0.1, size=testX.shape)
    testXNoisy = np.clip(testX + testNoise, 0, 1)
    
    return testX, testXNoisy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='day', type=str, help='The directory containing the subset of CULane data')
    parser.add_argument('-m', '--model_dir', default='dcae', type=str, help="The directory containing the saved model information")
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    model = model_load(model_dir)

    height = 128
    width = 320
    img_shape = (height, width) 
    test, test_noisy = data_load(data_dir, 1260, resize_shape=(width, height))
    
    # print(test_noisy.shape)
    test_loss = model.evaluate(test_noisy, test)
    print("Test set loss:", test_loss)

    num_display = 3
    fig, ax = plt.subplots(nrows=3, ncols=num_display, sharey='row')
    for r in range(len(ax)):
        for c in range(len(ax[0])):
            ax[r,c].set_xticks([])      
            ax[r,c].set_yticks([])

    y_labels = ["Original", "Noisy", "Reconstructed"]
    for i in range(len(y_labels)):
        ax[i,0].set_ylabel(y_labels[i])

    for c in range(num_display):
        # Plot original
        ax[0,c].imshow(test[c].reshape(img_shape), cmap=plt.cm.gray)

        # Plot noisy
        noisy_example = test_noisy[c]
        # print(noisy_example.shape)
        ax[1,c].imshow(test_noisy[c].reshape(img_shape), cmap=plt.cm.gray)

        # Plot reconstructed
        noisy_example = np.expand_dims(noisy_example, axis=0)
        # print(noisy_example.shape)
        reconstructed = model.predict(noisy_example, batch_size=1)
        ax[2,c].imshow(reconstructed.reshape(img_shape), cmap=plt.cm.gray)

    plt.tight_layout(pad=0.05)
    plt.savefig("sample_images.png")

