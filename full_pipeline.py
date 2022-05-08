import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

from load_and_eval_DCAE import model_load, data_load

def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(0, height-20), (width//2, 100), (width, height-20)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def avg_lines(img, lines):
    posM = []
    posX = []
    negM = []
    negX = []
    for l1 in lines:
        l = l1[0]
        if (l[2]-l[0]) == 0:
            m = 100000000
        else:
            m = (l[3]-l[1])/(l[2]-l[0])
        
        if m == 0: 
            continue
        elif m > 0:
            x = (160 - l[1])/m + l[0]
            posM.append(m)
            posX.append(x)
        else:
            x = (160 - l[1])/m + l[0]
            negM.append(m)
            negX.append(x)
            
    
    if len(posM) != 0:
        posMavg = sum(posM)/len(posM)
        posXavg = sum(posX)/len(posX)
        posX2avg = (100 - 160)/posMavg + posXavg

    if len(negM) != 0:
        negMavg = sum(negM)/len(negM) + 0.0000001
        negXavg = sum(negX)/len(negX) + 0.0000001
        negX2avg = (100 - 160)/negMavg + negXavg
    
#     print(posX2avg, posXavg)
    if len(posM) != 0:
        cv2.line(img, (int(posXavg), 160), (int(posX2avg), 100), (0), 2)
    if len(negM) != 0:
        cv2.line(img, (int(negXavg), 160), (int(negX2avg), 100), (0), 2)
    return img

def get_lines(imgG, resize_shape):
    imgG = imgG.reshape(resize_shape)*255

    mask_light = cv2.inRange(imgG, resize_shape[0] - 40, resize_shape[1] - 50)
    # print("mask_light", mask_light)
    
    isolated = region(mask_light)
    # print("isolated", isolated)

#     plt.imshow(isolated)
#     plt.show()

    lines = cv2.HoughLinesP(isolated,rho = 2,theta = 1*np.pi/180,threshold = 10,minLineLength = 10,maxLineGap = 10)
    # print("lines", lines)

    if lines is not None:
        imgG = avg_lines(imgG, lines)

    return imgG

def get_lines_MSE(img1, img2):
    im1Lines = get_lines(img1)
    im2Lines = get_lines(img2)

    err = np.sum((im1Lines.astype("float") - im2Lines.astype("float")) ** 2)
    err /= float(im1Lines.shape[0] * im1Lines.shape[1])

    return err

def get_MSE(img1, img2):
    im1 = img1.reshape((168,300))
    im2 = img2.reshape((168,300))

    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im1.shape[1])

    return err

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
    
    # print(test_noisy.shape)
    test_loss = model.evaluate(test_noisy, test)
    print("Test set loss:", test_loss)

    test_recon = model.predict(test_noisy)

    # Get MSE values
    num_test = test_noisy.shape[0]
    testing_reshape = test.reshape(num_test, -1)
    # print(testing_reshape.shape)
    
    base_mse = np.mean((np.square(test.reshape(num_test, -1) - test_noisy.reshape(num_test, -1))).mean(axis=1))
    reco_mse = np.mean((np.square(test.reshape(num_test, -1) - test_recon.reshape(num_test, -1))).mean(axis=1))

    assert(test_loss - np.mean(reco_mse) < 0.00001)  # account for different calculations having different precision

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
        ax[0,c].imshow(test[(c+1)*selector].reshape(img_shape), cmap=plt.cm.gray)

        # Plot noisy
        # noisy_example = test_noisy[c*selector]
        # print(noisy_example.shape)
        ax[1,c].imshow(test_noisy[(c+1)*selector].reshape(img_shape), cmap=plt.cm.gray)

        # Plot reconstructed
        # noisy_example = np.expand_dims(noisy_example, axis=0)
        # print(noisy_example.shape)
        # reconstructed = model.predict(noisy_example, batch_size=1)
        ax[2,c].imshow(test_recon[(c+1)*selector].reshape(img_shape), cmap=plt.cm.gray)

    fig.tight_layout(pad=0.05)
    # fig.subplots_adjust(top=0.85)
    plt.savefig("sample_images_%d.png" % (noise_level * 100))

    plt.clf()

    fig, ax = plt.subplots(nrows=3, ncols=num_display, sharey='row', sharex='all', figsize=(10,4.5))
    fig.suptitle("Lane Detection for Noise Level=%.1f, Base MSE=%.5f, Reconstruction MSE=%.5f" % (noise_level, base_mse, reco_mse), fontsize=14)
    for r in range(len(ax)):
        for c in range(len(ax[0])):
            ax[r,c].set_xticks([])      
            ax[r,c].set_yticks([])

    for i in range(len(y_labels)):
        ax[i,0].set_ylabel(y_labels[i], fontsize=14)

    for c in range(num_display):
        # Plot original
        # temp = get_lines(test[(c+1)*selector], img_shape)
        # print(temp.shape)
        # print(temp)
        # print(type(temp[0]))
        ax[0,c].imshow(np.expand_dims(get_lines(test[(c+1)*selector], img_shape), axis=-1)/255)  #, cmap=plt.cm.gray)

        # Plot noisy
        # noisy_example = test_noisy[c*selector]
        # print(noisy_example.shape)
        ax[1,c].imshow(np.expand_dims(get_lines(test_noisy[(c+1)*selector], img_shape), axis=-1)/255)  #, cmap=plt.cm.gray)

        # Plot reconstructed
        # noisy_example = np.expand_dims(noisy_example, axis=0)
        # print(noisy_example.shape)
        # reconstructed = model.predict(noisy_example, batch_size=1)
        ax[2,c].imshow(np.expand_dims(get_lines(test_recon[(c+1)*selector], img_shape), axis=-1)/255)  #, cmap=plt.cm.gray)

    fig.tight_layout(pad=0.05)
    # fig.subplots_adjust(top=0.85)
    plt.savefig("lane_detection_results_%d.png" % (noise_level * 100))

    # err_noisy = []
    # err_recon = []
    # err_r = []

    # for i in range(len(test)):
    # #     err_noisy.append(get_lines_MSE(testX[i], testXNoisy[i]))
    # #     err_recon.append(get_lines_MSE(testX[i], autoencoder(testXNoisy[i].reshape(1,168,300,1)).numpy()))
    #     err_r.append(get_MSE(test[i], model(test_noisy[i].reshape(1,168,300,1)).numpy()))
        
    # # print('Average MSE for Noisy Photos: ', sum(err_noisy)/len(err_noisy))
    # # print('Average MSE for Reconstructed Photos: ', sum(err_recon)/len(err_recon))
    # print(sum(err_r)/len(err_r))