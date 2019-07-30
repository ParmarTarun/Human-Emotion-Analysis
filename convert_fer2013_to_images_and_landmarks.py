import numpy as np
import pandas as pd
import os
import errno
import scipy.misc

image_height = 48
image_width = 48
SAVE_IMAGES = True
IMAGES_PER_LABEL = 500
OUTPUT_FOLDER_NAME = "fer2013_features"

new_labels = [0, 1, 2, 3, 4, 5, 6]

print( str(len(new_labels)) + " expressions")

nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))

try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise

def get_new_label(label):
    return new_labels.index(label)

print( "importing csv file...")
data = pd.read_csv('fer2013.csv')
for category in data['Usage'].unique():
    print( "converting set: " + category + "...")
    # create folder
    if not os.path.exists(category):
        try:
            os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
               pass
            else:
                raise

    # get samples and labels of the actual category
    category_data = data[data['Usage'] == category]
    samples = category_data['pixels'].values
    labels = category_data['emotion'].values
    
    images = []
    labels_list = []

    for i in range(len(samples)):
        try:
            if nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:
                image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                images.append(image)

                if SAVE_IMAGES:
                    scipy.misc.imsave(OUTPUT_FOLDER_NAME+'/'+ category + '/' + str(i) + '.jpg', image)        

                labels_list.append(get_new_label(labels[i]))
                nb_images_per_label[get_new_label(labels[i])] += 1

        except Exception as e:
            print( "error in image: " + str(i) + " - " + str(e))

    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)
    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)
    