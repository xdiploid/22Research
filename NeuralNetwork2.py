from importlib.resources import path
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

 #Shift + right click -> 'copy as path' to get file path
 #Make sure to use double backslashes to prevent 'unicodeescape' error
DATADIR  = "C:\\Users\\Kiwi\\Downloads\\catsdogs\\PetImages"
CATERGORIES = ["Dog", "Cat"]

# for category in CATERGORIES:
#     #path to cats or dogs directory
#     path = os.path.join(DATADIR, category) 
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap="gray")
#         plt.show()
#         break
#     break

training_data = []
IMG_SIZE = 50

def create_trainig_data():
    for category in CATERGORIES:
        #path to cats or dogs directory
        path = os.path.join(DATADIR, category)
        class_num = CATERGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            # plt.imshow(img_array, cmap="gray")
            # plt.show()
            
create_trainig_data()
print(len(training_data))

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
X[1]