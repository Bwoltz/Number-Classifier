#Basic Number Classifier using a CNN from Keras and MNIST dataset
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def previewData(x_train):
    plt.imshow(x_train[0])
    plt.show()

def fetchDataset():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #previewData(x_train)

    #Reshape the 28x28 pixel images to one dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    #One hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    #Normalize pixel data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

def model(x_train, y_train, x_test, y_test):
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print(len(x_train))
    print(len(y_train))

    model.fit(x_train, y_train, batch_size=128, epochs=9, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    
    filepath = "./saved_model"
    save_model(model, filepath)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def main():

    x_train, y_train, x_test, y_test = fetchDataset()

    model(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()