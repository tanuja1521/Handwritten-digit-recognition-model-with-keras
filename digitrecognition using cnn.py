 #Importing necessary libraries
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras

 batch_size = 128
 num_classes = 10
 epochs = 12
 # Image dimension is 28 pixels by 28 pixels.
 img_rows, img_cols = 28, 28
 # Load dataset. Each pixel is 0-255 grayscale value. Train shape is (60000, 28, 28). Test shape is (10000, 28, 28)
 (x_train, y_train), (x_test, y_test) = mnist.load_data()
 # Reshape dataset to (60000, 28, 28, 1) and (10000, 28, 28, 1).
 x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
 x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
 input_shape = (img_rows, img_cols, 1)
 # Convert 0-255 to float and scale to 0-1.
 x_train = x_train.astype('float32')
 x_test = x_test.astype('float32')
 x_train /= 255
 x_test /= 255
 # Encode targets using one hot encoding.
 y_train = keras.utils.to_categorical(y_train, num_classes)
 y_test = keras.utils.to_categorical(y_test, num_classes)
 # Define neural networks.
 model = Sequential()
 model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape))
 model.add(Conv2D(64, (3, 3), activation='relu'))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Dropout(0.25))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dropout(0.5))
 model.add(Dense(num_classes, activation='softmax'))
 model.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adadelta(),
               metrics=['accuracy'])
 # Train neural networks.
 model.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=epochs,
           verbose=2,
           validation_data=(x_test, y_test))
 # Evaluate using test set.
 test_loss, test_accuracy = model.evaluate(x_test, y_test)
 predictions = model.predict_classes(x_test)
 print(predictions[0])
 print(y_test[0])

 correct_indices = np.nonzero(predictions == y_test)[0]
 incorrect_indices = np.nonzero(predictions != y_test)[0]

print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")
