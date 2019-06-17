#Importing necessary libraries
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras

#Importing and loading data from MNIST dataset
np.random.seed(101)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

print(x_train.shape)
print(len(y_train))
print(y_train)

print(x_test.shape)
print(len(y_test))
print(y_test)

#scaling the values of pixels from 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

#Verifying the format of training set by displaying first 25 images
plt.figure(figsize = (28,28))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

#Building the model

#Configuring the layers of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

#Compiling the model    
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the model 
history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))  

#plotting graphs of accuracy and loss vs epochs for training and validation set.
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()
fig

#Evaluating the accuracy
test_loss, test_acc = model.evaluate(x_test, Y_test)
print('Test accuracy:', test_acc)

#Making predictions
predictions = model.predict_classes(x_test)
print(predictions[0])
print(y_test[0])

correct_indices = np.nonzero(predictions == y_test)[0]
incorrect_indices = np.nonzero(predictions != y_test)[0]

print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")
