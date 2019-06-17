import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() 


print(x_train.shape)
print(len(y_train))
print(y_train)

x_train = x_train / 255.0

y_train = y_train / 255.0

plt.figure(figsize = (28,28))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(x_test)

predictions[0]

print(np.argmax(predictions[0]))

print(y_test)


