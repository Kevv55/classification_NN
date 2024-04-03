import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# understanding how the pixels are arranged in the images
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# data preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# shows a table containing the training images and their labels
# to verify that the data correctly formated
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Building the model
# using three layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # transforms input from a 28x28 px array to a one d array
    tf.keras.layers.Dense(128, activation='relu'), # fully connected layer with a RLU activation function
    tf.keras.layers.Dense(10) # layer that returns the classification scores
])

# Compiling the model
# loss function computes the crossentrophy between predicted and actual classification
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training the model
print(train_images.shape)
print(train_labels.shape)

model.fit(train_images, train_labels, epochs=10)
# Getting the models performance
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# converting the output to probabilities
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
# giving the model unseen data
predictions = probability_model.predict(test_images)
# This serves to see how well the model performed on the 1st instance
# print(predictions[0]) # array list of all the probability levels of different
# print(np.argmax(predictions[0])) # the classification value which the model has the highest confidence
# print(test_labels[0]) # the actual label of the data

max_confidence = 0
max_confidence_class = ""
for i in range(10):
    if predictions[i][np.argmax(predictions[i])] > max_confidence:
        max_confidence = predictions[i][np.argmax(predictions[i])]
        max_confidence_class = class_names[i]

print(f"Class with the highest confidence: {max_confidence_class} with {max_confidence} accuracy")

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# looking at the prediction performance of the ith image
# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()