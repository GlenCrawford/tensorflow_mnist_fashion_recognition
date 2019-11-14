import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

### Import and configure the input data ###

# Import the MNIST Fashion dataset.
data = keras.datasets.fashion_mnist

# Split the dataset up into sub datasets, for training and testing.
# The images are a 2D array/matrix representing a 28 x 28 pixel image. Each value is a greyscale value from 0 to 255.
(train_images, train_labels), (test_images, test_labels) = data.load_data()
# print('Number of training images: ' + str(len(train_images)))
# print('Number of testing images: ' + str(len(test_images)))

# Each image has a label from 10 possible label values: digits between 0 and 9. Create a mapping to be more human-friendly.
class_names = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot'
]

# Uncomment to view raw image data as a 28 x 28 matrix of pixels:
# print(train_images[0])

# Uncomment to view an actual image:
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()

# Shrink the pixel values of all images down so that each value is a decimal between zero and one, rather than an integer between zero and 255 (images are numpy arrays).
train_images = train_images / 255.0
test_images = test_images / 255.0

### Define the network architecture ###

# Define the layers in sequential order. Keras will draw the connections between the layers for us.
model = keras.Sequential([
  # Input layer. Have Keras flatten out the 2D 28 x 28 grid.
  keras.layers.Flatten(input_shape=(28, 28)),
  # Hidden layer. Dense means fully connected layer.
  keras.layers.Dense(128, activation='relu'),
  # Output layer. Softmax activation will make all values of the layer's neurons add up to 1, e.g. 0.1 + 0.7 + 0.2 = 1. The prediction of the network is the neuron with the highest value.
  keras.layers.Dense(10, activation='softmax')
])

# Configure the model parameters.
model.compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy']
)

### Train the network ###

# "Epoch" means how many times the model is going to train on each input.
# The network will thus see each image multiple times, in a random order (as opposed to seeing the same image multiple times sequentially before moving on to the next image).
model.fit(train_images, train_labels, epochs=5)

### Automatically test the network ###
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test run loss: ' + str(test_loss))
print('Test run accuracy: ' + str(test_accuracy))

### Manually test/use the network ###

# Pass images from the test subset into the network and view the network's prediction.
# The result is an array of predictions, one for each input image.
# Each prediction is an array of the values of each of the neurons in the output layer. Ie, 10 values, decimals adding up to 1. The neuron with the highest number is the prediction.
# Test on the below number of random images from the test subset. Open a window with the image, prediction, and actual value.
test_image_predictions = model.predict(test_images)

number_of_images_to_test = 10
random_test_image_indexes = random.sample(range(0, len(test_images)), number_of_images_to_test)

for test_image_index in random_test_image_indexes:
  test_image = test_images[test_image_index]
  actual_label = class_names[test_labels[test_image_index]]

  # Get the values of the neurons in the output layer for this image.
  # Then get the index of the neuron with the highest value.
  output_layer_neuron_values = test_image_predictions[test_image_index]
  output_layer_highest_value_neuron_index = np.argmax(output_layer_neuron_values)
  prediction_label = class_names[output_layer_highest_value_neuron_index]

  plt.grid(False)
  plt.imshow(test_image, cmap=plt.cm.binary)
  plt.xlabel('Actual: ' + actual_label)
  plt.title('Prediction: ' + prediction_label)
  plt.show()
