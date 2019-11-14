# MNIST fashion image recognition with a Tensorflow and Keras neural network

Tensorflow/Keras neural network to train on the [MNIST Fashion dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist) and classify inputs with 87% accuracy.

The dataset is a collection of images (60,000 for training and 10,000 for testing), each one being an image of 28 by 28 pixels, each pixel being a greyscale value from 0 to 255. Each image has an associated label, which is an integer from between 0 and 9, mapping as follows:

Adapted/modified/annotated starting from a [tutorial](https://www.youtube.com/watch?v=6g4O5UOH304) by [@TechWithTimm](https://twitter.com/TechWithTimm).

| Label | Garment type |
|:-----:| ------------ |
|   0   | T-shirt/top  |
|   1   | Trouser      |
|   2   | Pullover     |
|   3   | Dress        |
|   4   | Coat         |
|   5   | Sandal       |
|   6   | Shirt        |
|   7   | Sneaker      |
|   8   | Bag          |
|   9   | Ankle boot   |

## Requirements

Python version: 3.7.4
See dependencies.txt for packages and versions (and below to install).

## Architecture of the neural network
Each image input is a 2D array, representing a 28 by 28 pixel image, with each value being a decimal from 0 to 1 (shrunk down from greyscale 0 to 255 values).

__Input layer:__ 784 neurons (28 * 28), one for each pixel.

__One hidden layer:__ 128 neurons.

__Output layer:__ 10 neurons, one for each of the 10 labels/classes, with as integers from 0 to 9. The total of all of the values of this layer's neurons will equal 1. The network's prediction is the neuron with the highest value.

## Setup

Clone the Git repo.

Install the dependencies:

```bash
pip install -r dependencies.txt
```

## Run

```bash
python main.py
```
