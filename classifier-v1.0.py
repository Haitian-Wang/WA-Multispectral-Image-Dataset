# %% [markdown]
# # Load from files
# 
# 

# %%
import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, UpSampling2D, concatenate, Input, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import rasterio
import re


# %%

base_path = './testb'
pattern = re.compile(r'smalldata_(\d+)_(\d+)')
indices = ['ExG', 'ExR', 'PRI', 'MGRVI', 'SAVI', 'MSAVI', 'EVI', 'REIP', 'NDVI', 'GNDVI', 'CI', 'OSAVI', 'TVI', 'MCARI', 'TCARI']

def load_tif(file_path):
    # Function to load .tif file and return as a numpy array using rasterio
    with rasterio.open(file_path) as src:
        return src.read(1)  # Read the first band

def read_X(dir=base_path, indices=indices):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for dir_name in dirs:
            match = pattern.match(dir_name)
            if match:
                group_number = match.group(1)
                sub_group_number = match.group(2)
                dir_path = os.path.join(root, dir_name)
                channels = []
                skip_directory = False
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.tif'):
                        for index in indices:
                            if file_name.startswith(index):
                                file_path = os.path.join(dir_path, file_name)
                                data = load_tif(file_path)
                                if np.isnan(data).any():
                                    print(f"Skipping directory due to NaN values in: {file_path}")
                                    skip_directory = True
                                    break
                                channels.append(data)
                        if skip_directory:
                            break
                    if file_name.startswith("label_matrix"):
                        file_path = os.path.join(dir_path, file_name)
                        label_matrix = pd.read_csv(file_path, header=None).values
                if not skip_directory:
                    images.append(channels)
                    labels.append(label_matrix)
    return np.array(images), np.array(labels)

# # Example usage
# images, labels = read_X()
# print(images.shape, labels.shape)


# %%
X,y=read_X()

# %% [markdown]
# # Helper

# %%
import numpy as np

def convert_to_one_hot(y):
    # Get the shape of the input array
    n, h, w = y.shape
    
    # Initialize an all-zero array with shape (n, h, w, 4)
    y_one_hot = np.zeros((n, h, w, 4), dtype=int)
    
    # Use advanced indexing to convert the original array values to one-hot encoding
    for i in range(4):
        y_one_hot[..., i] = (y == i)
    
    return y_one_hot


def get_predicted_labels(predictions):
    """
    Convert the predicted probability array to a label array.
    
    Parameters:
    predictions: A predicted probability array with shape (n, 512, 512, 4)
    
    Returns:
    A label array with shape (n, 512, 512), where each point represents its most probable class
    """
    predicted_labels = np.argmax(predictions, axis=-1)
    return predicted_labels


def one_hot_to_labels(y_one_hot):
    """
    Convert a one-hot encoded array back to a label array.
    
    Parameters:
    y_one_hot: A one-hot encoded array with shape (n, 512, 512, 4)
    
    Returns:
    A label array with shape (n, 512, 512)
    """
    # Use np.argmax to find the index of the maximum value in the fourth dimension
    y_labels = np.argmax(y_one_hot, axis=-1)
    
    return y_labels


# %% [markdown]
# # Model

# %%
# Normalize input data to [0, 1]
X = X / np.max(X)

# Transpose the dimensions of X to (0, 2, 3, 1)
X = X.transpose((0, 2, 3, 1))

# Convert y to one-hot encoding
y_one_hot = convert_to_one_hot(y)


# %%
import tensorflow as tf

def preprocess_image(image, label):
    # Do not resize, keep the original dimensions
    return image, label

def load_dataset(images, labels, batch_size=4):
    # Create a TensorFlow dataset from the images and labels
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Apply the preprocess_image function to each element in the dataset
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch the dataset and prefetch for better performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# Load the training dataset
train_dataset = load_dataset(X, y_one_hot)



# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout
from tensorflow.keras.models import Model

# Define a custom ResizeLayer
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width):
        super(ResizeLayer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, (self.target_height, self.target_width))

# Define the input tensor with shape (512, 512, 15)
input_tensor = Input(shape=(512, 512, 15))

# Add a convolutional layer to convert the input to a 3-channel input suitable for InceptionV3
x = Conv2D(3, (1, 1), padding='same', activation='relu')(input_tensor)
print(x.shape)  # 512

# Use the pre-trained InceptionV3 model, excluding the top classification layer
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Connect the custom input layer to the base model
x = base_model(x)

# Use convolutional layers to maintain the spatial dimensions
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
print(x.shape)

# Add the final convolutional layer
x = Conv2D(4, (1, 1), padding='same')(x)
print(x.shape)

# Use the custom resize layer to resize the output to (512, 512)
x = ResizeLayer(512, 512)(x)
print(x.shape)

# Apply the Softmax activation function to ensure the output represents a probability distribution
predictions = tf.keras.layers.Softmax(axis=-1)(x)

# Define the model with the input tensor and the predictions
model = Model(inputs=input_tensor, outputs=predictions)

# Freeze the convolutional layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with RMSprop optimizer and categorical cross-entropy loss
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Uncomment the lines below to print the predictions shape and verify the sum of probabilities
# print(predictions.shape)
# print(predictions)
# print(np.sum(predictions[0, :, :, :], axis=-1))


# %%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') <= 0.1099 and logs.get('loss') <= 0.1099:
            print('\n\n Reached The Destination!')
            self.model.stop_training = True

callbacks = myCallback()
history = model.fit(
    train_dataset,
    epochs=10
    # callbacks=[callbacks]
)


# %%
model_path='./inceptionv3_fcn_model_23_5.h5'

model.save(model_path)
print(model_path)

# %%

if os.path.exists(model_path):
    print(f"Model saved successfully at {model_path}")
else:
    print(f"Model not found at {model_path}")

# %% [markdown]
# # Get prediction

# %%
from keras.models import load_model
import tensorflow as tf

# Define custom layer without 'trainable' argument
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width])

# Add custom layer to the custom_objects dictionary
custom_objects = {'ResizeLayer': ResizeLayer}

# Load the model with the custom objects
with tf.keras.utils.custom_object_scope(custom_objects):
    model = load_model('inceptionv3_fcn_model_23_5.h5', compile=False)


# %%
def make_pred(dir):
    # Read the test data from the specified directory
    X_test, y_test = read_X(dir=dir)
    
    # Print the shape and data type of the test data
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Data type of X_test: {X_test.dtype}")
    
    # Normalize the test data to the range [0, 1]
    X_test = X_test / np.max(X_test)
    
    # Transpose the dimensions of X_test to (0, 2, 3, 1)
    X_test = X_test.transpose((0, 2, 3, 1))
    
    # Make predictions using the model
    predictions = model.predict(X_test)
    
    # Uncomment the following line to print the predictions
    # print(predictions)
    
    # Convert y_test to one-hot encoding
    y_test_one_hot = convert_to_one_hot(y_test)
    
    return y_test_one_hot, predictions


# %%
y_test,y_pred= make_pred(dir='./testset')


# %% [markdown]
# # Evaluation

# %%
# Check if the sum of channel values for each pixel is equal to 1
sum_of_channels = np.sum(y_pred, axis=-1)
#print(sum_of_channels)

# Verify if all values are close to 1
are_all_close_to_one = np.allclose(sum_of_channels, 1)
print(f"Are all sums of the four channels equal to 1? {are_all_close_to_one}")


# %%
def calculate_accuracy(y_true, y_pred, num_classes=4):
    """
    Calculate classification accuracy.
    
    :param y_true: Actual labels, shape (batch_size, height, width, num_classes)
    :param y_pred: Predicted labels, shape (batch_size, height, width, num_classes)
    :param num_classes: Number of classes
    :return: Classification accuracy
    """
    # Convert one-hot encoded labels to class indices
    y_true_class = np.argmax(y_true, axis=-1)
    y_pred_class = np.argmax(y_pred, axis=-1)
    
    # Calculate accuracy
    correct_predictions = np.sum(y_true_class == y_pred_class)
    total_predictions = y_true_class.size
    
    accuracy = correct_predictions / total_predictions
    return accuracy

# Call the function to calculate accuracy
calculate_accuracy(y_test, y_pred, num_classes=4)


# %%
def calculate_miou(y_true, y_pred, num_classes):
    """
    Calculate Mean Intersection over Union (mIoU).
    
    :param y_true: Actual labels, shape (batch_size, height, width, num_classes)
    :param y_pred: Predicted labels, shape (batch_size, height, width, num_classes)
    :param num_classes: Number of classes
    :return: Mean Intersection over Union (mIoU)
    """

    # Convert one-hot encoded labels to class indices
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    iou_list = []
    for c in range(num_classes):

        
        # Create boolean arrays for the current class
        true_class = (y_true == c)
        pred_class = (y_pred == c)
        
        # Calculate the intersection and union for the current class
        intersection = np.sum(true_class & pred_class)
        
        union = np.sum(true_class | pred_class)
        
        if union == 0:
            iou = 1.0  # If there is no ground truth or predicted instance in this class

        else:
            iou = intersection / union

        
        # Append the IoU for the current class to the list
        iou_list.append(iou)
    
    # Calculate the mean IoU across all classes
    miou = np.mean(iou_list)
    
    return miou

# Call the function to calculate mIoU
calculate_miou(y_test, y_pred, num_classes=4)



