#   python -m venv myenv
#   myenv\Scripts\activate
#   python weather.py

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth
from tensorflow.keras import Model
import random

# Define dataset directories
TRAIN_DIR = "H:/CV/Weather Detection/dataset/training"
VALIDATION_DIR = "H:/CV/Weather Detection/dataset/validation"
TEST_DIR = "H:/CV/Weather Detection/dataset/testing"

# Parameters
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Image data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

# Load the training dataset
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load the validation dataset
validation_generator = validation_test_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load the testing dataset
test_generator = validation_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model setup
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomHeight(0.2),
    RandomWidth(0.2),
], name='data_augmentation')

input_shape = (224, 224, 3)
base_model = tf.keras.applications.InceptionV3(include_top=False)
base_model.trainable = False

inputs = tf.keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs, outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model checkpoint callback
checkpoint_path = "weather_model_checkpoints/checkpoint.weights.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch',
    verbose=1
)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback]
)

# Fine-tune the model
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback]
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

# Function to predict a random image
def predict_random_image(model, dataset_dir):
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))

    if len(image_paths) > 0:
        random_image_path = random.choice(image_paths)
        img = image.load_img(random_image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        print(f"Random Image: {random_image_path}")
        if predicted_class[0]==0:
            print("Predicted Class: Dew")
        elif predicted_class[0]==1:
            print("Predicted Class: Fogsmog")
        elif predicted_class[0]==2:
            print("Predicted Class: Frost")
        elif predicted_class[0]==3:
            print("Predicted Class: Glaze")
        elif predicted_class[0]==4:
            print("Predicted Class: Hail")
        elif predicted_class[0]==5:
            print("Predicted Class: Lightning")
        elif predicted_class[0]==6:
            print("Predicted Class: Rain")
        elif predicted_class[0]==7:
            print("Predicted Class: Rainbow")
        elif predicted_class[0]==8:
            print("Predicted Class: Rime")
        elif predicted_class[0]==9:
            print("Predicted Class: Sandstorm")
        elif predicted_class[0]==10:
            print("Predicted Class: Snow")
    else:
        print("No images found in the specified directory.")

# Predict using a random image from the validation dataset
predict_random_image(model, VALIDATION_DIR)
