import numpy as np
from tensorflow import keras
from  tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import os
from tensorflow.keras.callbacks import EarlyStopping

path = r'P:\skind\Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration'
train_path = os.path.join(path, 'train')
disease_list = os.listdir(train_path)
print(disease_list)

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed = 42,
    shuffle = True,
)

val_data = datagen.flow_from_directory(
    train_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation',
    seed = 42,
    shuffle = True,
)



model = Sequential([
    layers.Conv2D(32, (3,3), padding = 'same', activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), padding = 'same', activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), padding = 'same', activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(disease_list), activation='softmax')                
])

print(model.summary())

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_split = val_data,
    epochs = 20,
    verbose = 1,
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
)



