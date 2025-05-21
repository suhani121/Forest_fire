import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Dataset paths
train_dir = r"C:\Users\HP\Downloads\Dataset\Dataset\Training and Validation"
test_dir = r"C:\Users\HP\Downloads\Dataset\Dataset\Testing"

# Parameters
image_size = (224, 224)
batch_size = 32

# Data Generators
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_gen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='training')
val_data = train_gen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='validation')

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(test_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')

# Load base model
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in base_model.layers:
    layer.trainable = False

# Add classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_data, validation_data=val_data, epochs=15, callbacks=[early_stop])

# Evaluation
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}")
