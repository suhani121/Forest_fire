from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = train_datagen.flow_from_directory(
    'forest fire',  # path to dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training')

val_gen = train_datagen.flow_from_directory(
    'forest fire',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Load pretrained ResNet50 model + custom head
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save the model
model.save("deepfire_resnet50.h5")
