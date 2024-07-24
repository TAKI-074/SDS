# SDS
pip install tensorflow pandas scikit-learn

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


base_dir = r"C:\Users\yogit\OneDrive\Desktop\sds---try"  
train_csv_path = os.path.join(base_dir, 'train.csv')
train_image_dir = os.path.join(base_dir, 'train_dataset')
test_image_dir = os.path.join(base_dir, 'test_dataset')


train_labels = pd.read_csv(train_csv_path)
train_imgs, val_imgs = train_test_split(train_labels, test_size=0.2, random_state=42)


train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=20, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_imgs, 
                                                    directory='train_dataset', 
                                                    x_col='File Name', 
                                                    y_col='Class', 
                                                    target_size=(224, 224), 
                                                    batch_size=32, 
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_dataframe(val_imgs, 
                                                directory='train_dataset', 
                                                x_col='File Name', 
                                                y_col='Class', 
                                                target_size=(224, 224), 
                                                batch_size=32, 
                                                class_mode='categorical')


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_labels['Class'].unique()), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])



train_generator = train_datagen.flow_from_dataframe(
    train_imgs,
    directory=train_image_dir,
    x_col='File Name',  
    y_col='Class',       
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_imgs,
    directory=train_image_dir,
    x_col='File Name',  
    y_col='Class',       
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("Classes in train_generator:", train_generator.class_indices)
print("Classes in val_generator:", val_generator.class_indices)


history = model.fit(train_generator, validation_data=val_generator, epochs=5)


model.save('model.h5')


test_image_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

test_df = pd.DataFrame({'image_name': test_image_files})

print("Test DataFrame head:\n", test_df.head())


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_image_dir,
    x_col='image_name',
    y_col=None,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)


predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=-1)


filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predicted_classes})
results.to_csv("test_predictions.csv", index=False)


output_path = 'test_predictions.csv'

results.to_csv(output_path, index=False)

if os.path.exists(output_path):
    print(f"File saved successfully at {output_path}")
else:
    print(f"Failed to save file at {output_path}")


for filename, prediction in zip(filenames[:10], predicted_classes[:10]):  
    print(f"Filename: {filename}, Prediction: {prediction}")
    
