import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot

training_director="venv\\newdataset\\train"
validation_images='venv\\newdataset\\val'

training_data_generator = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    #vertical_flip=True,
    fill_mode='nearest')

training_images=training_data_generator.flow_from_directory(
    training_director,
    target_size=(180,180),
    color_mode='grayscale',
    class_mode='categorical'
)

validation_data_generator=ImageDataGenerator(
    rescale=1.0/255.0,
    fill_mode='nearest'
)

validation_augmented_images=validation_data_generator.flow_from_directory(
    validation_images,
    target_size=(180,180),
    color_mode='grayscale',
    class_mode='categorical'
)

print(training_images.class_indices)

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(1,6),activation='relu', input_shape=(180,180,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(6,6), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(6,6),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')  
])

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])
model.fit(training_images, validation_data=validation_augmented_images,epochs=50)
model.save('RPSModel.h5')


'''model=load_model('P:\RockPaperScissor-mini project\\venv\RPSModel.h5')
test_img='dataset\\test'
image_directory=os.listdir(test_img)
print(len(image_directory))
for image in image_directory[55:75]:
    image_path=os.path.join(test_img, image)
    im1=load_img(image_path,target_size=(180,180), color_mode='grayscale')
    im2=img_to_array(im1)/255.0
    im3=np.expand_dims(im2,axis=0)
    #print(im3.shape)
    predict=model.predict(im3)
    predicted_class=np.argmax(predict,axis=1)
    if predicted_class==training_images.class_indices['paper']:
        print("Paper")
    if predicted_class==training_images.class_indices['scissors']:
        print("Scissors")
    if predicted_class==training_images.class_indices['rock']:
        print("Rock")
    plt.imshow(im2.astype('uint8'), cmap='gray')
    pyplot.show()
    
    validation_data_generator = ImageDataGenerator(rescale = 1.0/255)
validation_image_directory="venv\\newdataset\\val"
validation_augmented_images = validation_data_generator.flow_from_directory(
    validation_image_directory,
    target_size=(180,180),
    color_mode='grayscale'
)
print(validation_augmented_images)
    
    '''