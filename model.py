import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import cv2

######################## Data Loading ########################
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_training_data('./chest_xray/train/')
test = get_training_data('./chest_xray/val/')
val = get_training_data('./chest_xray/test/')

x_train = [] # image
y_train = [] # label

x_val = [] # image
y_val = [] # label

x_test = [] # image
y_test = [] # label

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
    
######################## Data Visualization ########################
def xray_image():
    xray_1 = plt.subplot(1,2,1)
    xray_1.imshow(train[-1][0], cmap='gray')
    xray_1.set_title("X-ray of Normal Lungs")
    
    xray_2 = plt.subplot(1,2,2)
    xray_2.imshow(train[0][0], cmap='gray')
    xray_2.set_title("X-ray of Lungs with Pnuemonia")
    
xray_image()

######################## Data Normalization & Resizing ########################
# Normalization
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# Resizing data
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train).reshape((-1,1))

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val).reshape((-1,1))

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test).reshape((-1,1))

######################## Data Augmentation ########################
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

######################## Model ########################
model = Sequential([
    Conv2D(input_shape=(150, 150, 1), filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.1),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    
    Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.1),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    Flatten(),
    Dense(units=128, activation="relu", kernel_regularizer='l1'),
    Dropout(0.3),
    Dense(units=2, activation="sigmoid")
    ])

#opt = keras.optimizers.Adam()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',  # Reduces learning rate by factor of 0.5
                                            patience = 5,             # if the validation accuracy doesn't change for 5 epochs
                                            verbose=1,factor=0.3, 
                                            min_lr=0.000001) 

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()

######################## Training ########################
epochs = 20
batch_size = 32
history = model.fit(datagen.flow(x_train,y_train, batch_size = batch_size),
                    epochs = epochs , validation_data = datagen.flow(x_val, y_val),
                    callbacks = [learning_rate_reduction])

######################## Evaluate ########################
model.evaluate(x_test, y_test, verbose="auto")

######################## Plot ########################
loss = history.history['loss']
val_loss = history.history['val_loss'] 
epochs = range(0,20)


plt.plot(epochs, val_loss, 'r')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(epochs, loss, 'y')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(loc = 'lower right')
plt.show()
