import numpy

import keras
from keras import layers
from keras import models
from keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
import os

# def plt_modle(model_hist):
#     acc = model_hist.history['acc']
#     val_acc = model_hist.history['val_acc']
#     loss = model_hist.history['loss']
#     val_loss = model_hist.history['val_loss']
#
#     epochs = range(1, len(acc) + 1)
#
#     plt.figure(figsize=(15, 6));
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
#     plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend(loc='best')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
#     plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend(loc='best')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#
#     plt.show()
#
# # Split images into Training and Validation Sets (20%)
#
# train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)
#
# img_size = 128
# batch_size = 20
# t_steps = 3462/batch_size
# v_steps = 861/batch_size
classes = 5
# flower_path = "C:/Users/User/PycharmProjects/flowers/flowers/flowers"
# train_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
# valid_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')
#
# # Model
#
# model = models.Sequential()
#
#
#
# # use model.add() to add any layers you like
# # read Keras documentation to find which layers you can use:
# #           https://keras.io/layers/core/
# #           https://keras.io/layers/convolutional/
# #           https://keras.io/layers/pooling/
# #
#
# #add model layers
#
# model.add(Conv2D(16, kernel_size=3, activation='relu',input_shape=((128,128,3))))
# model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
# model.add(Conv2D(32, kernel_size=5, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
# model.add(Flatten())
# model.add(layers.Dense(128, activation='relu'))
#
#
# # last layer should be with softmax activation function - do not change!!!
# model.add(layers.Dense(classes, activation='softmax'))
#
# # fill optimizer argument using one of keras.optimizers.
# # read Keras documentation : https://keras.io/models/model/
# optimizer ='adam'
#
# # fill loss argument using keras.losses.
# # reads Keras documentation https://keras.io/losses/
# loss ='categorical_crossentropy'
# model.compile(loss= loss ,optimizer=optimizer ,metrics=['accuracy'])
#
# # you can change number of epochs by changing the value of the 'epochs' paramter
# model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 30 , validation_data=valid_gen, validation_steps=v_steps)
# model.save('flowers_model.h5')
# plt_modle(model_hist)


def load_trained_model(old_model):  #gets H5 file to load
    return load_model(old_model)

def predict(model,path):
    test_image = []
    names = []
    for r, d, f in os.walk(path):
        for file in f:
            img = image.load_img(os.path.join(r, file), target_size=(128, 128, 3), grayscale=False)
            img = image.img_to_array(img)
            img = img / 255
            test_image.append(img)
            names.append(file)
    test = numpy.array(test_image)
    answer = model.predict_classes(test)
    finalAnswer = []
    classifyName = ['daisy','dandelion','rose','sunflower','tulip']
    for i in range(len(test_image)):
        finalAnswer.append(classifyName[answer[i]])
    createFinalFile(names, finalAnswer)
    return names,finalAnswer

def createFinalFile(imageNames, classifications):
    with open("predictionAnswer.csv","w+") as file:
        for i in range(len(imageNames)):
            file.write(imageNames[i]+","+ str(classifications[i])+"\n")
            #print(imageNames[i]+","+str(classifyName[classifications[i]]))

        file.close()

model = load_trained_model('flowers_model.h5')
predict(model,"C:/Users/User/PycharmProjects/flowers/flowers/shit")
#model.predict_classes()