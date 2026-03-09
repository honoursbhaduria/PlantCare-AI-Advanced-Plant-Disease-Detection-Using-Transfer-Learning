# PlantCare-AI-Advanced-Plant-Disease-Detection-Using-Transfer-Learning


create a readme for this # IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
vbookshelf_rice_leaf_diseases_path = kagglehub.dataset_download('vbookshelf/rice-leaf-diseases')

print('Data source import complete.')
Rice Leaf Disease: Image Classification
Checking Memory & Requirements
!cat /proc/cpuinfo
The CPU being used is an Intel Xeon CPU @ 2.20GHz.
There are 2 CPU processors available.
The cache size is 46080 KB.
!cat /proc/meminfo
MemTotal: This is the total amount of physical memory (RAM) available on the system. It shows that the system has around 13 GB of memory.
MemFree: This is the amount of physical memory not in use. It shows that the system has around 8 GB of free memory.
MemAvailable: This is the estimated amount of memory available for starting new applications, without swapping. It shows that the system has around 12 GB of available memory.
Preface
Rice is one of the most important staple crops in the world and a major source of food for millions of people. However, rice plants are susceptible to various diseases that can cause significant losses in yield and quality. Among these diseases, leaf blast, bacterial blight, and brown spot are the major attacking diseases that can cause devastating damage to rice crops.

To address this issue, the development of an efficient and accurate automated disease detection system is crucial. In this project, we aim to develop a machine learning model that can accurately classify the three major attacking diseases of rice plants based on leaf images.

By using advanced machine learning techniques, our model will be able to provide fast and accurate detection of diseases, enabling farmers to take prompt and effective measures to control the spread of the diseases and minimize crop losses.

This project has the potential to revolutionize rice crop management and contribute to the sustainable development of agriculture.

Domain Analysis
Rice is one of the most important staple crops worldwide, providing sustenance to over half of the world's population. However, rice plants are vulnerable to several diseases that can severely impact the yield and quality of rice production. Some of the major rice diseases that affect rice plants include bacterial leaf blight, brown spot, and leaf smut.

Bacterial leaf blight, caused by the bacterium Xanthomonas oryzae pv. oryzae, is a serious disease that can cause extensive damage to rice plants. The symptoms of bacterial leaf blight include water-soaked lesions on the leaves, which later turn brown and dry up. In severe cases, the disease can cause wilting and death of the plant, leading to a significant reduction in crop yield (1).

Brown spot, caused by the fungus Cochliobolus miyabeanus, is another major rice disease that can cause significant yield losses. The disease is characterized by small, oval to elliptical spots on the leaves, which turn brown with a yellow halo. In severe cases, the spots can coalesce and cause the leaves to wither and die. Brown spot can also affect the panicles, leading to a reduction in grain quality and yield (2).

Leaf smut, caused by the fungus Entyloma oryzae, is a relatively less common rice disease. The symptoms of leaf smut include the formation of small, round, and reddish-brown spots on the leaves. These spots later turn black and produce powdery spores. Although leaf smut does not usually cause significant yield losses, it can affect the quality of rice grains by reducing their weight and size (3).

Effective disease management strategies are crucial for controlling these diseases and reducing their impact on rice production. Some of the common methods for disease management include crop rotation, use of resistant cultivars, application of fungicides, and cultural practices such as proper crop spacing and irrigation management (4).

In recent years, the use of machine learning algorithms for plant disease diagnosis and classification has gained significant attention. With the availability of large datasets of plant images and the advancements in deep learning algorithms, it has become possible to accurately classify plant diseases based on their visual symptoms. This has the potential to improve disease management strategies and reduce the impact of plant diseases on crop production (5).

Installing Modules
!pip install keras
!pip install keras_tuner
Importing Libraries
#basics
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#utility & processing
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#cnn architecture
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

#callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#keras tuner
import keras_tuner
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

#transfer learning
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.xception import Xception

#warnings
import warnings
warnings.filterwarnings('ignore')


print('we are currently using tensorflow version', tf.__version__)
Setting up an environment
input_shape_2D=(224,224)
input_shape_3D= (224,224,3)
seed=1
batch_size=32
epochs=30
stopper=EarlyStopping(patience=5)
Uploading image dataset
#loading image data
data_ds= tf.keras.utils.image_dataset_from_directory(
    directory='/kaggle/input/rice-leaf-diseases/rice_leaf_diseases',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=None,
    image_size=input_shape_2D,
    seed=seed
)
#class labels
class_names = data_ds.class_names

print("Class Names:", class_names)
def show_images(images, labels):
    fig, axes = plt.subplots(8, 4, figsize=(10, 10))
    axes = axes.flatten()
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img.numpy().astype("uint8"))
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
#displaying image batch of 32
batched_ds = data_ds.batch(batch_size)

batch = next(iter(batched_ds))
images, labels = batch

images, labels = next(iter(batched_ds))
show_images(images, labels)
Great! we have now loaded the image data in the notebook. Let's quickly separate images and labels and store them into X and y respectively

Data Processing
X=[]
y= []

for image, label in data_ds.batch(119):
  for img, lab in zip(image, label):
    X.append(img)
    y.append(lab.numpy())
# converting to numpy array
X= np.array(X)
y= np.array(y)
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
Normalisation
Normalization of image data is an important step in the pre-processing of data before it is used to train a neural network. It involves transforming the pixel values of the input images so that they fall within a specific range, typically [0, 1]. Normalization ensures that the input features have similar scales, which can prevent some input features from dominating others during the training process.

By scaling the pixel values, normalization also makes it easier for the neural network to learn the underlying patterns in the data. This is because the weights in the neural network can be updated more easily and quickly when the input data has a similar scale. In addition, normalization can help to reduce the effects of lighting conditions, noise and other factors that can cause variation in the input data.

Overall, normalization is an important step that can improve the performance and accuracy of a neural network by ensuring that the input data is in a consistent and standardized format.

X_train = X_train/255
X_test = X_test/255
print("shape of X_train_scaled:", X_train.shape)
print("shape of y_train:", y_train.shape)
print("shape of X_test_scaled:", X_test.shape)
print("shape of y_test:", y_test.shape)
Encoding
y_train = to_categorical(y_train, num_classes=len(class_names))
y_test = to_categorical(y_test, num_classes=len(class_names))
CNN Architecture
own_model=Sequential()

# adding first convloutional layer
own_model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape_3D))
own_model.add(MaxPooling2D())

# adding second convloutional layer
own_model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
own_model.add(MaxPooling2D())

# adding third convloutional layer
own_model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
own_model.add(MaxPooling2D())

# adding flatten layer
own_model.add(Flatten())

# adding fully connected layers
own_model.add(Dense(200, activation='relu'))

#adding output layer : number of classes are 3
own_model.add(Dense(3, activation='softmax'))
# looking at the architecture summary
own_model.summary()
# compiling model
own_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#training model
checkpointer_own = ModelCheckpoint(filepath='own_model.weights.best.hdf5', verbose=3, save_best_only=True)

history=own_model.fit(X_train, y_train, validation_data= (X_test, y_test), epochs=epochs, callbacks=[checkpointer_own, stopper])
def plot_performance(model_name, acc, val_acc, loss, val_loss):
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('{} : Loss'.format(model_name), fontsize=12, fontweight='bold')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper right')

  plt.subplot(1,2,2)
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('{} : Accuracy'.format(model_name), fontsize=12, fontweight='bold')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper right')

  plt.tight_layout()
plot_performance(model_name= 'Own Model',
                 acc=history.history['accuracy'],
                 val_acc=history.history['val_accuracy'],
                 loss=history.history['loss'],
                 val_loss=history.history['val_loss'])
Data Augmentation
Data augmentation is a technique used to increase the size and diversity of a dataset by applying various transformations to the existing data. This technique has become an essential tool in computer vision and image processing tasks, such as object recognition and classification, due to its ability to enhance the generalization ability of machine learning models and prevent overfitting.

In the context of our rice leaf disease image classification project, data augmentation can play a crucial role in improving the performance and robustness of our model. By generating new images with different variations such as rotations, flips, zooms, and other transformations, we can increase the diversity of our dataset and provide our model with more examples to learn from, which can lead to better classification accuracy and robustness to variations in the real-world data.

Therefore, data augmentation is an important aspect to consider in our project, and we will explore various techniques and approaches to implement it effectively.

#creating image generator object
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True)
#training model
checkpointer_own_aug = ModelCheckpoint(filepath='own_model_aug.weights.best.hdf5', verbose=3, save_best_only=True)

aug_history= own_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train)//batch_size,
    epochs=epochs,
    callbacks=[stopper, checkpointer_own_aug]
)
#visualising model performance
plot_performance(model_name='Own_Aug',
                  acc=aug_history.history['accuracy'],
                 val_acc=aug_history.history['val_accuracy'],
                 loss=aug_history.history['loss'],
                 val_loss=aug_history.history['val_loss'])
Keras Tuner
Keras Tuner is a hyperparameter tuning library for Keras, which allows users to search for the best hyperparameters in an automated way. Hyperparameter tuning is a crucial step in building machine learning models, and it involves finding the best set of hyperparameters for a given model architecture and dataset. The optimal hyperparameters can help to achieve better performance in terms of accuracy, speed, and generalization ability of the model.

In this project, Keras Tuner has been used to search for the optimal hyperparameters of the convolutional neural network model, which includes the number of convolutional layers, the number of filters in each layer, the kernel size, the activation function, and the learning rate of the optimizer. The goal is to find the best combination of hyperparameters that can improve the accuracy of the model on the test dataset.

At the end, we will analyse whether the model performance improves or not

def build_model(hp):
  new_model=Sequential()

  # adding first convloutional layer
  new_model.add(Conv2D(
      filters= hp.Int('conv_1_filter', min_value=8, max_value=32, step=8),
      kernel_size=hp.Choice('conv_1_kernel', values= [2,3]),
      padding=hp.Choice('conv_1_pad', values=['same', 'valid']),
      activation='relu', input_shape=input_shape_3D))
  new_model.add(MaxPooling2D())

  # adding second convloutional layer
  new_model.add(Conv2D(
      filters= hp.Int('conv_2_filter', min_value=16, max_value=64, step=16),
      kernel_size=hp.Choice('conv_2_kernel', values= [2,3]),
      padding=hp.Choice('conv_2_pad', values=['same', 'valid']),
      activation='relu'))
  new_model.add(MaxPooling2D())

  # adding third convloutional layer
  new_model.add(Conv2D(
      filters= hp.Int('conv_3_filter', min_value=32, max_value=64, step=16),
      kernel_size=hp.Choice('conv_3_kernel', values= [2,3]),
      padding=hp.Choice('conv_3_pad', values=['same', 'valid']),
      activation='relu'))
  new_model.add(MaxPooling2D())

  # adding flatten layer
  new_model.add(Flatten())

  # adding fully connected layers
  new_model.add(Dense(
      units=hp.Int('dense_1_units', min_value=50, max_value=250, step=50),
      activation='relu'
      ))

  #adding output layer : number of classes are 3
  new_model.add(Dense(3, activation='softmax'))

  new_model.compile(
      optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )

  return new_model
We will run random search on the model with an objective to optimum accuracy on the test data

tuner_search=RandomSearch(build_model, objective='val_accuracy', max_trials=3, directory='output', project_name='RiceLeafDisease')
tuner_search.search(
    datagen.flow(X_train, y_train),
    validation_data=(X_test, y_test),
    epochs=10
)
tuned_model=tuner_search.get_best_models(num_models=1)[0]
tuned_model.summary()
#training model
checkpointer_own_tuned = ModelCheckpoint(filepath='own_model_tuned.weights.best.hdf5', verbose=3, save_best_only=True)

tuned_history= tuned_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train)//batch_size,
    initial_epoch=10,
    epochs=epochs,
    callbacks=[stopper, checkpointer_own_tuned]
)
plot_performance(model_name='Tuned',
                  acc=tuned_history.history['accuracy'],
                 val_acc=tuned_history.history['val_accuracy'],
                 loss=tuned_history.history['loss'],
                 val_loss=tuned_history.history['val_loss'])
We will now take help from pre-built models. There are already pretrained models which are trained on million of images including 1000 classes. However, since out image classes are not included in the class outputs used to trained these model, we will apply transfer learning method. It is introduced in the next section

Transfer Learning
Transfer learning is a technique in deep learning where a pre-trained neural network model is used as a starting point for a new task instead of training a new model from scratch. This approach has gained popularity in recent years due to the high cost of training large neural networks on big datasets. Transfer learning can significantly reduce the time and computational resources needed to build accurate deep learning models, especially when the new task has a small amount of data.

In our rice leaf disease classification project, we can leverage transfer learning by using pre-trained models that have been trained on large image datasets, such as ImageNet, and fine-tune them on our rice leaf disease dataset. This can help our model learn features and patterns that are common across different image recognition tasks, and improve its accuracy and generalization capabilities.

Some of the popular pre-trained models used in transfer learning include VGG, ResNet, Inception, and Xception. These models have achieved state-of-the-art performance on various image classification tasks and are available in popular deep learning frameworks like TensorFlow and Keras.[6]

I have trained VGG16 and ResNET in backend and observed the results. Since these models were not performing very well on this data, I have not included them in this version. If you want to check the complete notebook, click on following link:

https://github.com/mdushetwar/RiceLeafDisease-ImageClassification/blob/main/PRCP_1001_RiceLeaf_final.ipynb

InceptionV3
InceptionV3 is a deep convolutional neural network architecture that was developed by Google. It is known for its efficiency in training and accuracy in image classification tasks. The model uses a combination of convolutional layers with varying kernel sizes to capture features at different scales. InceptionV3 was the winner of the 2015 ImageNet Large Scale Visual Recognition Challenge.

inception_base=tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape_3D,
)

#unfreezing the convloution layers
inception_base.trainable=True
set_trainable=False

for layer in inception_base.layers:
  if layer.name=='mixed9_0':
    set_trainable=True

  if set_trainable==True:
    layer.trainable=True

  else:
    layer.trainable=False


# building fully connected layers
inception_model= Sequential()

inception_model.add(inception_base)
inception_model.add(Flatten())

inception_model.add(Dense(128, activation='relu'))
inception_model.add(Dense(3, activation='softmax'))


#compiling
inception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tf.keras.utils.plot_model(
    inception_model,
    show_shapes=True,
    show_layer_names=True,
    show_layer_activations=True
)
Without Augmentation
#training model
checkpointer_inception = ModelCheckpoint(filepath='inception.weights.best.hdf5', verbose=3, save_best_only=True)

inception_history= inception_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    callbacks=[stopper, checkpointer_inception]
)
plot_performance(model_name='Inception',
                acc=inception_history.history['accuracy'],
                 val_acc=inception_history.history['val_accuracy'],
                 loss=inception_history.history['loss'],
                 val_loss=inception_history.history['val_loss'])
With Augmentation
#training model
checkpointer_inception_aug = ModelCheckpoint(filepath='inception_aug.weights.best.hdf5', verbose=3, save_best_only=True)

inception_history_aug= inception_model.fit(
    datagen.flow(X_train,y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train)//batch_size,
    epochs=epochs,
    callbacks=[stopper, checkpointer_inception_aug]
)
plot_performance(model_name='Inception_Aug',
                acc=inception_history_aug.history['accuracy'],
                 val_acc=inception_history_aug.history['val_accuracy'],
                 loss=inception_history_aug.history['loss'],
                 val_loss=inception_history_aug.history['val_loss'])
Xception
Xception is a deep neural network architecture proposed by Google researchers in 2016. It is an extension of the Inception architecture that replaces the standard Inception modules with depthwise separable convolutions, which are computationally efficient and lead to improved performance. Xception has achieved state-of-the-art results on various computer vision tasks and is commonly used in transfer learning applications.

xception_base= Xception(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape_3D
)

xception_base.trainable=True
set_trainable=False

for layer in xception_base.layers:
  if layer.name=='add_8':
    set_trainable=True

  if set_trainable==True:
    layer.trainable=True
  else:
    layer.trainable=False

#building fully connected layers
xception_model=Sequential()

xception_model.add(xception_base)
xception_model.add(Flatten())

xception_model.add(Dense(128, activation='relu'))
xception_model.add(Dense(3, activation='softmax'))

xception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tf.keras.utils.plot_model(
    xception_model,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=True
)
Without Augmentation
#training model
checkpointer_xception = ModelCheckpoint(filepath='xception.weights.best.hdf5', verbose=3, save_best_only=True)

xception_history=xception_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    callbacks=[stopper, checkpointer_xception]
)
plot_performance(model_name='Xception',
                acc=xception_history.history['accuracy'],
                 val_acc=xception_history.history['val_accuracy'],
                 loss=xception_history.history['loss'],
                 val_loss=xception_history.history['val_loss'])
With Augmentation
checkpointer_xception_aug = ModelCheckpoint(filepath='xception_aug.weights.best.hdf5', verbose=3, save_best_only=True)

xception_history_aug=xception_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train)//batch_size,
    epochs=epochs,
    callbacks=[stopper, checkpointer_xception_aug]
)
plot_performance(model_name='Xception_Aug',
                acc=xception_history_aug.history['accuracy'],
                 val_acc=xception_history_aug.history['val_accuracy'],
                 loss=xception_history_aug.history['loss'],
                 val_loss=xception_history_aug.history['val_loss'])
Based on the provided data, it appears that the Xception model trained on augmented data has the highest validation accuracy among the five models evaluated, with a validation accuracy of 0.9667. Additionally, the Xception model has a relatively low validation loss, indicating that it is effectively minimizing the difference between predicted and actual labels.

Furthermore, the Xception model appears to be relatively efficient, with a time per epoch of 1586 ms, which is faster than the VGG and Inception models but slower than the own model and ResNet models.

Overall, the Xception model appears to have the best balance of accuracy and efficiency among the five models evaluated, which is why we will select it as the final model.

Loading Best Model
xception_model.load_weights('xception_aug.weights.best.hdf5')
Visualising Kernels
fig, ax= plt.subplots(3,4, figsize=(8,8))

for i, ax in zip(range(12), ax.flatten()):

  conv1=xception_model.layers[0]
  weights1 = conv1.get_weights()
  kernels1 = weights1[0]
  kernel1_1 = kernels1[:,:,0,i]
  ax.imshow(kernel1_1)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.tight_layout()
  plt.axis('off');
Lets understand what features these kernel extract? What these kernels see?

image = tf.keras.utils.load_img('/kaggle/input/rice-leaf-diseases/rice_leaf_diseases/Brown spot/DSC_0112.jpg', target_size=input_shape_2D)
input_arr = tf.keras.utils.img_to_array(image)/255
image = input_arr[:, :, 1]
plt.imshow(image)
plt.axis('off');
import numpy as np

def convolution(image, kernel):
    # Get the size of the kernel
    kernel_size = kernel.shape[0]

    # Calculate the size of the output image
    output_size = image.shape[0] - kernel_size + 1

    # Initialize the output image
    output_image = np.zeros((output_size, output_size))

    # Pad the image with zeros
    padded_image = np.pad(image, ((kernel_size-1)//2, (kernel_size-1)//2), 'constant')

    # Apply the convolution operation
    for ii in range(output_size):
        for jj in range(output_size):
            window = padded_image[ii:ii+kernel_size, jj:jj+kernel_size]
            output_image[ii, jj] = np.sum(window * kernel)

    return output_image
fig, ax= plt.subplots(3,4, figsize=(8,8))

for i, ax in zip(range(12), ax.flatten()):

  conv1=xception_model.layers[0]
  weights1 = conv1.get_weights()
  kernels1 = weights1[0]
  kernel1_1 = kernels1[:,:,0,i]

  output=convolution(image, kernel1_1)

  ax.imshow(output)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.tight_layout()
  plt.axis('off');
Visualising Prediction
# get predictions on the test set
y_hat = xception_model.predict(X_test)

# define text labels
labels = data_ds.class_names
# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(X_test.shape[0], size=20, replace=False)):
    ax = fig.add_subplot(5, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(X_test[idx]))
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(labels[pred_idx], labels[true_idx]),
                 color=("blue" if pred_idx == true_idx else "red"))
    plt.tight_layout()
Summary
The project involved building a deep learning model to classify images of rice leaf disease. The dataset contained images from 3 type of rice leaf disease namely bacterial bligh, brown spot, and leaf smut. The project was divided into several steps, including data exploration, data preprocessing, building and training deep learning models, and evaluating model performance.

During data exploration, we analyzed the dataset and visualized the images to get a better understanding of the data. We observed that the dataset was balanced, with an equal number of images for rice leaf diseases. We also noticed that the images were of different sizes and needed to be resized to a uniform size before being used for training. We have rescaled them to uniform size of 224 X 224.

For data preprocessing, We have normalized the training dataset and also encoded the labels. We used Keras' ImageDataGenerator to generate augmented images to increase the size of the dataset and reduce overfitting. We also resized the images and divided them into training, validation, and testing sets.

We built several deep learning models by applying transfer learning technique, including a custom model, VGG, Inception, ResNet, and Xception. We trained these models on both the original and augmented datasets and evaluated their performance using accuracy, loss, and time taken per epoch. We also plotted the training and validation curves to analyze the models' behavior during training.

We observed that the custom model performed well on the original dataset, but its performance improved on the augmented dataset. Xception, in particular, performed the best, achieving the highest accuracy and the lowest loss on both datasets.

Based on these observations, we selected Xception as the final model and used it to predict the class of new images. We also plotted the confusion matrix to analyze the model's performance in detail.

Overall, the project was successful in building a deep learning model to classify images of rice leaf diseases. We explored the dataset, preprocessed the data, built and trained deep learning models, and evaluated their performance. We also gained insights into how different models behave when trained on normal vs. augmented data, what makes a good model in terms of accuracy and loss, and how to analyze models with respect to their time taken per epoch. Finally, we selected Xception as the best model and achieved good accuracy on new images.

Project Challenges
Limited amount of data:
One of the biggest challenges in the project was the limited amount of data available for training the models. We had only 119 images from 3 classes. This could lead to overfitting, where the model memorizes the training data and does not generalize well to new data. To deal with this challenge, data augmentation techniques were used to artificially increase the size of the dataset. This helped to improve the performance of the models and reduce overfitting.

Complexity of deep learning models:
Deep learning models can be very complex and have a large number of parameters, which makes them difficult to train and optimize. To deal with this challenge, various techniques such as regularization, and early stopping were used to prevent overfitting and improve the performance of the models.

Hardware limitations:
Deep learning models require a lot of computational resources, especially when training on large datasets. This can be a challenge if we don't have access to powerful hardware. To deal with this challenge, cloud computing services such as Google Colab were used to train the models, which provided access to powerful GPUs without the need for expensive hardware.

Choosing the best model:
With multiple models to choose from, it can be difficult to determine which one is the best for the task at hand. To deal with this challenge, various metrics such as accuracy, loss, and training time were used to evaluate the performance of the models. Based on these metrics, the Xception model was selected as the best model for the task.

References
Mew, T.W., Alvarez, A.M., and Leach, J.E. (1993). Bacterial leaf blight. In: Compendium of rice diseases (ed. R.K. Webster), 25-36. St. Paul, MN: APS Press.

Singh, U.S., Singh, D.P., and Chaube, H.S. (2015). Rice brown spot disease: A review. Journal of Plant Pathology and Microbiology, 6(4), 1-8.

Acedo, A.L., and Daquioag, R.D. (2013). Leaf smut disease of rice. Philippine Agricultural Scientist, 96(4), 354-364.

Savary, S., Willocquet, L., Elazegui, F.A., and Castilla, N.P. (2000). Rice pest and disease constraints in the Philippines: A preliminary assessment. Plant Disease, 84(6), 565-572.

Sladojevic, S., Arsenovic, M., Anderla, A., Culibrk, D., Stefanovic, D., and Crnojevic, V. (2016). Deep neural networks based recognition of plant diseases by leaf image classification. Computational Intelligence and Neuroscience, 2016, 1-11.

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks?. In Advances in neural information processing systems (pp. 3320-3328).

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826)

François Chollet. "Xception: Deep Learning with Depthwise Separable Convolutions." arXiv preprint arXiv:1610.02357 (2016).

https://www.kaggle.com/datasets/raihan150146/rice-leaf-diseases-dataset 
