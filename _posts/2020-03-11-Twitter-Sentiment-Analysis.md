---
layout: post
title:  "Deep Learning Neural Network on 3D Voxelized Digits"
date:   2020-03-11 18:05:55 +0300
image:  '/assets/img/Twitter.jpg'
tags:   NPL
---
This project is a quick way to aggregate tweets and perform sentiment analysis. The goal of this project was to learn the overall sentiment (positive, negative, or neutral) of a particular topic on Twitter. Utilizing the official Twitter API and several natural language processing libraries such as TextBlob, we are able to gain insight into general public sentiment quickly and efficiently. Senitment Analysis is a huge field of natural language processing research. The advances in the field are utilized in a wide variety of industries from marketing to finance. 

# Introduction
For our implementation, we utilize a 3D convolutional neural network trained on 3D volumetric data. Our data comes from raw point clouds that have been transformed to 3D voxels for easy input into our CNN architecture. The performance of our 3D CNN is compared in terms of accuracy and performance to a similar 2D architecture on the regular 2D MNIST dataset.

# MNIST Dataset
The Modified National Institute of Standards and Technology (MNIST) database is a large database of handwritten digit images. MNIST was created by combining 30,000 images from NIST’s Special Database 3 (SD-3) and 30,000 images from Special Database 1 (SD-1). SD-3 images were collected among Census Bureau employees and SD-1 images were collected among American high-school students. The original NIST images were preprocessed through size-normalization and centered in a 28x28 image. Human performance of digit recognition upon the MNIST dataset hovers around a 2-2.5% test error rate.

In 2004, the best test error rate reported on the MNIST dataset was 0.42 percent. This was achieved by a team of researchers utilizing a neural classifier Limited Receptive Area (LIRA) for image recognition. Following landmark achievements within AI in 2015, a team of researchers using a single convolutional neural network reduced the test error rate to 0.31 percent. As it stands, the University of Virginia holds the best test error rate of 0.18 percent by simultaneously stacking fully connected, recurrent, and convolutional neural networks together simultaneously.

While the MNIST dataset is traditionally used in academia to benchmark new learning algorithms, it is gaining popularity among computer vision beginners and practitioners to learn different deep learning techniques on a clean pre-processed dataset.

<p align="center">
  <img src="/assets/img/training_set_images.jpg" />
</p>

The dataset used for implementing our CNN classifier is the 3D MNIST dataset. This dataset contains 3D point clouds generated from the original images of the MNIST dataset. The entire dataset is stored as a 4096-D vectors on an H5 file. These vectors were obtained through the voxelization of the raw 3D point clouds. 

# Preprocessing
To get our data ready for use in a 3D convolution, we must first transform the data into a four-dimensional shape. The four dimensions of our data are classified by: length, width, height, and color channel RGB. Finally, we convert all our target variables to binary class matrix utilizing TensorFlow’s to_categorical method. This method takes the value of 10 for the number of classes parameter because we are encoding the labels from the digits 0 through 9. After this is completed, our data is ready for input into our 3D CNN. 

# Modeling
To build our convolutional neural network we will utilize the Keras Sequential API. This API allows the user to define each layer of the model’s architecture individually starting from the input layer. Our model’s architecture is based off a Kaggle notebook by Ghouzam in [13]. Ghouzam’s model was able to achieve 98% classification accuracy upon the 2D MNIST training dataset and 99% classification accuracy on validation data. Our model incorporates a similar architecture like that is adjusted to handle 3D volumetric data. Ghouzam’s classification accuracy and loss results are displayed below. 

<p align="center">
  <img src="/assets/img/MNISTresults.jpg" />
</p>

The first layer of our model is the convolutional (Conv3D) layer. This layer creates a convolution kernel with the input to create a tensor of outputs. We will use 32 filters for the first two 3D convolution layers and 64 filters for the last two 3D convolution layers. These two pairs of filter layers each capture different characteristics by transforming parts of the image using a kernel filter. The kernel filter we utilize is a cube of size (5, 5, 5) for the first two layers and a similar cube of size (3, 3, 3) for the second pair of layers. Within each convolution layer we utilize the rectified linear activation function (ReLU). ReLU is one of the most commonly used activation functions in deep learning. The function outputs only a positive input. If the input is negative it returns 0. Despite the simplicity of ReLU, the function is effective in handling non-linearities and interaction events. 
	
Another important layer of our model’s architecture is the pooling layer. Maximum pooling, or MaxPool3D, is a pooling operation that acts as a down sampling filter.  Max pooling layers are used to reduce computational intensity and to reduce overfitting. Utilizing both convolution and max pooling layers allows our model to combine local features and learn global features of our 3D object. 

We also utilize a dropout regularization method to randomly drop a proportion of the networks data. Dropout layers allow our network to learn features in a distributed way which improves generalization and reduces overfitting. Our model uses multiple dropout methods between sets of filters and dense layers. Following some of our dropout layers, we incorporate the flatten layer to convert the final feature maps to a single vector. The flattening layer is needed to make use of convolution and max pool layers and to combine the local features. 

Finally, we call a dense layer with a SoftMax activation function. SoftMax outputs the probability distribution of each class to make the final decision. The full model’s architecture can be seen below.

<p align="center">
  <img src="/assets/img/model_plot.jpg" />
</p>

When we compile our model with the Keras compile method we see that our model consists of 1,247,722 total parameters. 

# Results

Now that our model’s architecture is created, we can begin training our CNN. Our model is trained on 30 epochs with a batch size of 128. We also split the training data into training and validation with a ratio of 0.3. After 30 epochs of training, our model achieves an 80.9% accuracy on training data and 73.5% accuracy on validation data as seen on figure 10. However, upon the training data, which is data that the model never sees, our accuracy drops to 73.9%. The full accuracy and loss results can be found below. 

<p align="center">
  <img src="/assets/img/training_validation_results.jpg" />
</p>

Although our model performs better than traditional machine learning methods like linear SVM, kernel SVM, and random forest, our model is not as accurate when compared to volumetric-based approaches like ShapeNet and VoxNet. One reasons for the discrepancy could be that this architecture was designed for the classification of digits in 2D space. Moreover, Ghouzam altered the training data by performing data augmentations to expand the handwritten digit dataset. These augmentations reduced the overfitting problem that our model suffers from.  Another reason for a low classification accuracy is due to information loss during the transformations of point clouds to 3D voxels. Methods that transform point clouds to voxels typically use spatial resolutions which underperform grid resolutions for processing data. This is likely to be insufficient for object classification tasks that require a large amount of attention to detail.

For further research, one should implement a more sophisticated approach. Given that our data was originally 3D point clouds, utilizing PointNet or PointNet++ would have yielded more effective results. The PointNet architectures take 3D point clouds as an input directly and therefore do not suffer from information loss and memory restrictions due to voxelization. 


# Where to find
This implementation relies on the 2D and 3D MNIST datasets. Due to the size of these files, the links to each of the datasets will be listed here as they are too large for upload to GitHub:
* [MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)
* [3D MNIST Dataset](hhttps://www.kaggle.com/daavoo/3d-mnist)

The link to the GitHub repository that stores our 3D notebook implementation can be found [here](https://github.com/brodyu/3D_Deep_Learning)


