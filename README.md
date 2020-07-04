# ~10 mini projects in machine learning

## 1. Introduction to Supervised Learning using Iris data

In this project, I use Logistic Regression, K-Nearest Neighbors algorithm, and the Support Vector Machine algorithm to analyze iris data.

## 2. Introduction to Unsupervised Learning using K-Means Clustering

In this project, I implement the K-Means Clustering algorithm using self-generated data.

## 3. Board Game Review Prediction

Reviews can make or break a product; as a result, many companies take drastic measures to ensure that their product receives good reviews. When it comes to board games, reviews and word-of-mouth are everything. In this project, I use a **linear regression** model as well as a **random forest regressor** model to predict the average review a board game will receive based on characteristics such as minimum and maximum number of players, playing time, complexity, etc.

I clone a GitHub repository that contains the data set I use. This can be accomplished by simply downloading the repository using the link provided:

`git clone https://github.com/ThaWeatherman/scrapers.git`

## 4. A Deep Reinforcement Algorithm in an OpenAI Gym Environment

In this project, I built a deep neural network and used reinforcement learning to solve a cart and pole balancing problem using OpenAI. OpenAI Gym is a tookit for developing and comparing reinforcement learning algorithms that was built by OpenAI, a non-profit artificial intelligence research company founded by Elon Musk and Sam Altman.

### Installing and Loading Libraries

The first step is to download and install all the dependencies. To install OpenAI Gym, I used Git.

Using Git, download and install OpenAI easily using the following commands:

`git clone https://github.com/openai/gym`
`cd gym`
`pip3 install -e . # minimal install`

This downloads the bare minimums for the OpenAI Gym environment. OpenAI Gym contains a wide variety of environments, even Atari games. Some of these environments will need additional dependencies installed.

The following python libraries are also needed:

- NumPy
- Keras
- Theano

Install them using conda or pip.

## 5. Credit Card Fraud Detection

Throughout the financial sector, machine learning algorithms are being developed to detect fraudulent transactions. In this project, that is exactly what I did as well. Using a dataset of of nearly 28,500 credit card transactions and multiple unsupervised anomaly detection algorithms, I identify transactions with a high probability of being credit card fraud. I build and deploy the following two machine learning algorithms:

- Local Outlier Factor (LOF)
- Isolation Forest Algorithm

Furthermore, using metrics suchs as precision, recall, and F1-scores, I investigate why the classification accuracy for these algorithms can be misleading.

In addition, I explore the use of data visualization techniques common in data science, such as parameter histograms and correlation matrices, to gain a better understanding of the underlying distribution of data in my data set.

## 6. Getting Started With Natural Language Processing in Python

Topics covered in this project are:

- Tokenizing - Splitting sentences and words from the body of text.
- Part of Speech tagging
- Chunking

I use the Natural Language Toolkit (NLTK) which is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language.

To install it, simply type the following in your terminal:

`pip install nltk`

or

`pip3 install nltk`

Lastly, in the notebook `movie_review_classification`, I train and test the accuracy of predicting whether a movie review is positive or negative using a Support Vector Classifier.

## 7. Object Recognition

In this project, I deploy a convolutional neural network (CNN) for object recognition. More specifically, I use the All-CNN network published in the 2015 ICLR paper, "Striving For Simplicity: The All Convolutional Net". This paper can be found at the following link:

https://arxiv.org/pdf/1412.6806.pdf

This convolutional neural network obtained state-of-the-art performance at object recognition on the CIFAR-10 image dataset in 2015. I build this model using Keras, a high-level neural network application programming interface (API) that supports both Theano and Tensorflow backends. Either backend can be used; however, I will be using Theano.

In this project, I do the following:

- Import datasets from Keras
- Use one-hot vectors for categorical labels
- Add layers to a Keras model
- Load pre-trained weights
- Make predictions using a trained Keras model

The dataset use is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## 8. Using The Super Resolution Convolutional Neural Network for Image Restoration

The goal of super-resolution (SR) is to recover a high resolution image from a low resolution input, or as they might say on any modern crime show, **enhance!**

To accomplish this goal, I deploy the super-resolution convolution neural network (SRCNN) using Keras. This network was published in the paper, "Image Super-Resolution Using Deep Convolutional Networks" by Chao Dong, et al. in 2014. You can read the full paper at https://arxiv.org/abs/1501.00092.

As the title suggests, the SRCNN is a deep convolutional neural network that learns end-to-end mapping of low resolution to high resolution images. As a result, we can use it to improve the image quality of low resolution images. To evaluate the performance of this network, use three image quality metrics: peak signal to noise ratio (PSNR), mean squared error (MSE), and the structural similarity (SSIM) index.

Furthermore, I use OpenCV, the Open Source Computer Vision Library. OpenCV was originally developed by Intel and is used for many real-time computer vision applications. In this particular project, I use it to pre and post process my images. I frequently be convert my images back and forth between the RGB, BGR, and YCrCb color spaces. This is necessary because the SRCNN network was trained on the luminance (Y) channel in the YCrCb color space.

During this project, I do the following:

- Use the PSNR, MSE, and SSIM image quality metrics,
- Process images using OpenCV,
- Convert between the RGB, BGR, and YCrCb color spaces,
- Build deep neural networks in Keras,
- Deploy and evaluate the SRCNN network

## 9. Natural Language Processing for Text Classification with NLTK and Scikit-learn

In this project, I explore different ways to improve text classification results. I use the following to text spam text messages:

- Regular Expressions
- Feature Engineering
- Multiple scikit-learn Classifiers
- Ensemble Methods

## 10. K Means Clustering for Imagery Analysis

In this project, I use a K-means algorithm to perform image classification. Clustering isn't limited to the consumer information and population sciences, it can be used for imagery analysis as well. Leveraging Scikit-learn and the MNIST dataset, I investigate the use of K-means clustering for computer vision.

In this project, I do the following:

- Preprocess images for clustering
- Deploy K-means clustering algorithms
- Use common metrics to evaluate cluster performance
- Visualize high-dimensional cluster centroids

## 11. Data Compression and Visualization using Principle Component Analysis (PCA)

This project focuses on mapping high dimensional data to a lower dimensional space, a necessary step for projects that utilize data compression or data visualizations. As the ethical discussions surrounding AI continue to grow, scientists and businesses alike are using visualizations of high dimensional data to explain results.

During this project, I perform K-Means clustering on the well known Iris data set, which contains 3 classes of 50 instances each, where each class refers to a type of iris plant. To visualize the clusters, I use principle component analysis (PCA) to reduce the number of features in the dataset.

I do the following:

- The KMeans Clustering Elbow Method
- Principle Component Analysis with Scikit-Learn
- Meshgrid Visualizations for PCA-reduced Data
