#' ---
#' title: "Convolutional neural network"
#' author: Brett Melbourne
#' date: 04 Mar 2024 (updated 26 Feb 2026)
#' output:
#'     github_document
#' ---

#' To demonstrate a CNN, we are using a standard benchmark dataset, CIFAR100 but
#' subsetted to images in ecological categories. These are small images, so
#' training is not too computationally demanding.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(keras3)
source("source/prep_cifar56eco.R")
tensorflow::set_random_seed(2726) #sets for tensorflow, keras, and R

#' Load the data (image set with labels). I have written a function to extract
#' the ecological images from the CIFAR100 benchmark image dataset packaged with
#' keras. The first time you run this, the data will download to your computer
#' into a folder in the working directory called `data_large`. The folder will
#' be created if it doesn't exist. The folder will be added to your `.gitignore`
#' file so the large file is not uploaded to GitHub. The file is 150MB.

if ( !file.exists("data_large/cifar56eco.RData") ) {
    prep_cifar56eco()
}
load("data_large/cifar56eco.RData")

#' This has loaded 5 objects: `x_train`, `x_test`, `y_train`, `y_test`,
#' `eco_labels`. Let's inspect these objects.
#' 

#' For `x_train` we have 28000 images, each 32 x 32 pixels in 3 channels (RGB),
#' arranged in a 4D array. Pixel values range from 0-255.

dim(x_train)
class(x_train)
range(x_train)
hist(sample(x_train, 5000))

#' For `x_test`, we have 5600 images.

dim(x_test)

#' For y we have integers coding for 56 categories arranged in a 2D array (1
#' column matrix). Integers range from 0 to 55. Indexing starts at zero in
#' Python and tensorflow.

dim(y_train)
class(y_train)
head(y_train)
sort(unique(y_train)) #56 ecological categories

#' For `eco_labels` we have a dataframe containing the category names associated
#' with the integer class codes

eco_labels

#' Data preparation 1: convert image data to 0-1 scale.

x_train <- x_train / 255
x_test <- x_test / 255

#' Data preparation 2: convert integer response to a dummy variable matrix
#' suitable for keras/tensorflow. We'll use the `to_categorical()` function from
#' `keras3` to do that.

y_train_int <- y_train #keep a copy of the integer version for labelling later
y_train <- to_categorical(y_train, 56)

#' The result is a matrix with 56 columns, 1 column for each category of
#' organism.

class(y_train)
dim(y_train)

#' Looking at some portions of the matrix (upper left 6x14; row 1) we see we
#' have rows of zeros and ones, with a 1 in the column that represents the
#' category of the organism in the image. In row 1, column 8 has a 1, which says
#' that the image is category 7 (because of the offset indexing;
#' column 1 is category 0), i.e. "cattle".

y_train[1:6,1:14]
y_train[1,]

#' There are 500 images in each category, so this a balanced training set.

colSums(y_train)

#' Here's a random selection of images. You can see the image quality is poor
#' and it's hard even for a human to identify many of these organisms even with
#' the label written on the image!

par(mar=c(0,0,0,0), mfrow=c(5,5))
for (i in sample(1:dim(x_train)[1], 25) ) {
    plot(as.raster(x_train[i,,,]))
    text(0, 30, labels=eco_labels$name[y_train_int[i,]+1], col="red", pos=4)
}

#' Each image has 3 channels: RGB. In these plots the level of each channel is
#' displayed in grayscale (bright indicates a higher level).

par(mar=c(0,0,0,0), mfrow=c(2,2))
plot(as.raster(x_train[175,,,]))
text(0, 30, "color", col="white", pos=4)
plot(as.raster(x_train[175,,,1]))
text(0, 30, "red channel", col="white", pos=4)
plot(as.raster(x_train[175,,,2]))
text(0, 30, "green channel", col="white", pos=4)
plot(as.raster(x_train[175,,,3]))
text(0, 30, "blue channel", col="white", pos=4)


#' Now we'll set up the model. Define the CNN architecture illustrated in the
#' slides. The input layer is a 32 x 32 x 3 array. The output layer is the
#' probability in each of the 56 categories.

modcnn1 <- keras_model_sequential(input_shape=c(32,32,3)) |>
#   1st convolution-pool layer sequence
    layer_conv_2d(filters=6, kernel_size=c(2,2), padding="same") |>
    layer_max_pooling_2d(pool_size=c(2,2)) |>
    layer_activation_relu() |>
#   2nd convolution-pool layer sequence    
    layer_conv_2d(filters=12, kernel_size=c(2,2), padding="same") |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
    layer_activation_relu() |> 
#   3rd convolution-pool layer sequence    
    layer_conv_2d(filters=24, kernel_size=c(2,2), padding="same") |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
    layer_activation_relu() |> 
#   Flatten (384 nodes)
    layer_flatten() |>
#   Dense connection to output layer with softmax (56 categories to predict)    
    layer_dense(units=56) |> 
    layer_activation_softmax()

#' Check the architecture

modcnn1

#' We see that the model has about 23,000 parameters. For example, in the first
#' convolutional layer we have 6 filters, each 2x2, for each of the 3 input
#' channels (RGB), so 3 x 6 x 2 x 2 = 72 weights to which we add 6 bias
#' parameters (one for each output channel) to give 78 parameters. In the second
#' convolutional layer we have 6 x 12 x 2 x 2 + 12 = 300. In the third
#' convolutional layer we have 12 x 24 x 2 x 2 + 24 = 1,176 parameters. At the
#' input to the dense feedforward network where the array is flattened we have
#' 384 nodes connected to 56 output nodes, and this is where most of the model
#' parameters are: 384 x 56 weights + 56 biases = 21560 parameters.
#' 

#' Compile the model, specifying a `categorical_crossentropy` loss function,
#' which will be used in the gradient descent algorithm. This is a measure of
#' fit and accuracy on a likelihood scale. `RMSprop` is the default training
#' algorithm, a variant of stochastic gradient descent that incorporates an
#' adaptive learning rate for each parameter (i.e. for each of the many weights
#' and biases). We'll also collect a second and more direct measure of accuracy.
 
#+ eval=FALSE

compile(modcnn1, loss="categorical_crossentropy", optimizer="rmsprop",
        metrics="accuracy")

#' Train the model using an 80/20 train/validate split. The purpose of splitting
#' the training data here is to monitor training progress. Right at the
#' beginning above, we held out a separate test set that will not be involved in
#' training, which we'll use to measure out of sample accuracy. Training will
#' take about 2 to 15 minutes on CPU (2s per epoch on my relatively fast
#' laptop), or about 35 seconds on a single NVidia A100 GPU (e.g. on a CU Alpine
#' compute node).
 
#+ eval=FALSE

fit(modcnn1, x_train, y_train, epochs=60, batch_size=128, 
    validation_split=0.2) -> history

#' Save the model or load previously trained model. The first time you save the
#' model you'll need to set up the directories before running the following
#' code.

# save_model(modcnn1, "08_3_convolutional_nnet_files/saved/modcnn1.keras")
# save(history, file="08_3_convolutional_nnet_files/saved/modcnn1_history.Rdata")
modcnn1 <- load_model("08_3_convolutional_nnet_files/saved/modcnn1.keras")
load("08_3_convolutional_nnet_files/saved/modcnn1_history.Rdata")

#' Plotting the training history, we see that after about 50 epochs the
#' validation loss plateaus. While the training accuracy continues to improve,
#' the validation accuracy plateaus at about 32%. This is obviously not
#' impressive!

plot(history, smooth=FALSE)


#' Can we do better with a larger neural network? We'll try more layers, more
#' filters, bigger filters, and regularization.

tensorflow::set_random_seed(8424)

modcnn2 <- keras_model_sequential(input_shape=c(32,32,3)) |>
    #   1st convolution-pool layer sequence
    layer_conv_2d(filters=32, kernel_size=c(3,3), padding="same") |>
    layer_max_pooling_2d(pool_size=c(2,2)) |>
    layer_activation_relu() |> 
    #   2nd convolution-pool layer sequence    
    layer_conv_2d(filters=64, kernel_size=c(3,3), padding="same") |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
    layer_activation_relu() |> 
    #   3rd convolution-pool layer sequence    
    layer_conv_2d(filters=128, kernel_size=c(3,3), padding="same") |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
    layer_activation_relu() |> 
    #   4th convolution-pool layer sequence
    layer_conv_2d(filters=256, kernel_size=c(3,3), padding="same") |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
    layer_activation_relu() |>
    #   Flatten with dropout regularization
    layer_flatten() |>
    layer_dropout(rate=0.5) |>
    #   Standard dense layer
    layer_dense(units=512) |>
    layer_activation_relu() |>
    #   Output layer with softmax (56 categories to predict)    
    layer_dense(units=56) |> 
    layer_activation_softmax()

#' Check the architecture

modcnn2

#' This model has almost 1 million parameters! For example, in the
#' first convolutional layer we have 32 filters, each 3x3, for each of the 3
#' input channels (RGB), so 3 x 32 x 3 x 3 = 864 weights to which we add 32 bias
#' parameters (one for each output channel) to give 896 parameters. In the
#' second convolutional layer we have 32 x 64 x 3 x 3 + 64 = 18496, and so on.
#' At the input to the dense feedforward network where the array is flattened we
#' have 1024 nodes connected to 512 nodes, so 1024 x 512 weights + 512 biases =
#' 524,800 parameters. Nevertheless, we do have a lot of data, about 86 million
#' pixels (28,000 x 32 x 32 x 3).
#' 

#' Compile the model with details as for the previous model

#+ eval=FALSE

compile(modcnn2, loss="categorical_crossentropy", optimizer="rmsprop",
        metrics="accuracy")

#' Train the model as for previous model. This larger model will take 4X longer
#' to train on CPU: 5-30 mins. On GPU it's actually faster to train than the
#' smaller model because of startup overhead and because we're training for
#' fewer epochs (25 seconds on a single NVidia A100 GPU, e.g. on a CU Alpine
#' compute node). The bigger the model, the greater the advantage of GPU.

#+ eval=FALSE

fit(modcnn2, x_train, y_train, epochs=30, batch_size=128, 
    validation_split=0.2) -> history

#' Save the model or load previously trained model

# save_model(modcnn2, "08_3_convolutional_nnet_files/saved/modcnn2.keras")
# save(history, file="08_3_convolutional_nnet_files/saved/modcnn2_history.Rdata")
modcnn2 <- load_model("08_3_convolutional_nnet_files/saved/modcnn2.keras")
load("08_3_convolutional_nnet_files/saved/modcnn2_history.Rdata")

#' Plotting the training history, we see evidence of overfitting after about 15
#' epochs as the validation loss begins to climb. While the training accuracy
#' improves, the validation accuracy is stuck at about 40%.

plot(history, smooth=FALSE)

#' This model has improved the predictive accuracy quite a bit but it's still
#' only around 40%.
#' 

#' Plot a random selection of predictions. While the model is incorrect on many
#' images, it is remarkable that it predicts many correctly (much better than
#' random guessing) and those that it gets wrong, you can often see how the
#' image resembles the model's prediction.

tensorflow::set_random_seed(4134)
selection <- sort(sample(1:dim(x_test)[1], 16))
par(mar=c(0,0,0,0), mfrow=c(4,4))
for ( i in selection ) {
    pred <- as.numeric(predict(modcnn2, x_test[i,,,,drop=FALSE]))
    plot(as.raster(x_test[i,,,]))
    text(0, 30, paste("prediction =", eco_labels$name[which.max(pred)]), col="red", pos=4)
    text(0, 28, paste("prob =", round(pred[which.max(pred)],2)), col="red", pos=4)
    text(0, 26, paste("actual =", eco_labels$name[y_test[i,]+1]), col="red", pos=4)
} 

#' Predictions and overall accuracy on the hold out test set (about 41%, which
#' agrees with the validation accuracy in training)

pred_prob <- predict(modcnn2, x_test)
pred_cat <- max.col(pred_prob) - 1  #subtract 1 because categories start at zero
mean(pred_cat == drop(y_test))      #drop converts 1D matrix to vector

#' Plot probabilities for the same selection of test cases as above. For some
#' images, multiple categories have high probability but for others a clear
#' winner is identified. Nevertheless, the clear winner is not necessarily
#' correct; the prediction can be confidently wrong!

nr <- nrow(pred_prob)
pred_prob |> 
    data.frame() |>
    mutate(case=seq(nr)) |>
    tidyr::pivot_longer(cols=starts_with("X"), names_to="category", values_to="probability") |> 
    mutate(category=as.integer(sub("X", "", category)) - 1) |> 
    filter(case %in% selection) |> 
    ggplot() +
    geom_point(aes(x=category, y=probability)) +
    facet_wrap(vars(case), nrow=4, ncol=4, labeller=label_both)

