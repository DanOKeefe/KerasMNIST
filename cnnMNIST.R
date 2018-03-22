#install.packages('keras')
library(keras)
#install_keras(tensorflow = "default", method = 'conda')
# rm(list=ls())

# Data

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Data prep

x_train <- x_train / 255
x_test <- x_test / 255

# CIFAR x_train data is (50000, 32, 32, 3)
dim(x_train) # MNIST x_train data is(60000, 28, 28). Need to convert to (60000, 28, 28, 1) done later.

# Create one-hot vectors for y data

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Model
# Optimal hyperparameters for classifying MNIST digits used from https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier

model <- keras_model_sequential() %>%
  
  # 2 Convolutional, activation, and pooling layers
  
  layer_conv_2d(filters = 32, kernel_size = c(5,5), padding = 'same', input_shape = c(28,28,1), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(5,5), padding = 'same', activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # unrolled to fully connected layer
  
  layer_flatten() %>%
  layer_dense(units = 1024) %>%
  layer_dropout(0.4) %>%
  layer_activation('relu') %>%

  # softmax output layer

  layer_dense(units = 10, activation = 'softmax') # 10 outputs

#summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metric = c('accuracy')
)

#dim(x_train)
# reshape x_train and x_test to proper dimension
x_train <- array(x_train, dim = c(60000, 28, 28, 1))
x_test <- array(x_test, dim = c(10000, 28, 28, 1))
#dim(x_train)


#?fit
history <- model %>% fit(
  x = x_train,
  y = y_train,
  batch_size = 1000,
  epochs = 15,
  validation_split = 0.1
)

model %>% evaluate(x_test, y_test)
