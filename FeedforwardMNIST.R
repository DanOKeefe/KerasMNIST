#install.packages('keras')
library(keras)

install_keras()

?dataset_mnist # 60,000 28x28 grayscale images of the 10 digits, with test set of 10,000 images.

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# x data is a 3 dimensional array (images, width, height)
# y data is a 1 dimension array containing  digits 0-9

# Convert x data to matrices by flattening the (width, height) dimensions into one dimension
# Make x data range from 0-1 instead of 0-255

dim(x_train) # (60000, 28, 28)
dim(x_test) # (10000, 28, 28)

x_train <- array_reshape(x_train, c(60000, 784))
x_test <- array_reshape(x_test, c(10000, 784))

# x values contain pixel intensity data ranging from 0-255. Change that to 0-1

x_train <- x_train / 255
x_test <- x_test / 255

# Convert y data to one-hot vectors representing the 10 possible digit classes

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Model: 3 hidden layers. 
# 784 inputs -> 200 nodes -> 100 nodes -> 50 nodes -> 10 outputs

model <- keras_model_sequential() %>%
  layer_dense(units=200, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax') # 10 outputs

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)


#summary(model)

history <- model %>% fit(
  x_train, y_train,
  batch_size = 1000,
  epochs = 20,
  validation_split = 0.1
)


model %>% evaluate(x_test, y_test)
