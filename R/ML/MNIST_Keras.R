# MNIST classifier using the R Keras API for Tensorflow
library(keras)

# The MNIST dataset is housed in an empty function
mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshape the 3D array to a matrix of 784 vectors signifying
# the flattened 28x28 images
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)

# Convert 0-255 grayscale to floating point value between 0 and 1
x_train <- x_train / 255
x_test <- x_test / 255

# Dummy code factor variables to make them easier to train
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Define the neural network using ReLU neurons and softmax regression
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

# Define the cost function and optimization algorithm
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# train and evaluate performance
history <- model %>% fit(
  x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)
model %>% evaluate(x_test, y_test)
# store plot of accuracy and loss over time
model_plot <- plot(history)
