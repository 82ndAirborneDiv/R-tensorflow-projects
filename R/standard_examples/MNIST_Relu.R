# Tensorflow cNN MNIST exammple with ReLU neurons
library(tensorflow)

sess <- tf$InteractiveSession()

# Read MNIST dataset
input_dataset <- tf$examples$tutorials$mnist$input_data
mnist <- input_dataset$read_data_sets("MNIST-data", one_hot = TRUE)

weight_variable <- function(shape) {
  # Returns an empty weight variable of a given shape
  initial <- tf$truncated_normal(shape, stddev = 0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  # returns an empty bias variable of a given shape
  initial <- tf$constant(0.1, shape = shape)
  tf$Variable(initial)
}

conv2d <- function(x, W) {
  # Creates a convolutional operator
  tf$nn$conv2d(x, W,
    strides = c(1L, 1L, 1L, 1L),
    padding = "SAME"
  )
}

max_pool_2x2 <- function(x) {
  tf$nn$max_pool(x,
    ksize = c(1L, 2L, 2L, 1L),
    strides = c(1L, 2L, 2L, 1L),
    padding = "SAME"
  )
}
# placeholders for images to be analyzed and their tags
x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))

# First convolutional layer
# Produces 32 5 x 5 patches for every one input
W_conv1 <- weight_variable(shape(5L, 5L, 1L, 32L))
b_conv1 <- bias_variable(shape(32L))

# Reshape x from 2 dimensional storage to 4D
# 1 x 784 -> 1 X 28 X 28 X 1 (color channel)
x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))
h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)

# First Pooling layer
# 28 x 28 -> 14 x 14
h_pool1 <- max_pool_2x2(h_conv1)

# Second convolutional layer
# produces 64 5 x 5 patches for each input
W_conv2 <- weight_variable(shape = shape(5L, 5L, 32L, 64L))
b_conv2 <- bias_variable(shape = shape(64L))
h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Second pooling layer
# 14 x 14 -> 7 x 7
h_pool2 <- max_pool_2x2(h_conv2)

# Fully-connected layer
# full image processing on 1024 neurons, returning 1024 results
W_fc1 <- weight_variable(shape(7L * 7L * 64L, 1024L))
b_fc1 <- bias_variable(shape(1024L))
# Pool from the resulting 64 7 x 7 patches into a 1D 1024 tensor
h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# trim some results to reduce overfitting
keep_prob <- tf$placeholder(tf$float32)
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

# Readout
# take the 1024L 1D tensor, and return a 10L 1D tensor
# Which is the result of softmax regression.
W_fc2 <- weight_variable(shape(1024L, 10L))
b_fc2 <- bias_variable(shape(10L))
y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

# Cross-entropy calculation
# Intuition: "Just how bad was the prediction by the neural net compared to
# what was expected?"
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv),
  reduction_indices = 1L
))

# Optimize the neural net with the Adam method, minimizing cross-entropy
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)

# "Is the number with the highest probability reported by from softmax equal to the label
# for the image?"
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
# Cast boolean value corresponding to whether value is correct to float
# then take the mean to determine % correct
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Initialize all variables
sess$run(tf$global_variables_initializer())


# Logging every 100 iterations
for (i in 1:20000) {
  batch <- mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = batch[[1]], y_ = batch[[2]], keep_prob = 1.0
    ))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  train_step$run(feed_dict = dict(
    x = batch[[1]], y_ = batch[[2]], keep_prob = 0.5
  ))
}

# How did we do?
nbatch <- floor(nrow(mnist$test$images) / 50)
test_accuracy <- 0
# Batch run the calculation to avoid memory issues on older hardware
for (i in 1:nbatch) {
  batch <- mnist$test$next_batch(50L)
  test_accuracy <- test_accuracy +
    accuracy$eval(feed_dict = dict(
      x = batch[[1]],
      y_ = batch[[2]],
      keep_prob = 1.0
    ))
}
# Report
cat(sprintf("test accuracy %g", test_accuracy / nbatch))
