# =========================================================================== #
# Multi-class Kernel SVM classification of the iris dataset                   #
# Adapted from the wonderful Tensorflow Cookbook by Nick McClure              #
# =========================================================================== #

library(reticulate)
library(tensorflow)
library(ggplot2)
library(gridExtra)

# Adjust as desired
batch_size <- 50L
hidden_nodes <- 10L
iter_length <- 100L
features <- c("Sepal.Length", "Petal.Length")
species <- c("setosa", "versicolor", "virginica")

# Pull two features out of the iris dataset to classify based on
x_vals <- as.matrix(iris[, features])

# Create 1-hot labels for each species
y_vals <- vapply(
  species, function(x) {
  ifelse(iris[["Species"]] == x, 1L, 0L)
}, integer(dim(iris)[1L]))

# We want to read the data in long (3 rows, 1 column per observation)
# so we need to transpose the matrix
y_vals <- t(y_vals)

# Create test and training example sets
train_idx <- sample(seq(dim(x_vals)[1L]), size = floor(dim(x_vals)[1L] * 0.8), replace = FALSE)

x_train <- x_vals[train_idx, ]
x_test <- x_vals[-train_idx, ]
y_train <- y_vals[, train_idx]
y_test <- y_vals[, -train_idx]

# To avoid vanishing gradient, we need to normalize the examples coming in.
col_normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

x_train <- apply(x_train, 2L, col_normalize)
x_test <- apply(x_train, 2L, col_normalize)

# Define input tensors
with(tf$name_scope("input"), {
  x_data <- tf$placeholder(tf$float32, shape(NULL, 2L), name = "x_data")
  y_target <- tf$placeholder(tf$float32, shape(3L, NULL), name = "y_labels")
})

# Define hidden layer
# NOTE: Here, we take in out x_data, multiply it by some weight matrix
# and add a bias. The result will be n values, where n is the number of
# hidden nodes
with(tf$name_scope("Hidden"), {
  A1 <- tf$Variable(tf$random_normal(shape(2L, hidden_nodes)))
  b1 <- tf$Variable(tf$random_normal(shape(hidden_nodes)))
  hid_out <- tf$nn$relu(tf$add(tf$matmul(x_data, A1), b1))
})

# Define output layer
# NOTE: To perform a classification on the three flower types, we use
# softmax regression to return three values representing the probability
# of each class. We then take the class with the highest probability and
# compare that to the 1-hot label to determine if we've got it right.
with(tf$name_scope("output"), {
  A2 <- tf$Variable(tf$random_normal(shape(hidden_nodes, 3L)))
  b2 <- tf$Variable(tf$random_normal(shape(3L)))
  soft_out <- tf$nn$softmax(tf$matmul(hid_out, A2) + b2)
  prediction <- tf$argmax(soft_out, 1L)
})

# Define loss function
# NOTE: here we calculate KL-divergence to compare the discrete distributions
# of our labels to our predictions.
loss <- tf$reduce_mean(
  -tf$reduce_sum(y_target * tf$log(soft_out)), 1L)

# Declare our training method
# NOTE: I use Adam here, but you could use gradient descent or whatever you want
opt <- tf$train$AdamOptimizer(learning_rate = 1e-4)
train_step <- opt$minimize(loss)


# Initialize variables and start session
init <- tf$global_variables_initializer()
session <- tf$Session()
session$run(init)


# Allocate vectors to store accuracy and loss values
loss_vec <- vector(mode = "numeric", length = iter_length)
batch_accuracy <- vector(mode = "numeric", length = iter_length)

# Training loop
# Run the model however many iterations
for (i in seq_len(iter_length)) {
  rand_index <- sample(seq(dim(x_train)[1L]), batch_size)
  rand_x <- x_train[rand_index, ]
  rand_y <- y_train[, rand_index]

  session$run(
    train_step,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  loss_vec[i] <- session$run(
    loss,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  batch_accuracy[i] <- session$run(
    accuracy,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y,
      prediction_grid = rand_x
    )
  )


  if (i %% 50 == 0) {
    cat("[", i, "] Loss:", loss_vec[i], "Accuracy:", batch_accuracy, "\n")
  }
}
