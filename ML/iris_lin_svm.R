# =========================================================================== #
# Linear SVM applied to the iris dataset                                      #
# Adapted from the wonderful Tensorflow Cookbook by Nick McClure               #
# =========================================================================== #

# NOTE: Intuition
# Unlike linear regression which attempts to find a line such that the margin
# between each data point and the regression line is minimized, a linear SVM
# takes two classes and attempts to find a line such that the margin between the
# points of each class and the line is maximized. The result is not a line which
# falls in the center of the clustering, but between the two clusters.
# The "Support Vectors" are those points at the furthest most edge of each class
# which "support" the class margin. For higher dimensional data, the result can
# be an n-dimensional hyperplane.

# NOTE: Classifying Iris
# Setosa happens to be a very easy target for linear classification using
# sepal length and petal length as predictors. Virginica and versicolor are
# poorly differentiated by these two predictors, and linear classification will
# perform poorly--often failing to converge altogether.

library(reticulate)
library(tensorflow)
library(ggplot2)

# Adjust as needed
batch_size <- round(dim(iris)[1L] * 0.8)
sort_species <- "setosa"
soft_margin <- 0.01

# Restructure the iris data set as a two-column matrix
x_vals <- as.matrix(iris[, c("Sepal.Length", "Petal.Length")])


# Create a binary label for identifying the species
# We want to store this as a matrix so it reads in "long" when we pass it
# to the feed dict.
y_vals <- matrix(NA_integer_, ncol = 1, nrow = dim(iris)[1L])
y_vals[, 1] <- ifelse(iris[["Species"]] == sort_species, 1L, -1L)


# Randomly sample observations to give us a training set and a test set
train_idx <- sample(seq_len(dim(x_vals)[1L]), batch_size)
test_idx <- seq_len(dim(x_vals)[1L])[-train_idx]

# Split the data and labels into test and training sets
# I store them as numpy arrays here just to make things easier later
x_train <- np_array(x_vals[train_idx, ])
x_test <- np_array(x_vals[test_idx, ])
y_train <- np_array(as.matrix(y_vals[train_idx, ]))
y_test <- np_array(as.matrix(y_vals[test_idx, ]))

# Define input tensors
with(tf$name_scope("input"), {
  x_data <- tf$placeholder(tf$float32, shape(NULL, 2L), name = "x_data")
  y_target <- tf$placeholder(tf$float32, shape(NULL, 1L), name = "y_labels")
})

# Set the soft margin term
alpha <- tf$constant(soft_margin)

# Define SVM variable tensors
with(tf$name_scope("SVM"), {
  A <- tf$Variable(tf$random_normal(shape(2L, 1L)))
  b <- tf$Variable(tf$random_normal(shape(1L, 1L)))
})

# Define SVM internal operations
model_output <- tf$subtract(tf$matmul(x_data, A), b)
l2_norm <- tf$reduce_sum(tf$square(A))


# Define classification term
# max(0, 1-predicted * actual)
classification_term <- tf$reduce_mean(
  tf$maximum(0, tf$subtract(1, tf$multiply(model_output, y_target)))
)

# Put terms together to define the loss function
# sum(1/n max(0, 1-predicted(i) * actual(i))) + alpha * sum(A^2)
loss <- tf$add(classification_term, tf$multiply(alpha, l2_norm))

# Declare prediction and accuracy functions
prediction <- tf$sign(model_output)
accuracy <- tf$reduce_mean(
  tf$cast(tf$equal(prediction, y_target), dtype = tf$float32)
)

# Declare gradient descent optimizer
opt <- tf$train$GradientDescentOptimizer(learning_rate = 0.01)
train_step <- opt$minimize(loss)

# Initialize variables and start session
init <- tf$global_variables_initializer()
session <- tf$Session()
session$run(init)


# Allocate vectors to store accuracy and loss values
loss_vec <- vector(mode = "numeric")
train_accuracy <- vector(mode = "numeric")
test_accuracy <- vector(mode = "numeric")

# Training loop
# Run the model 500 times
for (i in seq_len(500)) {
  # NOTE: Numpy arrays are, of course, 0-indexed.
  rand_index <- sample(seq_len(batch_size) - 1L, batch_size)
  rand_x <- x_train[rand_index]
  rand_y <- y_train[rand_index]

  session$run(
    train_step,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  temp_loss <- session$run(
    loss,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  loss_vec[i] <- temp_loss

  train_acc_temp <- session$run(
    accuracy,
    feed_dict = dict(
      x_data = x_train,
      y_target = y_train
    )
  )

  train_accuracy[i] <- train_acc_temp

  test_acc_temp <- session$run(
    accuracy,
    feed_dict = dict(
      x_data = x_test,
      y_target = y_test
    )
  )

  test_accuracy[i] <- test_acc_temp

  # Give a nice visual output of how we are performing
  if (i %% 50 == 0) {
    slope <- -session$run(A)[2, 1] / session$run(A)[1, 1]
    y_int <- session$run(b)[1, 1]

    plot <- ggplot(iris, aes(y = Sepal.Length, x = Petal.Length, color = Species)) +
      geom_point() +
      stat_function(fun = function(x) x * slope + y_int, geom = "line") +
      stat_function(fun = function(x) x * slope + y_int + 1, geom = "line", linetype = "dashed") +
      stat_function(fun = function(x) x * slope + y_int - 1, geom = "line", linetype = "dashed") +
      theme_bw() +
      labs(
        title = sprintf("Iris linear SVM -- step %i", i),
        subtitle = sprintf("Loss = %.2f", temp_loss)
      )

    print(plot)
  }
}
