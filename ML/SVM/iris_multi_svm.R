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
soft_margin <- 0.01
iter_length <- 100L
features <- c("Sepal.Length", "Petal.Length")
species <- c("setosa", "versicolor", "virginica")

# Pull two features out of the iris dataset to classify based on
x_vals <- as.matrix(iris[, features])

# Create 1-hot labels for each species
y_vals <- vapply(
  species, function(x) {
  ifelse(iris[["Species"]] == x, 1L, -1L)
}, integer(dim(iris)[1L]))

# We want to read the data in long (3 rows, 1 column per observation)
# so we need to transpose the matrix
y_vals <- t(y_vals)

# Define input tensors
with(tf$name_scope("input"), {
  x_data <- tf$placeholder(tf$float32, shape(NULL, 2L), name = "x_data")
  y_target <- tf$placeholder(tf$float32, shape(3L, NULL), name = "y_labels")
  prediction_grid <- tf$placeholder(tf$float32, shape(NULL, 2), name = "prediction_grid")
})

# Set the soft margin term
alpha <- tf$constant(soft_margin)

# Define SVM kernel tensors
with(tf$name_scope("SVM"), {
  b <- tf$Variable(tf$random_normal(shape(3L, batch_size)))
  gamma <- tf$constant(-10)

  dist <- tf$reduce_sum(tf$square(x_data), 1L)
  dist <- tf$reshape(dist, shape(-1L, 1L))

  sq_dists <- tf$multiply(2, tf$matmul(x_data, tf$transpose(x_data)))

  kernel <- tf$exp(tf$multiply(gamma, tf$abs(sq_dists)))
})


reshape_matmul <- function(mat){
  v1 <- tf$expand_dims(mat, 1L)
  v2 <- tf$reshape(v1, shape(3L, batch_size, 1L))

  return(tf$matmul(v2, v1))
}

# Define gaussian SVM kernel operations
first_term <- tf$reduce_sum(b)
b_vec_cross <- tf$matmul(tf$transpose(b), b)
y_target_cross <- reshape_matmul(y_target)
second_term <- tf$reduce_sum(tf$multiply(kernel, tf$multiply(b_vec_cross, y_target_cross)), shape(1L, 2L))

# Define loss function
loss <- tf$negative(tf$subtract(first_term, second_term))

# Define prediction operations
rA <- tf$reshape(tf$reduce_sum(tf$square(x_data), 1L), shape(-1L, 1L))
rB <- tf$reshape(tf$reduce_sum(tf$square(prediction_grid), 1L), shape(-1L, 1L))

pred_sq_dist <- tf$add(
  tf$subtract(rA, tf$multiply(2, tf$matmul(x_data, tf$transpose(prediction_grid)))),
  tf$transpose(rB)
)

pred_kernel <- tf$exp(tf$multiply(gamma, tf$abs(pred_sq_dist)))


# Define classification term
prediction_output <- tf$matmul(
  tf$multiply(y_target, b),
  pred_kernel
)

# Declare prediction and accuracy functions
prediction <- tf$argmax(prediction_output - tf$expand_dims(tf$reduce_mean(prediction_output, 1L), 1L), 0L)
accuracy <- tf$reduce_mean(
  tf$cast(
    tf$equal(prediction, tf$argmax(y_target, 0L)),
    dtype = tf$float32)
  )


# Declare gradient descent optimizer
opt <- tf$train$GradientDescentOptimizer(learning_rate = 0.01)
train_step <- opt$minimize(loss)

# Initialize variables and start session
init <- tf$global_variables_initializer()
session <- tf$Session()
session$run(init)


# Allocate vectors to store accuracy and loss values
loss_vec <- vector(mode = "numeric", length = iter_length)
batch_accuracy <- vector(mode = "numeric", length = iter_length)

# Training loop
# Run the model 100 times
for (i in seq_len(iter_length)) {
  rand_index <- sample(seq(dim(iris)[1L]), batch_size)
  rand_x <- x_vals[rand_index, ]
  rand_y <- y_vals[, rand_index]

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
      x_data = rand_x,
      y_target = rand_y,
      prediction_grid = rand_x
    )
  )

  batch_accuracy[i] <- train_acc_temp

  if (i %% 50 == 0) {
    # Every 50 steps give a visual update of how we are doing

    # Create cartesian grid to get predictions for
    mesh <- expand.grid(
      x = seq(min(iris[, features[1]]), max(iris[, features[1]]), by = 0.02),
      y = seq(min(iris[, features[2]]), max(iris[, features[2]]), by = 0.02)
    )

    # Generate predictions for all data points
    pred <- session$run(
      prediction,
      feed_dict = dict(
        x_data = rand_x,
        y_target = rand_y,
        prediction_grid = as.matrix(mesh))
    )

    # Add the predictions to the grid data
    mesh$pred <- factor(pred, labels = species)
    colnames(mesh) <- c("x", "y", "pred")

    # Make data frame of input point data and labels to layer over prediction surface plot
    actual <- vector(mode = "character", length = batch_size)

    for (i in seq_along(actual)) {
      actual[i] <- species[rand_y[, i] == 1L]
    }

    points <- data.frame(
      as.matrix(rand_x),
      label = actual
    )

    colnames(points) <- c("x", "y", "label")

    # Plot the grid then overlay the data points
    plot <- ggplot(mesh, aes(x, y, fill = pred)) +
      geom_raster() +
      geom_point(
        data = points,
        aes(x, y, shape = label),
        inherit.aes = FALSE) +
      labs(
        title = sprintf("Iris Gausian Kernel SVM -- Step %i", i),
        subtitle = sprintf("Loss = %.2f", temp_loss),
        fill = "Predicted",
        shape = "Actual",
        x = features[1],
        y = features[2]
    ) +
    theme_bw()

    print(plot)
  }
}

acc_df <- data.frame(
  accuracy = batch_accuracy,
  iter = seq_along(batch_accuracy)
)

loss_df <- data.frame(
  loss = loss_vec,
  iter = seq_along(loss_vec)
)

accuracy_plot <- ggplot(acc_df, aes(x = iter, y = accuracy)) +
  geom_line() +
  labs(
    title = "Batch accuracy",
    x = "Iteration",
    y = "Accuracy"
  ) +
  scale_y_continuous(labels = scales::percent) +
  theme_bw()

loss_plot <- ggplot(loss_df, aes(x = iter, y = loss)) +
  geom_line() +
  labs(
    title = "Batch loss",
    x = "Iteration",
    y = "Loss"
  ) +
  theme_bw()

# Give final output plot
grid.arrange(plot, accuracy_plot, loss_plot, ncol = 3)
