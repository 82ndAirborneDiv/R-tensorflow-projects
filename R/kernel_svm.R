# =========================================================================== #
# Kernel SVM classification of circular groupings                             #
# Adapted from the wonderful Tensorflow Cookbook by Nick McClure              #
# =========================================================================== #

library(reticulate)
library(tensorflow)
library(ggplot2)
library(gridExtra)

# Import the make_circles() function I wrote to make test data
source("R/util.R"))

# Adjust as desired
batch_size <- 350L
soft_margin <- 0.01
circle <- "inner"

# Create dummy data to classify using make_circles()
data <- make_circles(samples = 350L, factor = 0.5, noise = 0.1)


# Create a binary label for identifying inner vs. outer circle
# We want to store this as a matrix so it reads in "long" when we pass it
# to the feed dict.
y_vals <- matrix(NA_integer_, ncol = 1, nrow = dim(data)[1L])
y_vals[, 1] <- ifelse(data[["circ"]] == circle, 1L, -1L)

x_vals <- as.matrix(data[, c("x", "y")])

x_min <- min(data[, "x"])
x_max <- max(data[, "x"])
y_min <- min(data[, "x"])
y_max <- max(data[, "y"])

y_vals <- np_array(y_vals)
x_vals <- np_array(x_vals)

class1_x <- data[data[["circ"]] == circle, "x"]
class1_y <- data[data[["circ"]] == circle, "y"]
class2_x <- data[data[["circ"]] != circle, "x"]
class2_y <- data[data[["circ"]] != circle, "y"]


# Define input tensors
with(tf$name_scope("input"), {
  x_data <- tf$placeholder(tf$float32, shape(NULL, 2L), name = "x_data")
  y_target <- tf$placeholder(tf$float32, shape(NULL, 1L), name = "y_labels")
  prediction_grid <- tf$placeholder(tf$float32, shape(NULL, 2), name = "prediction_grid")
})

# Set the soft margin term
alpha <- tf$constant(soft_margin)

# Define SVM kernel tensors
with(tf$name_scope("SVM"), {
  b <- tf$Variable(tf$random_normal(shape(1L, batch_size)))
  gamma <- tf$constant(-50)

  dist <- tf$reduce_sum(tf$square(x_data), 1L)
  dist <- tf$reshape(dist, shape(-1L, 1L))

  sq_dists <- tf$add(
    tf$subtract(dist, tf$multiply(2, tf$matmul(x_data, tf$transpose(x_data)))),
    tf$transpose(dist))

  kernel <- tf$exp(tf$multiply(gamma, tf$abs(sq_dists)))
})

# Define gaussian SVM kernel operations
first_term <- tf$reduce_sum(b)
b_vec_cross <- tf$matmul(tf$transpose(b), b)
y_target_cross <- tf$matmul(y_target, tf$transpose(y_target))
second_term <- tf$reduce_sum(tf$multiply(kernel, tf$multiply(b_vec_cross, y_target_cross)))

# Define loss function
loss <- tf$negative(tf$subtract(first_term, second_term))

# Define Prediction kernel
rA <- tf$reshape(tf$reduce_sum(tf$square(x_data), 1L), shape(-1L, 1L))
rB <- tf$reshape(tf$reduce_sum(tf$square(prediction_grid), 1L), shape(-1L, 1L))

pred_sq_dist <- tf$add(
  tf$subtract(rA, tf$multiply(2, tf$matmul(x_data, tf$transpose(prediction_grid)))),
  tf$transpose(rB)
)

pred_kernel <- tf$exp(tf$multiply(gamma, tf$abs(pred_sq_dist)))

# Define classification term
prediction_output <- tf$matmul(
  tf$multiply(tf$transpose(y_target), b),
  pred_kernel
)

# Declare prediction and accuracy functions
prediction <- tf$sign(prediction_output - tf$reduce_mean(prediction_output))
accuracy <- tf$reduce_mean(
  tf$cast(
    tf$equal(tf$squeeze(prediction), tf$squeeze(y_target)),
    dtype = tf$float32)
  )


# Declare gradient descent optimizer
opt <- tf$train$GradientDescentOptimizer(learning_rate = 0.002)
train_step <- opt$minimize(loss)

# Initialize variables and start session
init <- tf$global_variables_initializer()
session <- tf$Session()
session$run(init)


# Allocate vectors to store accuracy and loss values
loss_vec <- vector(mode = "numeric", length = 1000L)
batch_accuracy <- vector(mode = "numeric", length = 1000L)

# Training loop
# Run the model 1000 times
for (i in seq_len(1000)) {
  # NOTE: Numpy arrays are, of course, 0-indexed.
  rand_index <- sample(seq_len(batch_size) - 1L, batch_size)
  rand_x <- x_vals[rand_index]
  rand_y <- y_vals[rand_index]

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
      x = seq(x_min, x_max, by = 0.02),
      y = seq(y_min, y_max, by = 0.02)
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
    mesh$pred <- factor(t(pred), labels = c("outer", "inner"))
    colnames(mesh) <- c("x", "y", "pred")

    # Make input point data and labels to layer over prediction surface plot
    points <- data.frame(
      as.matrix(rand_x),
      label = factor(as.numeric(rand_y), labels = c("outer", "inner"))
    )
    colnames(points) <- c("x", "y", "label")

    # Plot the grid then overlay the data points
    # NOTE: Extracting the data from a numpy array to plot is a little
    # hacky.
    plot <- ggplot(mesh, aes(x, y, fill = pred)) +
      geom_raster() +
      geom_point(
        data = points,
        aes(x, y, shape = label),
        inherit.aes = FALSE) +
      labs(
        title = sprintf("Gausian Kernel SVM -- Step %i", i),
        subtitle = sprintf("Loss = %.2f", temp_loss),
        fill = "Predicted",
        shape = "Actual"
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
