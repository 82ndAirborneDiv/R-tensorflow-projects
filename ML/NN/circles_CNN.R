# =========================================================================== #
# Convolutional neural network classification of circular groupings           #
# (Not really) Adapted from the wonderful Tensorflow Cookbook by Nick McClure #
# =========================================================================== #

# NOTE: This is just me messing around. I saw last time that a simple topology
# is not sufficient to explain more complex patterned data (which is not linearaly separable).
# That is often stated as a preface to modelling with MLPs, so the results were not all
# that interesting.
# For this, I will modify the original topology, and add a convolutional layer
# to attempt to account for data which are not linearly separable.

# This provides a very clear illustration of how limited a simple FF neural network
# is in dealing with more complex clustering patterns.
# If we do not use a convolutional kernel function, we are limiting the dimensionality
# of our prediction space, and embedded circular clustering cannot be well explained.

library(reticulate)
library(tensorflow)
library(ggplot2)
library(gridExtra)

# Import the make_circles() function I wrote to make test data
source("make_circles.R")

# Adjust as desired
hidden_neurons <- 100L
train_iter <- 1000L
n_samples <- 1000L
batch_size <- 300L
circle <- "outer"

# Create dummy data to classify using make_circles()
# NOTE: I make the inner circle 50% smaller and add a bit of gaussian noise
data <- make_circles(samples = n_samples, factor = 0.5, noise = 0.1)


# Create a binary label for identifying inner vs. outer circle
# We want to store this as a matrix so it reads in "long" when we pass it
# to the feed dict.
# We make it 1-hot, so it becomes simple to calculate divergence or cross-entropy
# 1-hot, meaning dummy-code
y_vals <- matrix(NA_integer_, ncol = 1, nrow = n_samples)
y_vals[, 1] <- as.integer(data[["circ"]] == circle)

# Cast the input data as a matrix so tensorflow doesn't at us
x_vals <- as.matrix(data[, c("x", "y")])

# Take a sample of the data to use to test for n-fold cross validation
train_idx <- sample(seq(n_samples), size = floor(n_samples * 0.8), replace = FALSE)

# Label the values we use for training and testing in case we care to visualize
# that sort of thing
data$train <- "training"
data$train[-train_idx] <- "testing"

# Define a function to normalize our data between 0 to 1
col_normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Create our training and testing datasets
training_set <- list(
  x = apply(x_vals[train_idx, ], 2L, col_normalize),
  y = y_vals[train_idx, ,drop = FALSE]
)

test_set <- list(
  x = apply(x_vals[-train_idx, ], 2L, col_normalize),
  y = y_vals[-train_idx, ,drop = FALSE]
)

# Make a dataframe containing the space that the input data occupies
# we will use this to visualize the boundries of each class as predicted
# by the model
mesh <- expand.grid(
  x = seq(min(x_vals[, "x"]), max(x_vals[, "x"]), by = 0.02),
  y = seq(min(x_vals[, "y"]), max(x_vals[, "y"]), by = 0.02)
)

# Define input layer
with(tf$name_scope("input"), {
  x_data <- tf$placeholder(tf$float32, shape(NULL, 2L), name = "x_data")
  y_target <- tf$placeholder(tf$float32, shape(NULL, 1L), name = "y_labels")
})

# Define the hidden layer
with(tf$name_scope("hidden"), {
  A1 <- tf$Variable(tf$random_normal(shape(2L, hidden_neurons)))
  b1 <- tf$Variable(tf$random_normal(shape(hidden_neurons)))
  hid_out <- tf$nn$sigmoid(tf$add(tf$matmul(x_data, A1), b1))
})

# Define the output layer
with(tf$name_scope("output"), {
  A2 <- tf$Variable(tf$random_normal(shape(hidden_neurons, 2L)))
  b2 <- tf$Variable(tf$random_normal(shape(2L)))
  output <- tf$nn$sigmoid(tf$add(tf$matmul(hid_out, A2), b2))
  prediction <- tf$argmax(output, 1L)
})

# Calculate cross entropy which we will use as our loss function
loss <- tf$reduce_mean(
  -tf$reduce_sum(y_target * tf$log(output), reduction_indices = 1L)
)

# Calculate accuracy
# NOTE: Commpare by observation if the predicted class is equal to the target class
# and take the mean across the examples
accuracy <- tf$reduce_mean(
  tf$cast(tf$equal(prediction, tf$argmax(y_target, 1L)), tf$float32)
)


# Declare training optimizer
# NOTE: Here we could use Adam or whatever. I've used standard gradient descent
# for no reason in particular
opt <- tf$train$GradientDescentOptimizer(learning_rate = 0.002)
train_step <- opt$minimize(loss)

# Initialize variables and start session
init <- tf$global_variables_initializer()
session <- tf$Session()
session$run(init)


# Allocate vectors to store accuracy and loss values
loss_vec <- vector(mode = "numeric", length = train_iter)
batch_accuracy <- vector(mode = "numeric", length = train_iter)

# Training loop
# Run the model however many times we defined above
for (i in seq_len(train_iter)) {
  # Take just a sample of the training set each iteration
  rand_index <- sample(seq_len(batch_size), batch_size)
  rand_x <- training_set$x[rand_index, ]
  rand_y <- training_set$y[rand_index, ,drop = FALSE]

  # Train our model
  session$run(
    train_step,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  # Calculate the loss against the training set
  loss_vec[i] <- session$run(
    loss,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  # Calculate the accuracy against the testing set
  batch_accuracy[i] <- session$run(
    accuracy,
    feed_dict = dict(
      x_data = test_set$x,
      y_target = test_set$y
    )
  )

  if (i %% 50 == 0) {
    # Every 50 steps give a visual update of how we are doing

    # Generate predictions for our input data mesh
    pred <- session$run(
      prediction,
      feed_dict = dict(
        x_data = apply(as.matrix(mesh[, c("x", "y")]), 2L, col_normalize)
    ))

    # Add the predictions to the grid data
    mesh$pred <- factor(pred, labels = c("inner", "outer")[unique(pred) + 1L])

    # Plot the grid then overlay the data points
    # NOTE: Extracting the data from a numpy array to plot is a little
    # hacky.
    plot <- ggplot(mesh, aes(x, y, fill = pred)) +
      geom_raster() +
      geom_point(
        data = data,
        aes(x, y, shape = circ),
        inherit.aes = FALSE) +
      labs(
        title = sprintf("Single layer FF NN (%i neurons) -- Step %i", hidden_neurons, i),
        subtitle = sprintf("Loss = %.2f, Accuracy = %.2f", loss_vec[i], batch_accuracy[i]),
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
