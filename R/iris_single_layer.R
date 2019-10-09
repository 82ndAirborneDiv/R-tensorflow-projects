# =========================================================================== #
# Feed-forward Neural Network (MLP) classifier of the iris dataset            #
# (somewhat) adapted from the wonderful Tensorflow Cookbook by Nick McClure   #
# =========================================================================== #
# NOTE: This specific example isn't found in the Tensorflow Cookbook,
# but I wanted to make a pretty much verbatim comparison to the multi-class SVM
# so there you go.

# === Lib ======================================================================
library(reticulate)
library(tensorflow)
library(ggplot2)
library(gridExtra)

# Adjust as desired
batch_size <- 50L
hidden_nodes <- 10L
iter_length <- 1000L
features <- c("Sepal.Length", "Petal.Length")
species <- c("setosa", "versicolor", "virginica")

# === Data in ==================================================================
# Pull two features out of the iris dataset to classify based on
x_vals <- as.matrix(iris[, features])

# Create 1-hot labels for each species
y_vals <- vapply(
  species, function(x) {
  ifelse(iris[["Species"]] == x, 1L, 0L)
}, integer(dim(iris)[1L]))

# Create test and training example sets
train_idx <- sample(seq(dim(x_vals)[1L]), size = floor(dim(x_vals)[1L] * 0.8), replace = FALSE)

x_train <- x_vals[train_idx, ]
x_test <- x_vals[-train_idx, ]
y_train <- y_vals[train_idx, ]
y_test <- y_vals[-train_idx, ]

# --- Plot data ----------------------------------------------------------------
# NOTE: We will make a plot to see how the class boundries change
# as the model trains.
mesh <- expand.grid(
  x = seq(min(iris[, features[1]]), max(iris[, features[1]]), by = 0.02),
  y = seq(min(iris[, features[2]]), max(iris[, features[2]]), by = 0.02)
)

points <- data.frame(
  x = x_test[, features[1]],
  y = x_test[, features[2]],
  lab = iris[["Species"]][-train_idx]
)

# -- Normalize data ------------------------------------------------------------
# Define function to normalize the examples coming in between 0 and 1
col_normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Normalize test and training data
x_train <- apply(x_train, 2L, col_normalize)
x_test <- apply(x_train, 2L, col_normalize)

# === Model ====================================================================

# --- Define input layer -------------------------------------------------------
with(tf$name_scope("input"), {
  x_data <- tf$placeholder(
    tf$float32,
    shape(NULL, length(features)),
     name = "x_data"
   )

  y_target <- tf$placeholder(
    tf$float32,
    shape(NULL, 3L),
    name = "y_labels"
  )
})

# --- Define hidden layer ------------------------------------------------------
# NOTE: Here, we take in our x_data, multiply it by some weight matrix
# and add a bias. The result will be n values, where n is the number of
# hidden nodes
with(tf$name_scope("Hidden"), {
  A1 <- tf$Variable(tf$random_normal(shape(length(features), hidden_nodes)))
  b1 <- tf$Variable(tf$random_normal(shape(hidden_nodes)))
  hid_out <- tf$nn$relu(tf$add(tf$matmul(x_data, A1), b1))
})

# --- Define output layer ------------------------------------------------------
# NOTE: To perform a classification on the three iris species, we use
# softmax regression to return three values representing the probability
# of each class. We then take the class with the highest probability and
# compare that to the 1-hot label to determine if we've got it right.
with(tf$name_scope("output"), {
  A2 <- tf$Variable(tf$random_normal(shape(hidden_nodes, 3L)))
  b2 <- tf$Variable(tf$random_normal(shape(3L)))
  soft_out <- tf$nn$softmax(tf$matmul(hid_out, A2) + b2)
  prediction <- tf$argmax(soft_out, 1L)
})

# --- Define loss function -----------------------------------------------------
# NOTE: here we calculate cross-entropy to compare the discrete distributions
# of our labels to our predictions.
loss <- tf$reduce_mean(
  -tf$reduce_sum(y_target * tf$log(soft_out), reduction_indices = 1L)
)

# --- Declare our training method ----------------------------------------------
# NOTE: I use Adam here, but you could use gradient descent or whatever you want
opt <- tf$train$AdamOptimizer(learning_rate = 1e-2)
train_step <- opt$minimize(loss)

# Define our accuracy
correct_prediction <- tf$equal(prediction, tf$argmax(y_target, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Initialize variables and start session
init <- tf$global_variables_initializer()
session <- tf$Session()
session$run(init)

# === Train ====================================================================
# Allocate vectors to store accuracy and loss values
loss_vec <- vector(mode = "numeric", length = iter_length)
batch_accuracy <- vector(mode = "numeric", length = iter_length)

# Training loop
# Run the model however many iterations
for (i in seq_len(iter_length)) {
  # Sample some random examples
  rand_index <- sample(seq(dim(x_train)[1L]), batch_size)
  rand_x <- x_train[rand_index, ]
  rand_y <- y_train[rand_index, ]

  # Train against sampled training examples
  session$run(
    train_step,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  # Compute loss
  loss_vec[i] <- session$run(
    loss,
    feed_dict = dict(
      x_data = rand_x,
      y_target = rand_y
    )
  )

  # Compute accuracy against test data
  batch_accuracy[i] <- session$run(
    accuracy,
    feed_dict = dict(
      x_data = x_train,
      y_target = y_train
    )
  )

  if (i %% 50 == 0) {
    # Every 50 iterations, give an update on how we're progressing

    # Get predicted values for the whole input space
    # NOTE: We have to normalize the features here too
    # this is actually quite sloppy, since we normalize every 50 iterations.
    val <- session$run(
      prediction,
      feed_dict = dict(
        x_data = vapply(mesh[, seq_along(features)], col_normalize, numeric(dim(mesh)[1L])
      )))

    # Append predicted label to the mesh
    mesh$pred <- factor(val, labels = species)

    # Plot the mesh and overlay the training examples
    fit_plot <- ggplot(mesh, aes(x, y, fill = pred)) +
      geom_raster() +
      geom_point(
        data = points,
        aes(x, y, shape = lab),
        inherit.aes = FALSE
      ) +
      labs(
        x = features[1],
        y = features[2],
        fill = "predicted",
        shape = "actual",
        title = sprintf("Iris Feed Forward NN (%i neurons) -- Step %i", hidden_nodes, i),
        subtitle = sprintf("Loss = %.2f, Accuracy = %.2f", loss_vec[i], batch_accuracy[i] * 100)
      ) +
      theme_bw()

    print(fit_plot)
  }
}

# === Summary plot =============================================================
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
    y = "Accuracy (%)"
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
grid.arrange(fit_plot, accuracy_plot, loss_plot, ncol = 3)
