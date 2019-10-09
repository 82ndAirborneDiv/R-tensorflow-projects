# Tensorflow for R with the Estimator API
# require(tensorflow)
require(tfestimators)

mtcars_input_fn <- function(data) {
  # Returns input function for subset of data
  input_fn(data,
    features = c("drat", "mpg", "am"),
    response = "vs"
  )
}

lin_cols <- feature_columns(column_numeric("mpg"))
dnn_cols <- feature_columns(column_numeric("drat"))

# Classifier
classifier <- dnn_linear_combined_classifier(
  linear_feature_columns = lin_cols,
  dnn_feature_columns = dnn_cols,
  dnn_hidden_units = c(3L, 3L),
  dnn_optimizer = "Adagrad"
)

# Sample data for training and testing sets
idx <- sample(1:nrow(mtcars), size = 0.80 * nrow(mtcars))
train <- mtcars[idx, ]
test <- mtcars[-idx, ]

# Train the model
classifier %>% train(mtcars_input_fn(train), steps = 5)

# Evaluation
results <- classifier %>% evaluate(mtcars_input_fn(test))
# view predictions
obs <- mtcars[1:3, ]
predictions <- classifier %>% predict(mtcars_input_fn(obs))
