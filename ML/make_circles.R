#' @title SK-learn.datasets make_circles() implemented in R
#' @description
#' Make a large circle containing a smaller circle in 2d.
#' A simple toy dataset to visualize clustering and classification algorithms.
#' Originally implemented in Scikit-learn.
#'
#' @param samples (int) The total number of points generated.
#' @param shuffle (bool) Whether to shuffle the samples.
#' @param noise (numeric) Standard deviation of Gaussian noise added to the data.
#' @param state (int) Seed for RNG scope
#' @param factor (numeric) Scale factor between inner and outer circle (<1).

#' @return data.frame of dim [samples, 3] containing x and y coordinates along with an indicator column for outer and inner values
#' @importFrom stats rnorm

#' @export
make_circles <- function(samples = 100L, shuffle = TRUE, noise = NULL, state = NULL, factor = 0.8) {

  if (! is.null(state)) {
    set.seed(state)
  }

  points <- seq(0, 2 * pi, length.out = floor(samples / 2) + 1L)[-1]

  outer_circ_x <- cos(points)
  outer_circ_y <- sin(points)
  inner_circ_x <- outer_circ_x * factor
  inner_circ_y <- outer_circ_y * factor

  out <- data.frame(
    x = c(outer_circ_x, inner_circ_x),
    y = c(outer_circ_y, inner_circ_y),
    circ = rep(c("outer", "inner"), each = floor(samples / 2))
  )

  if (! is.null(noise)) {
    out[["x"]] <- out[["x"]] + rnorm(n = samples, sd = noise)
    out[["y"]] <- out[["y"]] + rnorm(n = samples, sd = noise)
  }

  if (shuffle) {
    out <- out[sample(seq(samples), samples, replace = FALSE), ]
  }

  return(out)
}
