# ========================================================================= #
# General Utility functions which may prove helpful                         #
# ========================================================================= #

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


#' @title Sample points on a larger sphere enclosing a smaller sphere
#' @description
#' Dataset generation function to produce data points for two enclosed spheres
#' a simple toy dataset for classification in 3 dimensions. A logical extension
#' to the \code{make_circles()} function within SK-learn.
#'
#' @param samples (int) The total number of points generated.
#' @param shuffle (bool) Whether to shuffle the samples.
#' @param noise (numeric) Standard deviation of Gaussian noise added to the data.
#' @param state (int) Seed for RNG scope
#' @param factor (numeric) Scale factor between inner and outer sphere (<1).

#' @return data.frame of dim [samples, 4] containing x and y coordinates along with an indicator column for outer and inner values
#' @importFrom stats rnorm runif

#' @export
make_spheres <- function(samples = 1000L, shuffle = TRUE, noise = NULL, state = NULL, factor = 0.8) {

  if (! is.null(state)) {
    set.seed(state)
  }

  # Sample phi and theta from a uniform distribution [0 to 2pi]
  phi <- runif(n = floor(samples / 2), min = 0, max = 2 * pi)
  theta <- runif(n = floor(samples / 2), min = 0, max = 2 * pi)

  # Could also use Marsaglia (1972) method for sampling points on the surface of
  # an n-ball, where x1, x2, ..., xn are sampled from independent uniform distributions
  # (-1, 1), and rejecting any points such that x1 + x2 + ... + xn >= 1.
  # and using the following equations:
  # x	=	2 * x1 * sqrt(1 - x1^2 - x2^2)
  # y	=	2 * x2 * sqrt(1 - x1^2 - x2^2)
  # z	=	1 - 2(x1^2 + x2^2)

   outer <- data.frame(
    x = sqrt(1 - (cos(phi) ^ 2)) * cos(theta),
    y = sqrt(1 - (cos(phi) ^ 2)) * sin(theta),
    z = cos(phi)
  )

  # Create inner points by multiplying outer by a scale factor
  inner <- outer * factor

  out <- rbind(outer, inner)

  out$id <- rep(c("outer", "inner"), each = floor(samples / 2))


  if (! is.null(noise)) {
    out[["x"]] <- out[["x"]] + rnorm(n = samples, sd = noise)
    out[["y"]] <- out[["y"]] + rnorm(n = samples, sd = noise)
    out[["z"]] <- out[["z"]] + rnorm(n = samples, sd = noise)
  }

  if (shuffle) {
    out <- out[sample(seq(samples), samples, replace = FALSE), ]
  }

  return(out)
}
