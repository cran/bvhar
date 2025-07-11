#' Fitting Vector Heterogeneous Autoregressive Model
#' 
#' This function fits VHAR using OLS method.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param exogen Exogenous variables
#' @param s Lag of exogeneous variables in VHARX. By default, `s = 0`.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param method Method to solve linear equation system.
#' (`nor`: normal equation (default), `chol`: Cholesky, and `qr`: HouseholderQR)
#' @details 
#' For VHAR model
#' 
#' \deqn{Y_{t} = \Phi^{(d)} Y_{t - 1} + \Phi^{(w)} Y_{t - 1}^{(w)} + \Phi^{(m)} Y_{t - 1}^{(m)} + \epsilon_t}
#' 
#' the function gives basic values.
#' @return `vhar_lm()` returns an object named `vharlse` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{fitted.values}{Fitted response values}
#'   \item{residuals}{Residuals}
#'   \item{covmat}{LS estimate for covariance matrix}
#'   \item{df}{Numer of Coefficients}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `month`}
#'   \item{y0}{Multivariate response matrix}
#'   \item{p}{3 (The number of terms. `vharlse` contains this element for usage in other functions.)}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{totobs}{Total number of the observation}
#'   \item{process}{Process: VHAR}
#'   \item{type}{include constant term (`const`) or not (`none`)}
#'   \item{HARtrans}{VHAR linear transformation matrix}
#'   \item{design}{Design matrix of VAR(`month`)}
#'   \item{y}{Raw input}
#'   \item{method}{Solving method}
#'   \item{call}{Matched call}
#' }
#' It is also a `bvharmod` class.
#' @references 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495-510.
#' 
#' Bubák, V., Kočenda, E., & Žikeš, F. (2011). *Volatility transmission in emerging European foreign exchange markets*. Journal of Banking & Finance, 35(11), 2829-2841.
#' 
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174-196.
#' @seealso 
#' * [coef.vharlse()], [residuals.vharlse()], and [fitted.vharlse()]
#' * [summary.vharlse()] to summarize VHAR model
#' @examples 
#' # Perform the function using etf_vix dataset
#' fit <- vhar_lm(y = etf_vix)
#' class(fit)
#' str(fit)
#' 
#' # Extract coef, fitted values, and residuals
#' coef(fit)
#' head(residuals(fit))
#' head(fitted(fit))
#' @order 1
#' @export
vhar_lm <- function(y, har = c(5, 22), exogen = NULL, s = 0, include_mean = TRUE, method = c("nor", "chol", "qr")) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  method <- match.arg(method)
  method_fit <- switch(method, "nor" = 1, "chol" = 2, "qr" = 3)
  if (length(har) != 2 || !is.numeric(har)) {
    stop("'har' should be numeric vector of length 2.")
  }
  if (har[1] > har[2]) {
    stop("'har[1]' should be smaller than 'har[2]'.")
  }
  week <- har[1] # 5
  month <- har[2] # 22
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(ncol(y)))
  }
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean)
  if (!is.null(exogen)) {
    if (!is.matrix(exogen)) {
      exogen <- as.matrix(exogen)
    }
    if (!is.null(colnames(exogen))) {
      name_exogen <- colnames(exogen)
    } else {
      name_exogen <- paste0("x", seq_len(ncol(exogen)))
    }
    # if (include_mean) {
    #   # append name_lag before const
    #   name_har <- c(
    #     name_har[-length(name_har)],
    #     concatenate_colnames(name_exogen, c("day", "week", "month"), TRUE)
    #   )
    # } else {
    #   name_har <- c(
    #     name_har,
    #     concatenate_colnames(name_exogen, c("day", "week", "month"), FALSE)
    #   )
    # }
    res <- estimate_harx(y, exogen, week, month, s, include_mean, method_fit)
    # res$exogen_id <- length(name_har) + 1:(3 * ncol(exogen)) # row index for exogen in coefficient
    res$exogen_id <- length(name_har) + 1:((s + 1) * ncol(exogen)) # row index for exogen in coefficient
    # res$exogen_colid <- res$month * ncol(y) + 1:(res$month * ncol(exogen)) # col index for exogen in har transformation matrix
    # if (include_mean) {
    #   res$exogen_colid <- res$exogen_colid + 1
    # }
    name_har <- c(
      name_har,
      concatenate_colnames(name_exogen, 0:s, FALSE)
    )
    res$exogen_data <- exogen
    res$s <- s
    res$exogen_m <- ncol(exogen)
    # res$exogen <- TRUE
    # res$exogen_id <- 3 * ncol(y) + 1:(3 * ncol(exogen)) # row index for exogen in coefficient
    # res$exogen_colid <- res$month * ncol(y) + 1:(res$month * ncol(exogen)) # col index for exogen in har transformation matrix
  } else {
    res <- estimate_har(y, week, month, include_mean, method_fit)
    # res$exogen <- FALSE
  }
  colnames(res$y) <- name_var
  colnames(res$y0) <- name_var
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  colnames(res$covmat) <- name_var
  rownames(res$covmat) <- name_var
  # return as new S3 class-----------
  res$method <- method
  res$call <- match.call()
  class(res) <- c("vharlse", "olsmod", "bvharmod")
  res
}