#' Fitting Vector Heterogeneous Autoregressive Model
#' 
#' This function fits VHAR using OLS method.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param method Method to solve linear equation system.
#' (`"nor"`: normal equation (default), `"chol"`: Cholesky, and `"qr"`: HouseholderQR)
#' @details 
#' For VHAR model
#' \deqn{Y_{t} = \Phi^{(d)} Y_{t - 1} + \Phi^{(w)} Y_{t - 1}^{(w)} + \Phi^{(m)} Y_{t - 1}^{(m)} + \epsilon_t}
#' the function gives basic values.
#' @return `vhar_lm()` returns an object named `vharlse` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{fitted.values}{Fitted response values}
#'   \item{residuals}{Residuals}
#'   \item{covmat}{LS estimate for covariance matrix}
#'   \item{df}{Numer of Coefficients: 3m + 1 or 3m}
#'   \item{p}{3 (The number of terms. `vharlse` contains this element for usage in other functions.)}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - 22}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process: VHAR}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{HARtrans}{VHAR linear transformation matrix: \eqn{C_{HAR}}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' It is also a `bvharmod` class.
#' @references 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510.
#' 
#' Bubák, V., Kočenda, E., & Žikeš, F. (2011). *Volatility transmission in emerging European foreign exchange markets*. Journal of Banking & Finance, 35(11), 2829–2841.
#' 
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196.
#' @seealso 
#' * [coef.vharlse()], [residuals.vharlse()], and [fitted.vharlse()]
#' * [summary.vharlse()] to summarize VHAR model
#' * [predict.vharlse()] to forecast the VHAR process
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
vhar_lm <- function(y, har = c(5, 22), include_mean = TRUE, method = c("nor", "chol", "qr")) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  method <- match.arg(method)
  method <- switch(method, "nor" = 1, "chol" = 2, "qr" = 3)
  if (length(har) != 2 || !is.numeric(har)) {
    stop("'har' should be numeric vector of length 2.")
  }
  if (har[1] > har[2]) {
    stop("'har[1]' should be smaller than 'har[2]'.")
  }
  week <- har[1] # 5
  month <- har[2] # 22
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, month, month + 1) # 22, 23
  m <- ncol(y)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(m))
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, month, include_mean) # 22
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean) # in misc-r.R file
  # estimate Phi---------------------
  type <- ifelse(include_mean, "const", "none")
  vhar_est <- estimate_har(X0, Y0, week, month, include_mean, method)
  Phihat <- vhar_est$phihat
  colnames(Phihat) <- name_var
  rownames(Phihat) <- name_har
  # fitted values and residuals-----
  yhat <- vhar_est$fitted
  colnames(yhat) <- colnames(Y0)
  zhat <- Y0 - yhat
  # residual Covariance matrix------
  covmat <- compute_cov(zhat, nrow(Y0), nrow(Phihat)) # Sighat = z^T %*% z / (s - (3m + 1))
  colnames(covmat) <- name_var
  rownames(covmat) <- name_var
  # return as new S3 class-----------
  res <- list(
    # estimation---------------
    coefficients = Phihat,
    fitted.values = yhat, # X1 %*% Phihat
    residuals = zhat, # Y0 - X1 %*% Phihat
    covmat = covmat,
    # variables---------------
    df = nrow(Phihat), # nrow(Phihat) = 3 * m + 1 or 3 * m
    p = 3, # add for other function (df = 3m + 1 = mp + 1)
    week = week, # default: 5
    month = month, # default: 22
    m = ncol(y), # m
    obs = nrow(Y0), # s = n - 22
    totobs = nrow(y), # n
    # about model------------
    call = match.call(),
    process = "VHAR",
    type = type,
    # data-------------------
    HARtrans = vhar_est$HARtrans,
    y0 = Y0,
    design = X0,
    y = y
  )
  class(res) <- c("vharlse", "bvharmod")
  res
}