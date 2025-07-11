#' Forecasting Multivariate Time Series
#' 
#' Forecasts multivariate time series using given model.
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers with `n_ahead`.
#' @param ... not used
#' @section n-step ahead forecasting VAR(p):
#' See pp35 of Lütkepohl (2007).
#' Consider h-step ahead forecasting (e.g. n + 1, ... n + h).
#' 
#' Let \eqn{y_{(n)}^T = (y_n^T, ..., y_{n - p + 1}^T, 1)}.
#' Then one-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 1}^T = y_{(n)}^T \hat{B}}
#' 
#' Recursively, let \eqn{\hat{y}_{(n + 1)}^T = (\hat{y}_{n + 1}^T, y_n^T, ..., y_{n - p + 2}^T, 1)}.
#' Then two-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 2}^T = \hat{y}_{(n + 1)}^T \hat{B}}
#' 
#' Similarly, h-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + h}^T = \hat{y}_{(n + h - 1)}^T \hat{B}}
#' 
#' How about confident region?
#' Confidence interval at h-period is
#' \deqn{y_{k,t}(h) \pm z_(\alpha / 2) \sigma_k (h)}
#' 
#' Joint forecast region of \eqn{100(1-\alpha)}% can be computed by
#' \deqn{\{ (y_{k, 1}, y_{k, h}) \mid y_{k, n}(i) - z_{(\alpha / 2h)} \sigma_n(i) \le y_{n, i} \le y_{k, n}(i) + z_{(\alpha / 2h)} \sigma_k(i), i = 1, \ldots, h \}}
#' See the pp41 of Lütkepohl (2007).
#' 
#' To compute covariance matrix, it needs VMA representation:
#' \deqn{Y_{t}(h) = c + \sum_{i = h}^{\infty} W_{i} \epsilon_{t + h - i} = c + \sum_{i = 0}^{\infty} W_{h + i} \epsilon_{t - i}}
#' 
#' Then
#' 
#' \deqn{\Sigma_y(h) = MSE [ y_t(h) ] = \sum_{i = 0}^{h - 1} W_i \Sigma_{\epsilon} W_i^T = \Sigma_y(h - 1) + W_{h - 1} \Sigma_{\epsilon} W_{h - 1}^T}
#' 
#' @return `predbvhar` [class] with the following components:
#' \describe{
#'   \item{process}{object$process}
#'   \item{forecast}{forecast matrix}
#'   \item{se}{standard error matrix}
#'   \item{lower}{lower confidence interval}
#'   \item{upper}{upper confidence interval}
#'   \item{lower_joint}{lower CI adjusted (Bonferroni)}
#'   \item{upper_joint}{upper CI adjusted (Bonferroni)}
#'   \item{y}{object$y}
#' }
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @name predict
#' @importFrom stats qnorm
#' @importFrom utils tail
#' @order 1
#' @export
predict.varlse <- function(object, n_ahead, level = .05, newxreg, ...) {
  if (!is.null(eval.parent(object$call$exogen))) {
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = n_ahead)
    pred_res <- forecast_varx(
      response = object$y0,
      coef_mat = object$coefficients[-object$exogen_id, ],
      lag = object$p,
      step = n_ahead,
      include_mean = object$type == "const",
      exogen = rbind(tail(object$exogen_data, object$s), newxreg),
      exogen_coef = object$coefficients[object$exogen_id, ],
      exogen_lag = object$s
    )
  } else {
    pred_res <- forecast_var(object, n_ahead)
  }
  colnames(pred_res) <- colnames(object$y0)
  SE <- 
    compute_covmse(object, n_ahead) |> # concatenated matrix
    split.data.frame(gl(n_ahead, object$m)) |> # list of forecast MSE covariance matrix
    sapply(diag) |> 
    t() # extract only diagonal element to compute CIs
  SE <- sqrt(SE)
  colnames(SE) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n_ahead), lower.tail = FALSE)
  res <- list(
    process = object$process,
    forecast = pred_res,
    se = SE,
    lower = pred_res - z_quant * SE,
    upper = pred_res + z_quant * SE,
    lower_joint = pred_res - z_bonferroni * SE,
    upper_joint = pred_res + z_bonferroni * SE,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict
#' @param object A `vharlse` object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers with `n_ahead`.
#' @param ... not used
#' @section n-step ahead forecasting VHAR:
#' Let \eqn{T_{HAR}} is VHAR linear transformation matrix.
#' Since VHAR is the linearly transformed VAR(22),
#' let \eqn{y_{(n)}^T = (y_n^T, y_{n - 1}^T, ..., y_{n - 21}^T, 1)}.
#' 
#' Then one-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 1}^T = y_{(n)}^T T_{HAR} \hat{\Phi}}
#' 
#' Recursively, let \eqn{\hat{y}_{(n + 1)}^T = (\hat{y}_{n + 1}^T, y_n^T, ..., y_{n - 20}^T, 1)}.
#' Then two-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 2}^T = \hat{y}_{(n + 1)}^T T_{HAR} \hat{\Phi}}
#' 
#' and h-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + h}^T = \hat{y}_{(n + h - 1)}^T T_{HAR} \hat{\Phi}}
#' 
#' @references 
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174-196.
#' 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495-510.
#' @importFrom stats qnorm
#' @order 1
#' @export
predict.vharlse <- function(object, n_ahead, level = .05, newxreg, ...) {
  if (!is.null(eval.parent(object$call$exogen))) {
    if (missing(newxreg) || is.null(newxreg)) {
      stop("'newxreg' should be supplied when using VHARX model.")
    }
    if (!is.matrix(newxreg)) {
      newxreg <- as.matrix(newxreg)
    }
    if (nrow(newxreg) != n_ahead) {
      stop("Wrong row number of 'newxreg'")
    }
    pred_res <- forecast_harx(
      response = object$y0,
      coef_mat = object$coefficients[-object$exogen_id, ],
      week = object$week,
      month = object$month,
      step = n_ahead,
      include_mean = object$type == "const",
      exogen = rbind(tail(object$exogen_data, object$s), newxreg),
      exogen_coef = object$coefficients[object$exogen_id, ],
      exogen_lag = object$s
    )
  } else {
    pred_res <- forecast_vhar(object, n_ahead)
  }
  colnames(pred_res) <- colnames(object$y0)
  SE <-
    compute_covmse_har(object, n_ahead) |> # concatenated matrix
    split.data.frame(gl(n_ahead, object$m)) |> # list of forecast MSE covariance matrix
    sapply(diag) |>
    t() # extract only diagonal element to compute CIs
  SE <- sqrt(SE)
  colnames(SE) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n_ahead), lower.tail = FALSE)
  res <- list(
    process = object$process,
    forecast = pred_res,
    se = SE,
    lower = pred_res - z_quant * SE,
    upper = pred_res + z_quant * SE,
    lower_joint = pred_res - z_bonferroni * SE,
    upper_joint = pred_res + z_bonferroni * SE,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict
#' @param object Model object
#' @param n_ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param num_thread Number of threads
#' @param ... not used
#' @section n-step ahead forecasting BVAR(p) with minnesota prior:
#' Point forecasts are computed by posterior mean of the parameters.
#' See Section 3 of Bańbura et al. (2010).
#' 
#' Let \eqn{\hat{B}} be the posterior MN mean
#' and let \eqn{\hat{V}} be the posterior MN precision.
#' 
#' Then predictive posterior for each step
#' 
#' \deqn{y_{n + 1} \mid \Sigma_e, y \sim N( vec(y_{(n)}^T A), \Sigma_e \otimes (1 + y_{(n)}^T \hat{V}^{-1} y_{(n)}) )}
#' \deqn{y_{n + 2} \mid \Sigma_e, y \sim N( vec(\hat{y}_{(n + 1)}^T A), \Sigma_e \otimes (1 + \hat{y}_{(n + 1)}^T \hat{V}^{-1} \hat{y}_{(n + 1)}) )}
#' and recursively,
#' \deqn{y_{n + h} \mid \Sigma_e, y \sim N( vec(\hat{y}_{(n + h - 1)}^T A), \Sigma_e \otimes (1 + \hat{y}_{(n + h - 1)}^T \hat{V}^{-1} \hat{y}_{(n + h - 1)}) )}
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC.
#' 
#' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791-897.
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvarmn <- function(object, n_ahead, n_iter = 100L, level = .05, num_thread = 1, ...) {
  # dim_data <- object$m
  # num_chains <- object$chain
  # if (num_thread > get_maxomp()) {
  #   warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  # }
  # if (num_thread > num_chains && num_chains != 1) {
  #   warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  # }
  # alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha"))
  # pred_res <- forecast_bvar(
  #   num_chains = num_chains,
  #   var_lag = object$p,
  #   step = n_ahead,
  #   response_mat = object$y0,
  #   alpha_record = alpha_record,
  #   sig_record = as_draws_matrix(subset_draws(object$param, variable = "sigma")),
  #   include_mean = object$type == "const",
  #   nthreads = num_thread
  # )
  pred_res <- forecast_bvar(object, n_ahead, n_iter, sample.int(.Machine$integer.max, size = 1))
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  var_names <- colnames(object$y0)
  colnames(pred_mean) <- var_names
  # Predictive distribution-------------------------
  dim_data <- ncol(pred_mean)
  y_distn <-
    pred_res$predictive |>
    array(dim = c(n_ahead, dim_data, n_iter)) # 3d array: h x m x B
  # num_draw <- nrow(alpha_record) # concatenate multiple chains
  # y_distn <-
  #   pred_res |>
  #   unlist() |>
  #   array(dim = c(n_ahead, dim_data, num_draw))
  # pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  # colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  # Standard error----------------------------------
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(est_se) <- var_names
  # result------------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    # forecast = apply(y_distn, c(1, 2), mean),
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict
#' @param object Model object
#' @param n_ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param num_thread Number of threads
#' @param ... not used
#' @section n-step ahead forecasting BVHAR:
#' Let \eqn{\hat\Phi} be the posterior MN mean
#' and let \eqn{\hat\Psi} be the posterior MN precision.
#' 
#' Then predictive posterior for each step
#' 
#' \deqn{y_{n + 1} \mid \Sigma_e, y \sim N( vec(y_{(n)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n)}) )}
#' \deqn{y_{n + 2} \mid \Sigma_e, y \sim N( vec(y_{(n + 1)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n + 1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n + 1)}) )}
#' and recursively,
#' \deqn{y_{n + h} \mid \Sigma_e, y \sim N( vec(y_{(n + h - 1)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n + h - 1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n + h - 1)}) )}
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvharmn <- function(object, n_ahead, n_iter = 100L, level = .05, num_thread = 1, ...) {
  pred_res <- forecast_bvharmn(object, n_ahead, n_iter, sample.int(.Machine$integer.max, size = 1))
  # dim_data <- object$m
  # num_chains <- object$chain
  # if (num_thread > get_maxomp()) {
  #   warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  # }
  # if (num_thread > num_chains && num_chains != 1) {
  #   warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  # }
  # phi_record <- as_draws_matrix(subset_draws(object$param, variable = "phi"))
  # pred_res <- forecast_bvharmn(
  #   num_chains = num_chains,
  #   month = object$month,
  #   step = n_ahead,
  #   response_mat = object$y0,
  #   har_trans = object$HARtrans,
  #   phi_record = phi_record,
  #   sig_record = as_draws_matrix(subset_draws(object$param, variable = "sigma")),
  #   include_mean = object$type == "const",
  #   nthreads = num_thread
  # )
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  var_names <- colnames(object$y0)
  # colnames(pred_mean) <- var_names
  # Predictive distribution-------------------------
  dim_data <- ncol(pred_mean)
  y_distn <- 
    pred_res$predictive |> 
    array(dim = c(n_ahead, dim_data, n_iter)) # 3d array: h x m x B
  # num_draw <- nrow(phi_record) # concatenate multiple chains
  # y_distn <-
  #   pred_res |>
  #   unlist() |>
  #   array(dim = c(n_ahead, dim_data, num_draw))
  # pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  # Standard error----------------------------------
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(est_se) <- var_names
  # result------------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    # forecast = apply(y_distn, c(1, 2), mean),
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict
#'
#' @param object Model object
#' @param n_ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param num_thread Number of threads
#' @param ... not used
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvarflat <- function(object, n_ahead, n_iter = 100L, level = .05, num_thread = 1, ...) {
  # dim_data <- object$m
  # num_chains <- object$chain
  # if (num_thread > get_maxomp()) {
  #   warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  # }
  # if (num_thread > num_chains && num_chains != 1) {
  #   warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  # }
  # alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha"))
  # pred_res <- forecast_bvar(
  #   num_chains = num_chains,
  #   var_lag = object$p,
  #   step = n_ahead,
  #   response_mat = object$y0,
  #   alpha_record = alpha_record,
  #   sig_record = as_draws_matrix(subset_draws(object$param, variable = "sigma")),
  #   include_mean = object$type == "const",
  #   nthreads = num_thread
  # )
  pred_res <- forecast_bvar(object, n_ahead, n_iter, sample.int(.Machine$integer.max, size = 1))
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  var_names <- colnames(object$y0)
  # colnames(pred_mean) <- var_names
  # Predictive distribution-------------------------
  dim_data <- ncol(pred_mean)
  y_distn <-
    pred_res$predictive |>
    array(dim = c(n_ahead, dim_data, n_iter)) # 3d array: h x m x B
  # num_draw <- nrow(alpha_record) # concatenate multiple chains
  # y_distn <-
  #   pred_res |>
  #   unlist() |>
  #   array(dim = c(n_ahead, dim_data, num_draw))
  # pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  # Standard error----------------------------------
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(est_se) <- var_names
  # result------------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers with `n_ahead`.
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param num_thread Number of threads
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' Give CI level (e.g. `.05`) instead of `TRUE` to use credible interval across MCMC for restriction.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param warn Give warning for stability of each coefficients record. By default, `FALSE`.
#' @param ... not used
#' @references Korobilis, D. (2013). *VAR FORECASTING USING BAYESIAN VARIABLE SELECTION*. Journal of Applied Econometrics, 28(2).
#' @importFrom posterior subset_draws as_draws_matrix
#' @importFrom stats median
#' @order 1
#' @export
predict.bvarldlt <- function(object, n_ahead, level = .05, newxreg, stable = FALSE, num_thread = 1, sparse = FALSE, med = FALSE, warn = FALSE, ...) {
  dim_data <- object$m
  num_chains <- object$chain
  alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha"))
  if (warn) {
    is_stable <- apply(
      alpha_record,
      1,
      function(x) {
        eigen_vals <-
          matrix(x, ncol = object$m) |>
          compute_stablemat() |>
          eigen()
        all(Mod(eigen_vals$values) < 1)
      }
    )
    if (any(!is_stable)) {
      warning("Some alpha records are unstable, so add burn-in")
    }
  }
  if (object$type == "const") {
    alpha_record <- cbind(alpha_record, as_draws_matrix(subset_draws(object$param, variable = "c")))
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  # prior_nm <- object$spec$prior
  # ci_lev <- NULL
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
    # prior_nm <- "ci"
  }
  fit_ls <- get_records(object, TRUE)
  # prior_type <- switch(prior_nm,
  #   "ci" = 0,
  #   "Minnesota" = 1,
  #   "SSVS" = 2,
  #   "Horseshoe" = 3,
  #   "MN_Hierarchical" = 4,
  #   "NG" = 5,
  #   "DL" = 6,
  #   "GDP" = 7
  # )
  if (!is.null(eval.parent(object$call$exogen))) {
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = n_ahead)
    pred_res <- forecast_bvarxldlt(
      num_chains = num_chains,
      var_lag = object$p,
      step = n_ahead,
      response_mat = object$y,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      exogen = rbind(tail(object$exogen_data, object$s), newxreg),
      exogen_lag = object$s,
      stable = stable,
      nthreads = num_thread
    )
  } else {
    pred_res <- forecast_bvarldlt(
      num_chains = num_chains,
      var_lag = object$p,
      step = n_ahead,
      response_mat = object$y,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      stable = stable,
      nthreads = num_thread
    )
  }
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_draw <- nrow(alpha_record) # concatenate multiple chains
  y_distn <-
    pred_res |>
    unlist() |>
    array(dim = c(n_ahead, dim_data, num_draw))
  if (med) {
    pred_mean <- apply(y_distn, c(1, 2), median)
  } else {
    pred_mean <- apply(y_distn, c(1, 2), mean)
  }
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  res$object <- object
  class(res) <- c("predsv", "predbvhar")
  res
}

#' @rdname predict
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers with `n_ahead`.
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param num_thread Number of threads
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' Give CI level (e.g. `.05`) instead of `TRUE` to use credible interval across MCMC for restriction.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param warn Give warning for stability of each coefficients record. By default, `FALSE`.
#' @param ... not used
#' @importFrom posterior subset_draws as_draws_matrix
#' @order 1
#' @export
predict.bvharldlt <- function(object, n_ahead, level = .05, newxreg, stable = FALSE, num_thread = 1, sparse = FALSE, med = FALSE, warn = FALSE, ...) {
  dim_data <- object$m
  num_chains <- object$chain
  phi_record <- as_draws_matrix(subset_draws(object$param, variable = "phi"))
  if (warn) {
    is_stable <- apply(
      phi_record,
      1,
      function(x) {
        coef <- t(object$HARtrans[1:(object$p * dim_data), 1:(object$month * dim_data)]) %*% matrix(x, ncol = object$m)
        eigen_vals <-
          coef |>
          compute_stablemat() |>
          eigen()
        all(Mod(eigen_vals$values) < 1)
      }
    )
    if (any(!is_stable)) {
      warning("Some phi records are unstable, so add burn-in")
    }
  }
  if (object$type == "const") {
    phi_record <- cbind(phi_record, as_draws_matrix(subset_draws(object$param, variable = "c")))
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  # prior_nm <- object$spec$prior
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
    # prior_nm <- "ci"
  }
  fit_ls <- get_records(object, TRUE)
  # prior_type <- switch(prior_nm,
  #   "ci" = 0,
  #   "Minnesota" = 1,
  #   "SSVS" = 2,
  #   "Horseshoe" = 3,
  #   "MN_Hierarchical" = 4,
  #   "NG" = 5,
  #   "DL" = 6,
  #   "GDP" = 7
  # )
  if (!is.null(eval.parent(object$call$exogen))) {
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = n_ahead)
    pred_res <- forecast_bvharxldlt(
      num_chains = num_chains,
      month = object$month,
      step = n_ahead,
      response_mat = object$y,
      HARtrans = object$HARtrans,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      exogen = rbind(tail(object$exogen_data, object$s), newxreg),
      exogen_lag = object$s,
      stable = stable,
      nthreads = num_thread
    )
  } else {
    pred_res <- forecast_bvharldlt(
      num_chains = num_chains,
      month = object$month,
      step = n_ahead,
      response_mat = object$y,
      HARtrans = object$HARtrans,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      stable = stable,
      nthreads = num_thread
    )
  }
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_draw <- nrow(phi_record) # concatenate multiple chains
  y_distn <-
    pred_res |> 
    unlist() |> 
    array(dim = c(n_ahead, dim_data, num_draw))
  if (med) {
    pred_mean <- apply(y_distn, c(1, 2), median)
  } else {
    pred_mean <- apply(y_distn, c(1, 2), mean)
  }
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  res$object <- object
  class(res) <- c("predldlt", "predbvhar")
  res
}

#' @rdname predict
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers with `n_ahead`.
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param num_thread Number of threads
#' @param use_sv Use SV term
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' Give CI level (e.g. `.05`) instead of `TRUE` to use credible interval across MCMC for restriction.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param warn Give warning for stability of each coefficients record. By default, `FALSE`.
#' @param ... not used
#' @references
#' Korobilis, D. (2013). *VAR FORECASTING USING BAYESIAN VARIABLE SELECTION*. Journal of Applied Econometrics, 28(2).
#' 
#' Huber, F., Koop, G., & Onorante, L. (2021). *Inducing Sparsity and Shrinkage in Time-Varying Parameter Models*. Journal of Business & Economic Statistics, 39(3), 669-683.
#' @importFrom posterior subset_draws as_draws_matrix
#' @order 1
#' @export
predict.bvarsv <- function(object, n_ahead, level = .05, newxreg, stable = FALSE, num_thread = 1, use_sv = TRUE, sparse = FALSE, med = FALSE, warn = FALSE, ...) {
  dim_data <- object$m
  num_chains <- object$chain
  alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha"))
  if (warn) {
    is_stable <- apply(
      alpha_record,
      1,
      function(x) {
        eigen_vals <-
          matrix(x, ncol = object$m) |>
          compute_stablemat() |>
          eigen()
        all(Mod(eigen_vals$values) < 1)
      }
    )
    if (any(!is_stable)) {
      warning("Some alpha records are unstable, so add burn-in")
    }
  }
  if (object$type == "const") {
    alpha_record <- cbind(alpha_record, as_draws_matrix(subset_draws(object$param, variable = "c")))
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  # prior_nm <- object$spec$prior
  # ci_lev <- NULL
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
    # prior_nm <- "ci"
  }
  fit_ls <- get_records(object, TRUE)
  # prior_type <- switch(prior_nm,
  #   "ci" = 0,
  #   "Minnesota" = 1,
  #   "SSVS" = 2,
  #   "Horseshoe" = 3,
  #   "MN_Hierarchical" = 4,
  #   "NG" = 5,
  #   "DL" = 6,
  #   "GDP" = 7
  # )
  if (!is.null(eval.parent(object$call$exogen))) {
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = n_ahead)
    pred_res <- forecast_bvarxsv(
      num_chains = num_chains,
      var_lag = object$p,
      step = n_ahead,
      response_mat = object$y,
      sv = use_sv,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      exogen = rbind(tail(object$exogen_data, object$s), newxreg),
      exogen_lag = object$s,
      stable = stable,
      nthreads = num_thread
    )
  } else {
    pred_res <- forecast_bvarsv(
      num_chains = num_chains,
      var_lag = object$p,
      step = n_ahead,
      response_mat = object$y,
      sv = use_sv,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      stable = stable,
      nthreads = num_thread
    )
  }
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_draw <- nrow(alpha_record) # concatenate multiple chains
  y_distn <-
    pred_res |> 
    unlist() |> 
    array(dim = c(n_ahead, dim_data, num_draw))
  if (med) {
    pred_mean <- apply(y_distn, c(1, 2), median)
  } else {
    pred_mean <- apply(y_distn, c(1, 2), mean)
  }
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  res$object <- object
  class(res) <- c("predsv", "predbvhar")
  res
}

#' @rdname predict
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers with `n_ahead`.
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param num_thread Number of threads
#' @param use_sv Use SV term
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' Give CI level (e.g. `.05`) instead of `TRUE` to use credible interval across MCMC for restriction.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param warn Give warning for stability of each coefficients record. By default, `FALSE`.
#' @param ... not used
#' @importFrom posterior subset_draws as_draws_matrix
#' @order 1
#' @export
predict.bvharsv <- function(object, n_ahead, level = .05, newxreg, stable = FALSE, num_thread = 1, use_sv = TRUE, sparse = FALSE, med = FALSE, warn = FALSE, ...) {
  dim_data <- object$m
  num_chains <- object$chain
  phi_record <- as_draws_matrix(subset_draws(object$param, variable = "phi"))
  if (warn) {
    is_stable <- apply(
      phi_record,
      1,
      function(x) {
        coef <- t(object$HARtrans[1:(object$p * dim_data), 1:(object$month * dim_data)]) %*% matrix(x, ncol = object$m)
        eigen_vals <-
          coef |>
          compute_stablemat() |>
          eigen()
        all(Mod(eigen_vals$values) < 1)
      }
    )
    if (any(!is_stable)) {
      warning("Some phi records are unstable, so add burn-in")
    }
  }
  if (object$type == "const") {
    phi_record <- cbind(phi_record, as_draws_matrix(subset_draws(object$param, variable = "c")))
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  # prior_nm <- object$spec$prior
  # ci_lev <- NULL
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
    # prior_nm <- "ci"
  }
  fit_ls <- get_records(object, TRUE)
  # prior_type <- switch(prior_nm,
  #   "ci" = 0,
  #   "Minnesota" = 1,
  #   "SSVS" = 2,
  #   "Horseshoe" = 3,
  #   "MN_Hierarchical" = 4,
  #   "NG" = 5,
  #   "DL" = 6,
  #   "GDP" = 7
  # )
  if (!is.null(eval.parent(object$call$exogen))) {
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = n_ahead)
    pred_res <- forecast_bvharxsv(
      num_chains = num_chains,
      month = object$month,
      step = n_ahead,
      response_mat = object$y,
      HARtrans = object$HARtrans,
      sv = use_sv,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      exogen = rbind(tail(object$exogen_data, object$s), newxreg),
      exogen_lag = object$s,
      stable = stable,
      nthreads = num_thread
    )
  } else {
    pred_res <- forecast_bvharsv(
      num_chains = num_chains,
      month = object$month,
      step = n_ahead,
      response_mat = object$y,
      sv = use_sv,
      sparse = sparse,
      level = ci_lev,
      fit_record = fit_ls,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      include_mean = object$type == "const",
      stable = stable,
      nthreads = num_thread
    )
  }
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_draw <- nrow(phi_record) # concatenate multiple chains
  y_distn <-
    pred_res |> 
    unlist() |> 
    array(dim = c(n_ahead, dim_data, num_draw))
  if (med) {
    pred_mean <- apply(y_distn, c(1, 2), median)
  } else {
    pred_mean <- apply(y_distn, c(1, 2), mean)
  }
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  res$object <- object
  class(res) <- c("predsv", "predbvhar")
  res
}
