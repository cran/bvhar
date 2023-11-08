#' Summarizing VAR and VHAR with SSVS Prior Model
#' 
#' Conduct variable selection.
#' 
#' @param object `ssvsmod` object
#' @param ... not used
#' @details 
#' In each cell, variable selection can be done by giving threshold for each cell of coefficient:
#' \deqn{\lvert \phi_{i} \rvert \le 3 \tau_{0i}}
#' and
#' \deqn{\lvert \eta_{ij} \rvert \le 3 \kappa_{0ij}}
#' @return `summary.ssvsmod` object
#' @references 
#' George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881–889.
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580.
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358.
#' 
#' O’Hara, R. B., & Sillanpää, M. J. (2009). *A review of Bayesian variable selection methods: what, how and which*. Bayesian Analysis, 4(1), 85–117.
#' @export
summary.ssvsmod <- function(object, ...) {
  # coefficients-------------------------------
  # coef_mean <- object$coefficients
  coef_mean <- switch(
    object$type,
    "none" = object$coefficients,
    "const" = object$coefficients[-object$df,]
  )
  # coef_spike <- matrix(object$spec$coef_spike, ncol = object$m)
  coef_dummy <- object$pip
  # var_selection <- abs(coef_mean) <= 3 * coef_spike
  var_selection <- object$pip <= .5
  # coef_res <- ifelse(var_selection, 0L, coef_mean)
  # coef_res <- switch(
  #   object$type,
  #   "none" = ifelse(var_selection, 0L, coef_mean),
  #   "const" = rbind(ifelse(var_selection, 0L, coef_mean), object$coefficients[object$df,])
  # )
  coef_res <- switch(
    object$type,
    "none" = ifelse(var_selection, 0L, coef_mean),
    "const" = rbind(ifelse(var_selection, 0L, coef_mean), object$coefficients[object$df,])
  )
  if (object$type == "const") {
    rownames(coef_res)[object$df] <- "const"
  }
  # cholesky factor----------------------------
  chol_mean <- object$chol_posterior
  # chol_spike <- diag(object$m)
  # chol_spike[upper.tri(chol_spike, diag = FALSE)] <- object$spec$chol_spike
  # chol_selection <- abs(chol_mean) <= 3 * chol_spike
  # diag(chol_selection) <- FALSE
  # chol_res <- ifelse(chol_selection, 0L, chol_mean)
  chol_dummy <- object$omega_posterior
  chol_selection <- chol_dummy <= .5
  chol_res <- ifelse(chol_selection, 0L, chol_mean)
  # return S3 object---------------------------
  res <- list(
    call = object$call,
    process = object$process,
    p = object$p,
    m = object$m,
    type = object$type,
    coefficients = coef_res,
    posterior_mean = coef_mean,
    cholesky = chol_res,
    choose_coef = !var_selection,
    choose_chol = !chol_selection
  )
  class(res) <- c("summary.ssvsmod", "summary.bvharsp")
  res
}

#' Evaluate the Estimation Based on Frobenius Norm
#' 
#' This function computes estimation error given estimated model and true coefficient.
#' 
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param ... not used
#' @return Frobenius norm value
#' @export
fromse <- function(x, y, ...) {
  UseMethod("fromse", x)
}

#' @rdname fromse
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param ... not used
#' @details 
#' Consider the Frobenius Norm \eqn{\lVert \cdot \rVert_F}.
#' let \eqn{\hat{\Phi}} be nrow x k the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes estimation error by
#' \deqn{MSE = 100 \frac{\lVert \hat{\Phi} - \Phi \rVert_F}{nrow \times k}}
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170.
#' @export
fromse.bvharsp <- function(x, y, ...) {
  100 * norm(x$coefficients - y, type = "F") / (x$df * x$m)
}

#' Evaluate the Estimation Based on Spectral Norm Error
#' 
#' This function computes estimation error given estimated model and true coefficient.
#' 
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param ... not used
#' @return Spectral norm value
#' @export
spne <- function(x, y, ...) {
  UseMethod("spne", x)
}

#' @rdname spne
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param ... not used
#' @details 
#' Let \eqn{\lVert \cdot \rVert_2} be the spectral norm of a matrix,
#' let \eqn{\hat{\Phi}} be the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes estimation error by
#' \deqn{\lVert \hat{\Phi} - \Phi \rVert_2}
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @export
spne.bvharsp <- function(x, y, ...) {
  norm(x$coefficients - y, type = "2")
}

#' Evaluate the Estimation Based on Relative Spectral Norm Error
#' 
#' This function computes relative estimation error given estimated model and true coefficient.
#' 
#' @param x Estimated model.
#' @param y True coefficient matrix.
#' @param ... not used
#' @return Spectral norm value
#' @export
relspne <- function(x, y, ...) {
  UseMethod("relspne", x)
}

#' @rdname relspne
#' @param x Estimated model.
#' @param y Coefficient matrix to be compared.
#' @param ... not used
#' @details 
#' Let \eqn{\lVert \cdot \rVert_2} be the spectral norm of a matrix,
#' let \eqn{\hat{\Phi}} be the estimates,
#' and let \eqn{\Phi} be the true coefficients matrix.
#' Then the function computes relative estimation error by
#' \deqn{\frac{\lVert \hat{\Phi} - \Phi \rVert_2}{\lVert \Phi \rVert_2}}
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @export
relspne.bvharsp <- function(x, y, ...) {
  spne(x, y) / norm(y, type = "2")
}

#' Evaluate the Sparsity Estimation Based on Confusion Matrix
#' 
#' This function computes FDR (false discovery rate) and FNR (false negative rate) for sparse element of the true coefficients given threshold.
#' 
#' @param x Estimated model.
#' @param y True inclusion variable.
#' @param ... not used
#' @return Confusion table as following.
#' 
#' |True-estimate|Positive (0) | Negative (1) |
#' |:-----------:|:-----------:|:------------:|
#' | Positive (0) | TP | FN |
#' | Negative (1) | FP | TN |
#' @export
confusion <- function(x, y, ...) {
  UseMethod("confusion", x)
}

#' @rdname confusion
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' 
#' In this confusion matrix, positive (0) means sparsity.
#' FP is false positive, and TP is true positive.
#' FN is false negative, and FN is false negative.
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170.
#' @export
confusion.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  est <- factor(c(x$choose_coef * 1), levels = c(0L, 1L))
  truth <- ifelse(c(abs(y)) <= truth_thr, 0L, 1L) %>% factor(levels = c(0L, 1L))
  table(truth = truth, estimation = est)
}

#' Evaluate the Sparsity Estimation Based on FDR
#' 
#' This function computes false discovery rate (FDR) for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param ... not used
#' @return FDR value in confusion table
#' @export
conf_fdr <- function(x, y, ...) {
  UseMethod("conf_fdr", x)
}

#' @rdname conf_fdr
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' When using this function, the true coefficient matrix \eqn{\Phi} should be sparse.
#' False discovery rate (FDR) is computed by
#' \deqn{FDR = \frac{FP}{TP + FP}}
#' where TP is true positive, and FP is false positive.
#' @seealso [confusion()]
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170.
#' @export
conf_fdr.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[2, 1] / sum(conftab[, 1])
}

#' Evaluate the Sparsity Estimation Based on Precision
#' 
#' This function computes precision for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param ... not used
#' @return Precision value in confusion table
#' @export
conf_prec <- function(x, y, ...) {
  UseMethod("conf_prec", x)
}

#' @rdname conf_prec
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' If the element of the estimate \eqn{\hat\Phi} is smaller than some threshold,
#' it is treated to be zero.
#' Then the precision is computed by
#' \deqn{precision = \frac{TP}{TP + FP}}
#' where TP is true positive, and FP is false positive.
#' @seealso [confusion()]
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170.
#' @export
conf_prec.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[1, 1] / sum(conftab[, 1])
}

#' Evaluate the Sparsity Estimation Based on FNR
#' 
#' This function computes false negative rate (FNR) for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param ... not used
#' @return FNR value in confusion table
#' @export
conf_fnr <- function(x, y, ...) {
  UseMethod("conf_fnr", x)
}

#' @rdname conf_fnr
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' False negative rate (FNR) is computed by
#' \deqn{FNR = \frac{FN}{TP + FN}}
#' where TP is true positive, and FN is false negative.
#' @seealso [confusion()]
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170.
#' @export
conf_fnr.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[1, 2] / sum(conftab[1,])
}

#' Evaluate the Sparsity Estimation Based on Recall
#' 
#' This function computes recall for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param ... not used
#' @return Recall value in confusion table
#' @export
conf_recall <- function(x, y, ...) {
  UseMethod("conf_recall", x)
}

#' @rdname conf_recall
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' Precision is computed by
#' \deqn{recall = \frac{TP}{TP + FN}}
#' where TP is true positive, and FN is false negative.
#' @seealso [confusion()]
#' @references Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170.
#' @export
conf_recall.summary.bvharsp <- function(x, y, truth_thr = 0L, ...) {
  conftab <- confusion(x, y, truth_thr = truth_thr)
  conftab[1, 1] / sum(conftab[1,])
}

#' Evaluate the Sparsity Estimation Based on F1 Score
#' 
#' This function computes F1 score for sparse element of the true coefficients given threshold.
#' 
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param ... not used
#' @return F1 score in confusion table
#' @export
conf_fscore <- function(x, y, ...) {
  UseMethod("conf_fscore", x)
}

#' @rdname conf_fscore
#' @param x `summary.bvharsp` object.
#' @param y True inclusion variable.
#' @param truth_thr Threshold value when using non-sparse true coefficient matrix. By default, `0` for sparse matrix.
#' @param ... not used
#' @details 
#' The F1 score is computed by
#' \deqn{F_1 = \frac{2 precision \times recall}{precision + recall}}
#' @seealso [confusion()]
#' @export
conf_fscore.summary.bvharsp <- function(x, y, truth_thr = 0, ...) {
  prec_score <- conf_prec(x, y, truth_thr = truth_thr)
  rec_score <- conf_recall(x, y, truth_thr = truth_thr)
  2 * prec_score * rec_score / (prec_score + rec_score)
}