#' @rdname var_lm
#' @param x A `varlse` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.varlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: B1, ..., Bp
  # bhat_mat <- split_coef(x)
  if (!is.null(eval.parent(x$call$exogen))) {
    ols_coef <- split_endog_coef(x$coefficients[-x$exogen_id, ], x$p, x$m)
  } else {
    ols_coef <- split_endog_coef(x$coefficients, x$p, x$m)
  }
  cat(sprintf("VAR(%i) Estimation using least squares\n", x$p))
  cat("====================================================\n\n")
  for (i in 1:(x$p)) {
    cat(sprintf("LSE for A%i:\n", i))
    # B1, ..., Bp--------------------
    print.default(
      ols_coef[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  if (!is.null(eval.parent(x$call$exogen))) {
    exog_coef <- split_exogen_coef(x$coefficients, x$exogen_id, x$s, x$exogen_m)
    for (i in seq_len(x$s + 1)) {
      cat(sprintf("LSE for exogenous B%i:\n", i - 1))
      print.default(
        exog_coef[[i]],
        digits = digits,
        print.gap = 2L,
        quote = FALSE
      )
      cat("\n\n")
    }
  }
  # const term----------------------
  if (x$type == "const") {
    # intercept <- x$coefficients[x$df,]
    intercept <- x$coefficients[x$p * x$m + 1,]
    cat("LSE for constant:\n")
    print.default(
      intercept,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  cat("--------------------------------------------------\n")
  cat("*_j of the Coefficient matrix: corresponding to the j-th VAR lag\n\n")
  invisible(x)
}

#' @rdname var_lm
#' @exportS3Method knitr::knit_print
knit_print.varlse <- function(x, ...) {
  print(x)
}

#' @rdname summary.varlse
#' @param x \code{summary.varlse} object
#' @param digits digit option to print
#' @param signif_code Check significant rows (Default: `TRUE`)
#' @param ... not used
#' @importFrom tidyr separate
#' @importFrom dplyr case_when
#' @order 2
#' @export
print.summary.varlse <- function(x, digits = max(3L, getOption("digits") - 3L), signif_code = TRUE, ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # for VAR(p)--------------------------------------------------------------------
  cat(sprintf("VAR(p = %i)\n", x$p))
  cat("====================================================\n")
  # variables--------------------------
  cat("Variables: ")
  cat(paste(x$names, collapse = ", "))
  if (x$type == "const") {
    cat("\nwith const added\n")
  }
  cat(
    paste("Number of variables:", "m =", length(x$names))
  )
  cat("\n")
  # obs num----------------------------
  cat(
    paste("Observation size:", "T =", x$totobs)
  )
  cat("\n")
  cat(
    paste("Number of sample used for fitting:", "n = T - p =", x$obs)
  )
  # stability--------------------------
  cat("\n\nCharacteristic polynomial roots:\n")
  print(x$roots)
  cat(
    paste("The process is", ifelse(x$is_stable, "stable", "not stable"))
  )
  # coefficients-----------------------
  coef_mat <- x$coefficients
  dim_data <- ncol(x$covmat)
  dim_design <- nrow(coef_mat) / dim_data
  coef_mat <- 
    coef_mat |> 
    separate(term, into = c("term", "variable"), sep = "\\.") |> 
    split.data.frame(f = gl(dim_data, dim_design))
  if (signif_code) {
    sig_star <- numeric(dim_design)
    p_val <- numeric(dim_design)
  }
  cat("\n====================================================\n")
  for (i in 1:length(coef_mat)) {
    cat(paste0(unique(coef_mat[[i]]$variable), " variable", ":\n"))
    if (signif_code) {
      p_val <- coef_mat[[i]][, "p.value"]
      sig_star <- case_when(
        p_val <= .001 ~ "***",
        p_val > .001 & p_val <= .01 ~ "**",
        p_val > .01 & p_val <= .05 ~ "*",
        p_val > .05 & p_val <= .1 ~ ".",
        p_val > .1 ~ " "
      )
      coef_mat[[i]][, " "] <- sig_star
    }
    # print Ai----------------------
    print(
      coef_mat[[i]][,-2], # without variable name column since it is printed in the header
      digits = digits,
      print.gap = 2L,
      quote = FALSE,
      right = FALSE,
      row.names = FALSE
    )
    cat("\n")
  }
  if (signif_code) {
    cat(paste0("---\n", "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", "\n"))
  }
  cat("\n")
  # df---------------------------------------
  cat("------------------------------------------------------\n")
  cat(paste("Degrees of freedom:", "df =", x$df))
  cat("\n\n")
  # cov and corr-----------------------------
  cat("LS Estimate for Covariance matrix:\n")
  print(x$covmat)
  cat("\nLS Estimate for Correlation matrix:\n")
  print(x$corrmat)
  # information criteria--------------------
  cat("\n====================================================\n")
  cat("log-likelihood:\n")
  print(x$log_lik)
  cat("Information criteria:\n")
  print(x$ic)
}

#' @rdname summary.varlse
#' @exportS3Method knitr::knit_print
knit_print.summary.varlse <- function(x, ...) {
  print(x)
}
