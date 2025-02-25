## ----rmdsetup, include = FALSE------------------------------------------------
knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  out.width = "70%",
  fig.align = "center",
  fig.width = 6,
  fig.asp = .618,
  error = TRUE # temporarily until investigating the error: 'recursive' must be of length 1
)
orig_opts <- options("digits")
options(digits = 3)
set.seed(1)

## ----setup--------------------------------------------------------------------
library(bvhar)

## ----etfdat-------------------------------------------------------------------
etf <- etf_vix[1:55, 1:3]
# Split-------------------------------
h <- 5
etf_eval <- divide_ts(etf, h)
etf_train <- etf_eval$train
etf_test <- etf_eval$test

## ----fitssvs------------------------------------------------------------------
(fit_ssvs <- vhar_bayes(etf_train, num_chains = 1, num_iter = 20, bayes_spec = set_ssvs(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))

## ----heatssvs-----------------------------------------------------------------
autoplot(fit_ssvs)

## ----fiths--------------------------------------------------------------------
(fit_hs <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_horseshoe(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))

## ----heaths-------------------------------------------------------------------
autoplot(fit_hs)

## ----fitmn--------------------------------------------------------------------
(fit_mn <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_bvhar(lambda = set_lambda()), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))

## ----fitng--------------------------------------------------------------------
(fit_ng <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_ng(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))

## ----fitdl--------------------------------------------------------------------
(fit_dl <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_dl(), cov_spec = set_ldlt(), include_mean = FALSE, minnesota = "longrun"))

## -----------------------------------------------------------------------------
autoplot(fit_hs, type = "trace", regex_pars = "tau")

## ----denshs-------------------------------------------------------------------
autoplot(fit_hs, type = "dens", regex_pars = "kappa", facet_args = list(dir = "v", nrow = nrow(fit_hs$coefficients)))

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

