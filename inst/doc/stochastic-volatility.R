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

## ----setsv--------------------------------------------------------------------
set_sv()

## ----svssvs-------------------------------------------------------------------
(fit_ssvs <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_ssvs(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))

## ----hssv---------------------------------------------------------------------
(fit_hs <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_horseshoe(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))

## ----ngsv---------------------------------------------------------------------
(fit_ng <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_ng(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))

## ----dlsv---------------------------------------------------------------------
(fit_dl <- vhar_bayes(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_dl(), cov_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))

## -----------------------------------------------------------------------------
autoplot(fit_hs, type = "trace", regex_pars = "tau")

## ----denshs-------------------------------------------------------------------
autoplot(fit_hs, type = "dens", regex_pars = "tau")

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

