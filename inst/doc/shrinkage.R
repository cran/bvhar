## ----rmdsetup, include = FALSE------------------------------------------------
knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  out.width = "70%",
  fig.align = "center",
  fig.width = 6,
  fig.asp = .618
)
orig_opts <- options("digits")
options(digits = 3)
set.seed(1)

## ----setup--------------------------------------------------------------------
library(bvhar)

## ----etfdat-------------------------------------------------------------------
etf <- etf_vix[1:100, 1:3]
# Split-------------------------------
h <- 5
etf_eval <- divide_ts(etf, h)
etf_train <- etf_eval$train
etf_test <- etf_eval$test

## ----fitssvs------------------------------------------------------------------
(fit_ssvs <- bvhar_ssvs(etf_train, num_chains = 1, num_iter = 20, include_mean = FALSE, minnesota = "longrun"))

## ----heatssvs-----------------------------------------------------------------
autoplot(fit_ssvs)

## ----fiths--------------------------------------------------------------------
(fit_hs <- bvhar_horseshoe(etf_train, num_chains = 2, num_iter = 20, include_mean = FALSE, minnesota = "longrun"))

## ----heaths-------------------------------------------------------------------
autoplot(fit_hs)

## ----svssvs-------------------------------------------------------------------
(fit_ssvs_sv <- bvhar_sv(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_ssvs(), sv_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))

## ----heatssvssv---------------------------------------------------------------
autoplot(fit_ssvs_sv)

## -----------------------------------------------------------------------------
(fit_hs_sv <- bvhar_sv(etf_train, num_chains = 2, num_iter = 20, bayes_spec = set_horseshoe(), sv_spec = set_sv(), include_mean = FALSE, minnesota = "longrun"))

## -----------------------------------------------------------------------------
autoplot(fit_hs_sv, type = "trace", regex_pars = "tau")

## ----denshs-------------------------------------------------------------------
autoplot(fit_hs_sv, type = "dens", regex_pars = "kappa", facet_args = list(dir = "v", nrow = nrow(fit_hs_sv$coefficients)))

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

