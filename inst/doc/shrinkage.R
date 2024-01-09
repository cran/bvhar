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
var_idx <- c("GVZCLS", "OVXCLS", "EVZCLS", "VXFXICLS")
etf <-
  etf_vix[1:100, ] %>%
  dplyr::select(dplyr::all_of(var_idx))
# Split-------------------------------
h <- 5
etf_eval <- divide_ts(etf, h)
etf_train <- etf_eval$train
etf_test <- etf_eval$test

## ----fitssvs------------------------------------------------------------------
(fit_ssvs <- bvhar_ssvs(etf_train, num_iter = 50, include_mean = FALSE, minnesota = "longrun"))

## ----heatssvs-----------------------------------------------------------------
autoplot(fit_ssvs)

## -----------------------------------------------------------------------------
autoplot(fit_ssvs, type = "trace", regex_pars = "psi")

## ----fiths--------------------------------------------------------------------
(fit_hs <- bvhar_horseshoe(etf_train, num_iter = 50, include_mean = FALSE, minnesota = "longrun", verbose = TRUE))

## ----heaths-------------------------------------------------------------------
autoplot(fit_hs)

## ----denshs-------------------------------------------------------------------
autoplot(fit_hs, type = "dens", regex_pars = "tau")

## ----svssvs-------------------------------------------------------------------
(fit_ssvs_sv <- bvhar_sv(etf_train, num_iter = 50, bayes_spec = set_ssvs(), include_mean = FALSE, minnesota = "longrun"))

## ----heatssvssv---------------------------------------------------------------
autoplot(fit_ssvs_sv)

## -----------------------------------------------------------------------------
(fit_hs_sv <- bvhar_sv(etf_train, num_iter = 50, bayes_spec = set_horseshoe(), include_mean = FALSE, minnesota = "longrun"))

## ----heathssv-----------------------------------------------------------------
autoplot(fit_hs_sv)

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

