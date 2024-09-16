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

## ----gennormal----------------------------------------------------------------
sim_mnormal(3, rep(0, 2), diag(2))

## ----genmatnorm---------------------------------------------------------------
sim_matgaussian(matrix(1:20, nrow = 4), diag(4), diag(5), FALSE)

## ----geniw--------------------------------------------------------------------
sim_iw(diag(5), 7)

## ----genmniw------------------------------------------------------------------
sim_mniw(2, matrix(1:20, nrow = 4), diag(4), diag(5), 7, FALSE)

## ----minnesotaset-------------------------------------------------------------
bvar_lag <- 5
(spec_to_sim <- set_bvar(
  sigma = c(3.25, 11.1, 2.2, 6.8), # sigma vector
  lambda = .2, # lambda
  delta = rep(1, 4), # 4-dim delta vector
  eps = 1e-04 # very small number
))

## ----simminnesota-------------------------------------------------------------
(sim_mncoef(bvar_lag, spec_to_sim))

## ----bvharvarset--------------------------------------------------------------
(bvhar_var_spec <- set_bvhar(
  sigma = c(1.2, 2.3), # sigma vector
  lambda = .2, # lambda
  delta = c(.3, 1), # 2-dim delta vector
  eps = 1e-04 # very small number
))

## ----simbvhars----------------------------------------------------------------
(sim_mnvhar_coef(bvhar_var_spec))

## ----bvharvharset-------------------------------------------------------------
(bvhar_vhar_spec <- set_weight_bvhar(
  sigma = c(1.2, 2.3), # sigma vector
  lambda = .2, # lambda
  eps = 1e-04, # very small number
  daily = c(.5, 1), # 2-dim daily weight vector
  weekly = c(.2, .3), # 2-dim weekly weight vector
  monthly = c(.1, .1) # 2-dim monthly weight vector
))

## ----simbvharl----------------------------------------------------------------
(sim_mnvhar_coef(bvhar_vhar_spec))

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

