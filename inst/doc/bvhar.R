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

## ----setup--------------------------------------------------------------------
library(bvhar)

## ----etfdat-------------------------------------------------------------------
var_idx <- c("GVZCLS", "OVXCLS", "EVZCLS", "VXFXICLS")
etf <- 
  etf_vix %>% 
  dplyr::select(dplyr::all_of(var_idx))
etf

## ----hstepsplit---------------------------------------------------------------
h <- 19
etf_eval <- divide_ts(etf, h) # Try ?divide_ts
etf_train <- etf_eval$train # train
etf_test <- etf_eval$test # test
# dimension---------
m <- ncol(etf)

## ----varlag-------------------------------------------------------------------
var_lag <- 5

## ----varfit-------------------------------------------------------------------
(fit_var <- var_lm(etf_train, var_lag))

## ----varlist------------------------------------------------------------------
# class---------------
class(fit_var)
# inheritance---------
is.varlse(fit_var)
# names---------------
names(fit_var)

## ----harfit-------------------------------------------------------------------
(fit_har <- vhar_lm(etf_train))

## ----harlist------------------------------------------------------------------
# class----------------
class(fit_har)
# inheritance----------
is.varlse(fit_har)
is.vharlse(fit_har)
# complements----------
names(fit_har)

## ----minnesotaset-------------------------------------------------------------
bvar_lag <- 5
sig <- apply(etf_train, 2, sd) # sigma vector
lam <- .2 # lambda
delta <- rep(0, m) # delta vector (0 vector since RV stationary)
eps <- 1e-04 # very small number
(bvar_spec <- set_bvar(sig, lam, delta, eps))

## ----bvarfit------------------------------------------------------------------
(fit_bvar <- bvar_minnesota(etf_train, bvar_lag, num_iter = 10, bayes_spec = bvar_spec))

## ----bvarlist-----------------------------------------------------------------
# class---------------
class(fit_bvar)
# inheritance---------
is.bvarmn(fit_bvar)
# names---------------
names(fit_bvar)

## ----flatspec-----------------------------------------------------------------
(flat_spec <- set_bvar_flat(U = 5000 * diag(m * bvar_lag + 1))) # c * I

## ----flatfit------------------------------------------------------------------
(fit_ghosh <- bvar_flat(etf_train, bvar_lag, num_iter = 10, bayes_spec = flat_spec))

## ----flatlist-----------------------------------------------------------------
# class---------------
class(fit_ghosh)
# inheritance---------
is.bvarflat(fit_ghosh)
# names---------------
names(fit_ghosh)

## ----bvharvarspec-------------------------------------------------------------
(bvhar_spec_v1 <- set_bvhar(sig, lam, delta, eps))

## -----------------------------------------------------------------------------
(fit_bvhar_v1 <- bvhar_minnesota(etf_train, num_iter = 10, bayes_spec = bvhar_spec_v1))

## ----bvharlist----------------------------------------------------------------
# class---------------
class(fit_bvhar_v1)
# inheritance---------
is.bvharmn(fit_bvhar_v1)
# names---------------
names(fit_bvhar_v1)

## -----------------------------------------------------------------------------
daily <- rep(.1, m)
weekly <- rep(.1, m)
monthly <- rep(.1, m)
(bvhar_spec_v2 <- set_weight_bvhar(sig, lam, eps, daily, weekly, monthly))

## -----------------------------------------------------------------------------
fit_bvhar_v2 <- bvhar_minnesota(
  etf_train,
  num_iter = 10,
  bayes_spec = bvhar_spec_v2
)
fit_bvhar_v2

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

