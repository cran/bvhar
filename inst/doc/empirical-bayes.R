## ----rmdsetup, include = FALSE------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
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

## ----datasub------------------------------------------------------------------
etf_split <- 
  etf_vix %>% 
  dplyr::select(GVZCLS, OVXCLS, VXEEMCLS, VXGDXCLS, VXSLVCLS) %>% 
  divide_ts(20)
# split---------------------
etf_train <- etf_split$train
etf_test <- etf_split$test
dim_data <- ncol(etf_train)

## ----bvarspec-----------------------------------------------------------------
sig <- apply(etf_train, 2, sd)
lam <- .2
del <- rep(1, dim_data)
# bvharspec------------------
(bvar_spec <- set_bvar(
  sigma = sig,
  lambda = lam,
  delta = del,
  eps = 1e-04
))

## ----bvarinit-----------------------------------------------------------------
bvar_cand <- bvar_minnesota(
  y = etf_train, 
  p = 3, 
  bayes_spec = bvar_spec, 
  include_mean = TRUE
)

## ----bvharsspec---------------------------------------------------------------
(bvhar_short_spec <- set_bvhar(
  sigma = sig,
  lambda = lam,
  delta = del,
  eps = 1e-04
))

## ----bvharsinit---------------------------------------------------------------
bvhar_short_cand <- bvhar_minnesota(
  y = etf_train, 
  har = c(5, 22), 
  bayes_spec = bvhar_short_spec, 
  include_mean = TRUE
)

## ----bvharlspec---------------------------------------------------------------
dayj <- rep(.8, dim_data)
weekj <- rep(.2, dim_data)
monthj <- rep(.1, dim_data)
# bvharspec------------------
bvhar_long_spec <- set_weight_bvhar(
  sigma = sig,
  lambda = lam,
  eps = 1e-04,
  daily = dayj,
  weekly = weekj,
  monthly = monthj
)

## ----bvharlinit---------------------------------------------------------------
bvhar_long_cand <- bvhar_minnesota(
  y = etf_train, 
  har = c(5, 22), 
  bayes_spec = bvhar_long_spec, 
  include_mean = TRUE
)

## ----parallelcl---------------------------------------------------------------
cl <- parallel::makeCluster(2)

## ----bvaroptim----------------------------------------------------------------
(bvar_optim <- choose_bvar(
  bayes_spec = bvar_spec,
  lower = c(
    rep(1, dim_data), # sigma
    1e-2, # lambda
    rep(1e-2, dim_data) # delta
  ),
  upper = c(
    rep(15, dim_data), # sigma
    Inf, # lambda
    rep(1, dim_data) # delta
  ),
  eps = 1e-04,
  y = etf_train,
  p = 3,
  include_mean = TRUE,
  parallel = list(cl = cl, forward = FALSE, loginfo = FALSE)
))

## ----optimlist----------------------------------------------------------------
class(bvar_optim)
names(bvar_optim)

## ----bvarchoose---------------------------------------------------------------
bvar_final <- bvar_optim$fit

## ----bvharsoptim--------------------------------------------------------------
(bvhar_short_optim <- choose_bvhar(
  bayes_spec = bvhar_short_spec,
  lower = c(
    rep(1, dim_data), # sigma
    1e-2, # lambda
    rep(1e-2, dim_data) # delta
  ),
  upper = c(
    rep(15, dim_data), # sigma
    Inf, # lambda
    rep(1, dim_data) # delta
  ),
  eps = 1e-04,
  y = etf_train,
  har = c(5, 22),
  include_mean = TRUE,
  parallel = list(cl = cl, forward = FALSE, loginfo = FALSE)
))

## ----optimbvharslist----------------------------------------------------------
class(bvhar_short_optim)
names(bvhar_short_optim)

## ----bvharschoose-------------------------------------------------------------
bvhar_short_final <- bvhar_short_optim$fit

## ----bvharloptim--------------------------------------------------------------
(bvhar_long_optim <- choose_bvhar(
  bayes_spec = bvhar_long_spec,
  lower = c(
    rep(1, dim_data), # sigma
    1e-2, # lambda
    rep(1e-2, dim_data), # daily
    rep(1e-2, dim_data), # weekly
    rep(1e-2, dim_data) # monthly
  ),
  upper = c(
    rep(15, dim_data), # sigma
    Inf, # lambda
    rep(1, dim_data), # daily
    rep(1, dim_data), # weekly
    rep(1, dim_data) # monthly
  ),
  eps = 1e-04,
  y = etf_train,
  har = c(5, 22),
  include_mean = TRUE,
  parallel = list(cl = cl, forward = FALSE, loginfo = FALSE)
))

## ----optimbvharllist----------------------------------------------------------
class(bvhar_long_optim)
names(bvhar_long_optim)

## ----bvharlchoose-------------------------------------------------------------
bvhar_long_final <- bvhar_long_optim$fit

## ----boundemp-----------------------------------------------------------------
# lower bound----------------
bvar_lower <- set_bvar(
  sigma = rep(1, dim_data),
  lambda = 1e-2,
  delta = rep(1e-2, dim_data)
)
# upper bound---------------
bvar_upper <- set_bvar(
  sigma = rep(15, dim_data),
  lambda = Inf,
  delta = rep(1, dim_data)
)
# bound--------------------
(bvar_bound <- bound_bvhar(
  init_spec = bvar_spec,
  lower_spec = bvar_lower,
  upper_spec = bvar_upper
))

## ----boundemplist-------------------------------------------------------------
class(bvar_bound)
names(bvar_bound)

## ----bvaroptimother-----------------------------------------------------------
(bvar_optim_v2 <- choose_bayes(
  bayes_bound = bvar_bound,
  eps = 1e-04,
  y = etf_train,
  order = 3,
  include_mean = TRUE,
  parallel = list(cl = cl, forward = FALSE, loginfo = FALSE)
))

## ----stopparallel-------------------------------------------------------------
parallel::stopCluster(cl)

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

