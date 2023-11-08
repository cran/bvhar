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

## ----evalcoef, echo=FALSE-----------------------------------------------------
etf_eval <- 
  etf_vix %>% 
  dplyr::select(GVZCLS, OVXCLS, EVZCLS, VXFXICLS) %>% 
  divide_ts(20)
etf_train <- etf_eval$train
etf_test <- etf_eval$test
ex_fit <- var_lm(etf_train, p = 5)

## ----whatcoef-----------------------------------------------------------------
coef(ex_fit)
ex_fit$covmat

## ----simvar-------------------------------------------------------------------
m <- ncol(ex_fit$coefficients)
# generate VAR(5)-----------------
y <- sim_var(
  num_sim = 1500, 
  num_burn = 100, 
  var_coef = coef(ex_fit), 
  var_lag = 5L, 
  sig_error = ex_fit$covmat, 
  init = matrix(0L, nrow = 5L, ncol = m)
)
# colname: y1, y2, ...------------
colnames(y) <- paste0("y", 1:m)
head(y)

## ----outofsample--------------------------------------------------------------
h <- 20
y_eval <- divide_ts(y, h)
y_train <- y_eval$train # train
y_test <- y_eval$test # test

## ----fitvar-------------------------------------------------------------------
# VAR(5)
model_var <- var_lm(y_train, 5)
# VHAR
model_vhar <- vhar_lm(y_train)

## ----fitbvar------------------------------------------------------------------
# hyper parameters---------------------------
y_sig <- apply(y_train, 2, sd) # sigma vector
y_lam <- .2 # lambda
y_delta <- rep(.2, m) # delta vector (0 vector since RV stationary)
eps <- 1e-04 # very small number
spec_bvar <- set_bvar(y_sig, y_lam, y_delta, eps)
# fit---------------------------------------
model_bvar <- bvar_minnesota(y_train, 5, spec_bvar)

## ----fitbvhars----------------------------------------------------------------
spec_bvhar_v1 <- set_bvhar(y_sig, y_lam, y_delta, eps)
# fit---------------------------------------
model_bvhar_v1 <- bvhar_minnesota(y_train, bayes_spec = spec_bvhar_v1)

## ----fitbvharl----------------------------------------------------------------
# weights----------------------------------
y_day <- rep(.1, m)
y_week <- rep(.01, m)
y_month <- rep(.01, m)
# spec-------------------------------------
spec_bvhar_v2 <- set_weight_bvhar(
  y_sig,
  y_lam,
  eps,
  y_day,
  y_week,
  y_month
)
# fit--------------------------------------
model_bvhar_v2 <- bvhar_minnesota(y_train, bayes_spec = spec_bvhar_v2)

## ----predvar------------------------------------------------------------------
(pred_var <- predict(model_var, n_ahead = h))

## ----varpredlist--------------------------------------------------------------
class(pred_var)
names(pred_var)

## ----msevar-------------------------------------------------------------------
(mse_var <- mse(pred_var, y_test))

## ----predvhar-----------------------------------------------------------------
(pred_vhar <- predict(model_vhar, n_ahead = h))

## ----msevhar------------------------------------------------------------------
(mse_vhar <- mse(pred_vhar, y_test))

## ----predbvar-----------------------------------------------------------------
(pred_bvar <- predict(model_bvar, n_ahead = h))

## ----msebvar------------------------------------------------------------------
(mse_bvar <- mse(pred_bvar, y_test))

## ----predbvharvar-------------------------------------------------------------
(pred_bvhar_v1 <- predict(model_bvhar_v1, n_ahead = h))

## ----msebvharvar--------------------------------------------------------------
(mse_bvhar_v1 <- mse(pred_bvhar_v1, y_test))

## ----predbvharvhar------------------------------------------------------------
(pred_bvhar_v2 <- predict(model_bvhar_v2, n_ahead = h))

## ----msebvharvhar-------------------------------------------------------------
(mse_bvhar_v2 <- mse(pred_bvhar_v2, y_test))

## ----predplot-----------------------------------------------------------------
autoplot(pred_var, x_cut = 1470, ci_alpha = .7, type = "wrap") +
  autolayer(pred_vhar, ci_alpha = .5) +
  autolayer(pred_bvar, ci_alpha = .4) +
  autolayer(pred_bvhar_v1, ci_alpha = .2) +
  autolayer(pred_bvhar_v2, ci_alpha = .1) +
  geom_eval(y_test, colour = "#000000", alpha = .5)

## ----msevalues----------------------------------------------------------------
list(
  VAR = mse_var,
  VHAR = mse_vhar,
  BVAR = mse_bvar,
  BVHAR1 = mse_bvhar_v1,
  BVHAR2 = mse_bvhar_v2
) %>% 
  lapply(mean) %>% 
  unlist() %>% 
  sort()

## ----evalplot-----------------------------------------------------------------
list(
  pred_var,
  pred_vhar,
  pred_bvar,
  pred_bvhar_v1,
  pred_bvhar_v2
) %>% 
  gg_loss(y = y_test, "mse")

## ----relmape------------------------------------------------------------------
list(
  VAR = pred_var,
  VHAR = pred_vhar,
  BVAR = pred_bvar,
  BVHAR1 = pred_bvhar_v1,
  BVHAR2 = pred_bvhar_v2
) %>% 
  lapply(rmape, pred_bench = pred_var, y = y_test) %>% 
  unlist()

## ----rollvar------------------------------------------------------------------
(var_roll <- forecast_roll(model_var, 5, y_test))

## ----rollvarlist--------------------------------------------------------------
class(var_roll)
names(var_roll)

## ----otherroll----------------------------------------------------------------
vhar_roll <- forecast_roll(model_vhar, 5, y_test)
bvar_roll <- forecast_roll(model_bvar, 5, y_test)
bvhar_roll_v1 <- forecast_roll(model_bvhar_v1, 5, y_test)
bvhar_roll_v2 <- forecast_roll(model_bvhar_v2, 5, y_test)

## ----relroll------------------------------------------------------------------
list(
  VAR = var_roll,
  VHAR = vhar_roll,
  BVAR = bvar_roll,
  BVHAR1 = bvhar_roll_v1,
  BVHAR2 = bvhar_roll_v2
) %>% 
  lapply(rmape, pred_bench = var_roll, y = y_test) %>% 
  unlist()

## ----expandvar----------------------------------------------------------------
(var_expand <- forecast_expand(model_var, 5, y_test))

## ----expandvarlist------------------------------------------------------------
class(var_expand)
names(var_expand)

## ----otherexpand--------------------------------------------------------------
vhar_expand <- forecast_expand(model_vhar, 5, y_test)
bvar_expand <- forecast_expand(model_bvar, 5, y_test)
bvhar_expand_v1 <- forecast_expand(model_bvhar_v1, 5, y_test)
bvhar_expand_v2 <- forecast_expand(model_bvhar_v2, 5, y_test)

## ----relexpand----------------------------------------------------------------
list(
  VAR = var_expand,
  VHAR = vhar_expand,
  BVAR = bvar_expand,
  BVHAR1 = bvhar_expand_v1,
  BVHAR2 = bvhar_expand_v2
) %>% 
  lapply(rmape, pred_bench = var_expand, y = y_test) %>% 
  unlist()

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

