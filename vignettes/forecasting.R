## ----rmdsetup, include = FALSE------------------------------------------------
knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  out.width = "70%",
  fig.align = "center",
  fig.width = 6,
  fig.asp = .618
  )
options(digits = 3)
set.seed(1)

## ----setup--------------------------------------------------------------------
library(bvhar)

## ---- echo=FALSE--------------------------------------------------------------
etf_eval <- 
  etf_vix %>% 
  dplyr::select(GVZCLS, OVXCLS, EVZCLS, VXFXICLS) %>% 
  divide_ts(19)
etf_train <- etf_eval$train
etf_test <- etf_eval$test
ex_fit <- var_lm(etf_train, 5)

## -----------------------------------------------------------------------------
coef(ex_fit)
ex_fit$covmat

## -----------------------------------------------------------------------------
m <- ncol(ex_fit$coefficients)
# generate VAR(5)-----------------
y <- sim_var(
  1500, 
  100, 
  coef(ex_fit), 
  5, 
  diag(ex_fit$covmat) %>% diag(), 
  matrix(0L, nrow = 5, ncol = m)
)
# colname: y1, y2, ...------------
colnames(y) <- paste0("y", 1:m)
head(y)

## ----outofsample--------------------------------------------------------------
h <- 20
y_eval <- divide_ts(y, h)
y_train <- y_eval$train # train
y_test <- y_eval$test # test

## -----------------------------------------------------------------------------
# VAR(5)
model_var <- var_lm(y_train, 5)
# VHAR
model_vhar <- vhar_lm(y_train)

## -----------------------------------------------------------------------------
# hyper parameters---------------------------
y_sig <- apply(y_train, 2, sd) # sigma vector
y_lam <- .2 # lambda
y_delta <- rep(.2, m) # delta vector (0 vector since RV stationary)
eps <- 1e-04 # very small number
spec_bvar <- set_bvar(y_sig, y_lam, y_delta, eps)
# fit---------------------------------------
model_bvar <- bvar_minnesota(y_train, 5, spec_bvar)

## -----------------------------------------------------------------------------
spec_bvhar_v1 <- set_bvhar(y_sig, y_lam, y_delta, eps)
# fit---------------------------------------
model_bvhar_v1 <- bvhar_minnesota(y_train, spec_bvhar_v1)

## -----------------------------------------------------------------------------
# weights----------------------------------
y_day <- rep(.2, m)
y_week <- rep(.1, m)
y_month <- rep(.1, m)
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
model_bvhar_v2 <- bvhar_minnesota(y_train, spec_bvhar_v2)

## -----------------------------------------------------------------------------
(pred_var <- predict(model_var, n_ahead = h))

## -----------------------------------------------------------------------------
class(pred_var)
names(pred_var)

## -----------------------------------------------------------------------------
(mse_var <- mse(pred_var, y_test))

## -----------------------------------------------------------------------------
(pred_vhar <- predict(model_vhar, n_ahead = h))

## -----------------------------------------------------------------------------
(mse_vhar <- mse(pred_vhar, y_test))

## -----------------------------------------------------------------------------
(pred_bvar <- predict(model_bvar, n_ahead = h))

## -----------------------------------------------------------------------------
(mse_bvar <- mse(pred_bvar, y_test))

## -----------------------------------------------------------------------------
(pred_bvhar_v1 <- predict(model_bvhar_v1, n_ahead = h))

## -----------------------------------------------------------------------------
(mse_bvhar_v1 <- mse(pred_bvhar_v1, y_test))

## -----------------------------------------------------------------------------
(pred_bvhar_v2 <- predict(model_bvhar_v2, n_ahead = h))

## -----------------------------------------------------------------------------
(mse_bvhar_v2 <- mse(pred_bvhar_v2, y_test))

## -----------------------------------------------------------------------------
autoplot(pred_var, x_cut = 1450, ci_alpha = .7) +
  autolayer(pred_vhar, ci_alpha = .5) +
  autolayer(pred_bvar, ci_alpha = .4) +
  autolayer(pred_bvhar_v1, ci_alpha = .2) +
  autolayer(pred_bvhar_v2, ci_alpha = .1)

## -----------------------------------------------------------------------------
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

## -----------------------------------------------------------------------------
list(
  pred_var,
  pred_vhar,
  pred_bvar,
  pred_bvhar_v1,
  pred_bvhar_v2
) %>% 
  gg_loss(y = y_test, "mse")

