#' Hyperparameters for Bayesian Models
#' 
#' Set hyperparameters of Bayesian VAR and VHAR models.
#' 
#' @param sigma Standard error vector for each variable (Default: sd of each variable)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Litterman sets 1 = random walk prior (default: rep(1, number of variables)), White noise prior = 0)
#' @param eps Very small number (Default: 1e-04)
#' @details 
#' * Missing arguments will be set to be default values in each model function mentioned above.
#' * `set_bvar()` sets hyperparameters for [bvar_minnesota()].
#' * Each `delta` (vector), `lambda` (length of 1), `sigma` (vector), `eps` (vector) corresponds to \eqn{\delta_j}, \eqn{\lambda}, \eqn{\delta_j}, \eqn{\epsilon}.
#' 
#' \eqn{\delta_i} are related to the belief to random walk.
#' 
#' * If \eqn{\delta_i = 1} for all i, random walk prior
#' * If \eqn{\delta_i = 0} for all i, white noise prior
#' 
#' \eqn{\lambda} controls the overall tightness of the prior around these two prior beliefs.
#' 
#' * If \eqn{\lambda = 0}, the posterior is equivalent to prior and the data do not influence the estimates.
#' * If \eqn{\lambda = \infty}, the posterior mean becomes OLS estimates (VAR).
#' 
#' \eqn{\sigma_i^2 / \sigma_j^2} in Minnesota moments explain the data scales.
#' @return Every function returns `bvharspec` [class].
#' It is the list of which the components are the same as the arguments provided.
#' If the argument is not specified, `NULL` is assigned here.
#' The default values mentioned above will be considered in each fitting function.
#' \describe{
#'   \item{process}{Model name: `BVAR`, `BVHAR`}
#'   \item{prior}{
#'   Prior name: `Minnesota` (Minnesota prior for BVAR),
#'   `Hierarchical` (Hierarchical prior for BVAR),
#'   `MN_VAR` (BVHAR-S),
#'   `MN_VHAR` (BVHAR-L),
#'   `Flat` (Flat prior for BVAR)
#'   }
#'   \item{sigma}{Vector value (or `bvharpriorspec` class) assigned for sigma}
#'   \item{lambda}{Value (or `bvharpriorspec` class) assigned for lambda}
#'   \item{delta}{Vector value assigned for delta}
#'   \item{eps}{Value assigned for epsilon}
#' }
#' @note 
#' By using [set_psi()] and [set_lambda()] each, hierarchical modeling is available.
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' @examples 
#' # Minnesota BVAR specification------------------------
#' bvar_spec <- set_bvar(
#'   sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'   lambda = .2, # lambda = .2
#'   delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
#'   eps = 1e-04 # eps = 1e-04
#' )
#' class(bvar_spec)
#' str(bvar_spec)
#' @seealso 
#' * lambda hyperprior specification [set_lambda()]
#' * sigma hyperprior specification [set_psi()]
#' @order 1
#' @export
set_bvar <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  if (missing(sigma)) {
    sigma <- NULL
  }
  if (missing(delta)) {
    delta <- NULL
  }
  hierarchical <- is.bvharpriorspec(lambda)
  if (hierarchical) {
    # if (!all(is.bvharpriorspec(sigma) & is.bvharpriorspec(lambda))) {
    #   stop("When using hierarchical model, each 'sigma' and 'lambda' should be 'bvharpriorspec'.")
    # }
    # prior_type <- "MN_Hierarchical"
    if (all(is.bvharpriorspec(sigma) | is.bvharpriorspec(lambda))) {
      prior_type <- "MN_Hierarchical"
    } else if (is.bvharpriorspec(lambda)) {
      prior_type <- "Minnesota"
    } else {
      stop("Invalid hierarchical setting.")
    }
  } else {
    if (lambda <= 0) {
      stop("'lambda' should be larger than 0.")
    }
    if (length(sigma) > 0 && any(sigma <= 0)) {
      stop("'sigma' should be larger than 0.")
    }
    if (length(delta) > 0 && any(delta < 0)) {
      stop("'delta' should not be smaller than 0.")
    }
    if (length(sigma) > 0 && length(delta) > 0) {
      if (length(sigma) != length(delta)) {
        stop("Length of 'sigma' and 'delta' must be the same as the dimension of the time series.")
      }
    }
    prior_type <- "Minnesota"
  }
  bvar_param <- list(
    process = "BVAR",
    prior = prior_type,
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps,
    hierarchical = hierarchical
  )
  class(bvar_param) <- "bvharspec"
  bvar_param
}

#' Hyperpriors for Bayesian Models
#'
#' Set hyperpriors of Bayesian VAR and VHAR models.
#' 
#' @param mode Mode of Gamma distribution. By default, `.2`.
#' @param sd Standard deviation of Gamma distribution. By default, `.4`.
#' @param param Shape and rate of Gamma distribution, in the form of `c(shape, rate)`. If specified, ignore `mode` and `sd`.
#' @param lower `r lifecycle::badge("experimental")` Lower bound for [stats::optim()]. By default, `1e-5`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound for [stats::optim()]. By default, `3`.
#' @param grid_size Griddy gibbs grid size for lag scaling
#' @details
#' In addition to Normal-IW priors [set_bvar()], [set_bvhar()], and [set_weight_bvhar()],
#' these functions give hierarchical structure to the model.
#' * `set_lambda()` specifies hyperprior for \eqn{\lambda} (`lambda`), which is Gamma distribution.
#' * `set_psi()` specifies hyperprior for \eqn{\psi / (\nu_0 - k - 1) = \sigma^2} (`sigma`), which is Inverse gamma distribution.
#' @examples
#' # Hirearchical BVAR specification------------------------
#' set_bvar(
#'   sigma = set_psi(shape = 4e-4, scale = 4e-4),
#'   lambda = set_lambda(mode = .2, sd = .4),
#'   delta = rep(1, 3),
#'   eps = 1e-04 # eps = 1e-04
#' )
#' @return `bvharpriorspec` object
#' @references Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2).
#' @order 1
#' @export
set_lambda <- function(mode = .2, sd = .4, param = NULL, lower = 1e-5, upper = 3, grid_size = 100L) {
  if (length(grid_size) != 1) {
    stop("'grid_size' should be length 1 numeric.")
  }
  if (grid_size %% 1 != 0) {
    stop("Provide integer for 'grid_size'.")
  }
  if (is.null(param)) {
    params <- get_gammaparam(mode, sd)
    # param <- c(params$shape, params$rate)
    lam_prior <- list(
      hyperparam = "lambda",
      param = c(params$shape, params$rate),
      grid_size = grid_size,
      mode = mode,
      lower = lower,
      upper = upper
    )
  } else {
    mode <- ifelse(param[1] >= 1, (param[1] - 1) / param[2], 0)
    lam_prior <- list(
      hyperparam = "lambda",
      param = param,
      grid_size = grid_size
    )
  }
  # lam_prior <- list(
  #   hyperparam = "lambda",
  #   param = param,
  #   mode = mode,
  #   lower = lower,
  #   upper = upper
  # )
  class(lam_prior) <- "bvharpriorspec"
  lam_prior
}

#' @rdname set_lambda
#' @param shape Shape of Inverse Gamma distribution. By default, `(.02)^2`.
#' @param scale Scale of Inverse Gamma distribution. By default, `(.02)^2`.
#' @param lower `r lifecycle::badge("experimental")` Lower bound for [stats::optim()]. By default, `1e-5`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound for [stats::optim()]. By default, `3`.
#' @details
#' The following set of `(mode, sd)` are recommended by Sims and Zha (1998) for `set_lambda()`.
#' * `(mode = .2, sd = .4)`: default
#' * `(mode = 1, sd = 1)`
#'
#' Giannone et al. (2015) suggested data-based selection for `set_psi()`.
#' It chooses (0.02)^2 based on its empirical data set.
#' @order 1
#' @export
set_psi <- function(shape = 4e-4, scale = 4e-4, lower = 1e-5, upper = 3) {
  psi_prior <- list(
    hyperparam = "psi",
    param = c(shape, scale),
    mode = scale / (shape + 1),
    lower = lower,
    upper = upper
  )
  class(psi_prior) <- "bvharpriorspec"
  psi_prior
}

#' @rdname set_bvar
#' @param U Positive definite matrix. By default, identity matrix of dimension ncol(X0)
#' @details 
#' * `set_bvar_flat` sets hyperparameters for [bvar_flat()].
#' @examples 
#' # Flat BVAR specification-------------------------
#' # 3-dim
#' # p = 5 with constant term
#' # U = 500 * I(mp + 1)
#' bvar_flat_spec <- set_bvar_flat(U = 500 * diag(16))
#' class(bvar_flat_spec)
#' str(bvar_flat_spec)
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @order 1
#' @export
set_bvar_flat <- function(U) {
  if (missing(U)) {
    U <- NULL
  }
  bvar_param <- list(
    process = "BVAR",
    prior = "Flat",
    U = U
  )
  class(bvar_param) <- "bvharspec"
  bvar_param
}

#' @rdname set_bvar
#' @param sigma Standard error vector for each variable (Default: sd)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Default: Litterman sets 1 = random walk prior, White noise prior = 0)
#' @param eps Very small number (Default: 1e-04)
#' @details 
#' * `set_bvhar()` sets hyperparameters for [bvhar_minnesota()] with VAR-type Minnesota prior, i.e. BVHAR-S model.
#' @examples 
#' # BVHAR-S specification-----------------------
#' bvhar_var_spec <- set_bvhar(
#'   sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'   lambda = .2, # lambda = .2
#'   delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
#'   eps = 1e-04 # eps = 1e-04
#' )
#' class(bvhar_var_spec)
#' str(bvhar_var_spec)
#' @references Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
#' @order 1
#' @export
set_bvhar <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  if (missing(sigma)) {
    sigma <- NULL
  }
  if (missing(delta)) {
    delta <- NULL
  }
  hierarchical <- is.bvharpriorspec(lambda)
  if (hierarchical) {
    if (all(is.bvharpriorspec(sigma) | is.bvharpriorspec(lambda))) {
      prior_type <- "MN_Hierarchical"
    } else if (is.bvharpriorspec(lambda)) {
      prior_type <- "MN_VAR"
    } else {
      stop("Invalid hierarchical setting.")
    }
  } else {
    if (length(sigma) > 0 && length(delta) > 0) {
      if (length(sigma) != length(delta)) {
        stop("Length of 'sigma' and 'delta' must be the same as the dimension of the time series.")
      }
    }
    prior_type <- "MN_VAR"
  }
  bvhar_param <- list(
    process = "BVHAR",
    prior = prior_type,
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps,
    hierarchical = hierarchical
  )
  class(bvhar_param) <- "bvharspec"
  bvhar_param
}

#' @rdname set_bvar
#' @param sigma Standard error vector for each variable (Default: sd)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param eps Very small number (Default: 1e-04)
#' @param daily Same as delta in VHAR type (Default: 1 as Litterman)
#' @param weekly Fill the second part in the first block (Default: 1)
#' @param monthly Fill the third part in the first block (Default: 1)
#' @details 
#' * `set_weight_bvhar()` sets hyperparameters for [bvhar_minnesota()] with VHAR-type Minnesota prior, i.e. BVHAR-L model.
#' @return `set_weight_bvhar()` has different component with `delta` due to its different construction.
#' \describe{
#'   \item{daily}{Vector value assigned for daily weight}
#'   \item{weekly}{Vector value assigned for weekly weight}
#'   \item{monthly}{Vector value assigned for monthly weight}
#' }
#' @references Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
#' @examples 
#' # BVHAR-L specification---------------------------
#' bvhar_vhar_spec <- set_weight_bvhar(
#'   sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'   lambda = .2, # lambda = .2
#'   eps = 1e-04, # eps = 1e-04
#'   daily = rep(.2, 3), # daily1 = .2, daily2 = .2, daily3 = .2
#'   weekly = rep(.1, 3), # weekly1 = .1, weekly2 = .1, weekly3 = .1
#'   monthly = rep(.05, 3) # monthly1 = .05, monthly2 = .05, monthly3 = .05
#' )
#' class(bvhar_vhar_spec)
#' str(bvhar_vhar_spec)
#' @order 1
#' @export
set_weight_bvhar <- function(sigma,
                             lambda = .1,
                             eps = 1e-04,
                             daily,
                             weekly,
                             monthly) {
  if (missing(sigma)) {
    sigma <- NULL
  }
  if (missing(daily)) {
    daily <- NULL
  }
  if (missing(weekly)) {
    weekly <- NULL
  }
  if (missing(monthly)) {
    monthly <- NULL
  }
  hierarchical <- is.bvharpriorspec(lambda)
  if (hierarchical) {
    if (all(is.bvharpriorspec(sigma) | is.bvharpriorspec(lambda))) {
      prior_type <- "MN_Hierarchical"
    } else if (is.bvharpriorspec(lambda)) {
      prior_type <- "MN_VHAR"
    } else {
      stop("Invalid hierarchical setting.")
    }
  } else {
    if (length(sigma) > 0) {
      if (length(daily) > 0) {
        if (length(sigma) != length(daily)) {
          stop("Length of 'sigma' and 'daily' must be the same as the dimension of the time series.")
        }
      }
      if (length(weekly) > 0) {
        if (length(sigma) != length(weekly)) {
          stop("Length of 'sigma' and 'weekly' must be the same as the dimension of the time series.")
        }
      }
      if (length(monthly) > 0) {
        if (length(sigma) != length(monthly)) {
          stop("Length of 'sigma' and 'monthly' must be the same as the dimension of the time series.")
        }
      }
    }
    prior_type <- "MN_VHAR"
  }
  bvhar_param <- list(
    process = "BVHAR",
    prior = prior_type,
    sigma = sigma,
    lambda = lambda,
    eps = eps,
    daily = daily,
    weekly = weekly,
    monthly = monthly,
    hierarchical = hierarchical
  )
  class(bvhar_param) <- "bvharspec"
  bvhar_param
}

#' Prior for Constant Term
#'
#' Set Normal prior hyperparameters for constant term
#'
#' @param mean Normal mean of constant term
#' @param sd Normal standard deviance for constant term
#'
#' @order 1
#' @export
set_intercept <- function(mean = 0, sd = .1) {
  if (!is.vector(mean)) {
    stop("'mean' should be a vector.")
  }
  if (length(sd) != 1) {
    stop("'sd' should be length 1 numeric.")
  }
  if (sd < 0) {
    stop("'sd' should be positive.")
  }
  non_param <- list(
    process = "Intercept",
    prior = "Normal",
    mean_non = mean,
    sd_non = sd
  )
  class(non_param) <- c("interceptspec", "bvharspec")
  non_param
}

#' Stochastic Search Variable Selection (SSVS) Hyperparameter for Coefficients Matrix and Cholesky Factor
#'
#' Set SSVS hyperparameters for VAR or VHAR coefficient matrix and Cholesky factor.
#'
#' @param spike_grid Griddy gibbs grid size for scaling factor (between 0 and 1) of spike sd which is Spike sd = c * slab sd
#' @param slab_shape Inverse gamma shape for slab sd
#' @param slab_scl Inverse gamma scale for slab sd
#' @param s1 First shape of coefficients prior beta distribution
#' @param s2 Second shape of coefficients prior beta distribution
#' @param shape Gamma shape parameters for precision matrix (See Details).
#' @param rate Gamma rate parameters for precision matrix (See Details).
#' @details 
#' Let \eqn{\alpha} be the vectorized coefficient, \eqn{\alpha = vec(A)}.
#' Spike-slab prior is given using two normal distributions.
#' \deqn{\alpha_j \mid \gamma_j \sim (1 - \gamma_j) N(0, \tau_{0j}^2) + \gamma_j N(0, \tau_{1j}^2)}
#' As spike-slab prior itself suggests, set \eqn{\tau_{0j}} small (point mass at zero: spike distribution)
#' and set \eqn{\tau_{1j}} large (symmetric by zero: slab distribution).
#' 
#' \eqn{\gamma_j} is the proportion of the nonzero coefficients and it follows
#' \deqn{\gamma_j \sim Bernoulli(p_j)}
#' 
#' * `coef_spike`: \eqn{\tau_{0j}}
#' * `coef_slab`: \eqn{\tau_{1j}}
#' * `coef_mixture`: \eqn{p_j}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * If one value is provided, model function will read it by replicated value.
#' * `coef_non`: vectorized constant term is given prior Normal distribution with variance \eqn{cI}. Here, `coef_non` is \eqn{\sqrt{c}}.
#' 
#' Next for precision matrix \eqn{\Sigma_e^{-1}}, SSVS applies Cholesky decomposition.
#' \deqn{\Sigma_e^{-1} = \Psi \Psi^T}
#' where \eqn{\Psi = \{\psi_{ij}\}} is upper triangular.
#' 
#' Diagonal components follow the gamma distribution.
#' \deqn{\psi_{jj}^2 \sim Gamma(shape = a_j, rate = b_j)}
#' For each row of off-diagonal (upper-triangular) components, we apply spike-slab prior again.
#' \deqn{\psi_{ij} \mid w_{ij} \sim (1 - w_{ij}) N(0, \kappa_{0,ij}^2) + w_{ij} N(0, \kappa_{1,ij}^2)}
#' \deqn{w_{ij} \sim Bernoulli(q_{ij})}
#' 
#' * `shape`: \eqn{a_j}
#' * `rate`: \eqn{b_j}
#' * `chol_spike`: \eqn{\kappa_{0,ij}}
#' * `chol_slab`: \eqn{\kappa_{1,ij}}
#' * `chol_mixture`: \eqn{q_{ij}}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * \eqn{i = 1, \ldots, j - 1} and \eqn{j = 2, \ldots, m}: \eqn{\eta = (\psi_{12}, \psi_{13}, \psi_{23}, \psi_{14}, \ldots, \psi_{34}, \ldots, \psi_{1m}, \ldots, \psi_{m - 1, m})^T}
#' * `chol_` arguments can be one value for replication, vector, or upper triangular matrix.
#' @return `ssvsinput` object
#' @references 
#' George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881-889.
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553-580.
#' 
#' Ishwaran, H., & Rao, J. S. (2005). *Spike and slab variable selection: Frequentist and Bayesian strategies*. The Annals of Statistics, 33(2).
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267-358.
#' @order 1
#' @export
set_ssvs <- function(spike_grid = 100L,
                     slab_shape = .01,
                     slab_scl = .01,
                     s1 = c(1, 1),
                     s2 = c(1, 1),
                     shape = .01,
                     rate = .01) {
  if (!(is.vector(shape) && is.vector(rate))) {
    stop("'shape' and 'rate' be a vector.")
  }
  # if (!(length(chol_s1) == 1 && length(chol_s2 == 1))) {
  #   stop("'chol_s1' and 'chol_s2' should be length 1 numeric.")
  # }
  # if (!(length(coef_slab_shape) == 1 && length(chol_slab_shape) == 1)) {
  #   stop("'*_slab_*' should be length 1 numeric.")
  # }
  # if (!(length(coef_s1) == 2 && length(coef_s2 == 2))) {
  #   stop("'coef_s1' and 'coef_s2' should be length 2 numeric, each indicating own and cross lag.")
  # }
  # if (coef_s1[1] < coef_s2[1]) {
  #   stop("'coef_s1[1]' should be same or larger than 'coef_s2[1]'.") # own-lag
  # }
  # if (coef_s1[2] > coef_s2[2]) {
  #   stop("'coef_s1[2]' should be same or smaller than 'coef_s2[2]'.") # cross-lag
  # }
  # coefficients---------------------
  res <- list(
    shape = shape,
    rate = rate,
    # coef_spike_scl = coef_spike_scl,
    grid_size = spike_grid,
    slab_shape = slab_shape,
    slab_scl = slab_scl,
    s1 = s1,
    s2 = s2,
    process = "VAR",
    prior = "SSVS"
  )
  # non_param <- list(
  #   mean_non = mean_non,
  #   sd_non = sd_non
  # )
  len_param <- sapply(res, length)
  if (length(unique(len_param[len_param != 1])) > 1) {
    stop("The length of 'coef_spike', 'coef_slab', and 'coef_mixture' should be the same.")
  }
  # res <- append(coef_param, non_param)
  # # cholesky factor-------------------
  # chol_param <- list(
  #   shape = shape,
  #   rate = rate,
  #   # chol_spike_scl = chol_spike_scl,
  #   chol_grid = 100,
  #   chol_slab_shape = chol_slab_shape,
  #   chol_slab_scl = chol_slab_scl,
  #   chol_s1 = chol_s1,
  #   chol_s2 = chol_s2,
  #   process = "VAR",
  #   prior = "SSVS"
  # )
  # len_param <- sapply(chol_param, length)
  # len_gamma <- len_param[1:2]
  # len_eta <- len_param[3:5]
  # if (length(unique(len_gamma[len_gamma != 1])) > 1) {
  #   stop("The length of 'shape' and 'rate' should be the same.")
  # }
  # if (length(unique(len_eta[len_eta != 1])) > 1) {
  #   stop("The size of 'chol_spike', 'chol_slab', and 'chol_mixture' should be the same.")
  # }
  # res <- append(res, chol_param)
  class(res) <- "ssvsinput"
  res
}

#' Horseshoe Prior Specification
#'
#' Set initial hyperparameters and parameter before starting Gibbs sampler for Horseshoe prior.
#'
#' @param local_sparsity Initial local shrinkage hyperparameters
#' @param group_sparsity Initial group shrinkage hyperparameters
#' @param global_sparsity Initial global shrinkage hyperparameter
#' @details
#' Set horseshoe prior initialization for VAR family.
#'
#' * `local_sparsity`: Initial local shrinkage
#' * `group_sparsity`: Initial group shrinkage
#' * `global_sparsity`: Initial global shrinkage
#'
#' In this package, horseshoe prior model is estimated by Gibbs sampling,
#' initial means initial values for that gibbs sampler.
#' @references
#' Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. Biometrika, 97(2), 465-480.
#'
#' Makalic, E., & Schmidt, D. F. (2016). *A Simple Sampler for the Horseshoe Estimator*. IEEE Signal Processing Letters, 23(1), 179-182.
#' @order 1
#' @export
set_horseshoe <- function(local_sparsity = 1, group_sparsity = 1, global_sparsity = 1) {
  if (!is.vector(local_sparsity)) {
    stop("'local_sparsity' should be a vector.")
  }
  # if (length(local_sparsity) > 1) {
  #   warning("Scalar 'local_sparsity' works.")
  # }
  # if (!is.matrix(init_cov)) {
  #   stop("'init_cov' should be a matrix.")
  # }
  # if (ncol(init_cov) != nrow(init_cov)) {
  #   stop("'init_cov' should be a square matrix.")
  # }
  if (length(global_sparsity) > 1) {
    stop("'global_sparsity' should be a scalar.")
  }
  res <- list(
    process = "VAR",
    prior = "Horseshoe",
    local_sparsity = local_sparsity,
    group_sparsity = group_sparsity,
    global_sparsity = global_sparsity # ,init_cov = init_cov
  )
  class(res) <- "horseshoespec"
  res
}

#' Normal-Gamma Hyperparameter for Coefficients and Contemporaneous Coefficients
#'
#' `r lifecycle::badge("experimental")` Set NG hyperparameters for VAR or VHAR coefficient and contemporaneous coefficient.
#'
#' @param shape_sd Standard deviation used in MH of Gamma shape
#' @param group_shape Inverse gamma prior shape for coefficient group shrinkage
#' @param group_scale Inverse gamma prior scale for coefficient group shrinkage
#' @param global_shape Inverse gamma prior shape for coefficient global shrinkage
#' @param global_scale Inverse gamma prior scale for coefficient global shrinkage
#' @return `ngspec` object
#' @references
#' Chan, J. C. C. (2021). *Minnesota-type adaptive hierarchical priors for large Bayesian VARs*. International Journal of Forecasting, 37(3), 1212-1226.
#' 
#' Huber, F., & Feldkircher, M. (2019). *Adaptive Shrinkage in Bayesian Vector Autoregressive Models*. Journal of Business & Economic Statistics, 37(1), 27-39.
#' 
#' Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4), 230-354.
#' @order 1
#' @export
set_ng <- function(shape_sd = .01,
                   group_shape = .01,
                   group_scale = .01,
                   global_shape = .01,
                   global_scale = .01) {
  # if (!(is.vector(local_shape) && is.vector(contem_shape))) {
  #   stop("'local_shape' and 'contem_shape' should be a vector.")
  # }
  if (!(
    length(shape_sd) == 1 &&
      length(group_shape) == 1 &&
      length(group_scale) == 1 &&
      length(global_shape) == 1 &&
      length(global_scale) == 1
  )) {
    stop("'group_shape', 'group_scale', 'global_shape', 'global_scale', 'contem_global_shape' and 'contem_global_scale' should be length 1 numeric.")
  }
  res <- list(
    process = "VAR",
    prior = "NG",
    shape_sd = shape_sd,
    # local_shape = local_shape,
    group_shape = group_shape,
    group_scale = group_scale,
    global_shape = global_shape,
    global_scale = global_scale
    # # contem_shape = contem_shape,
    # contem_global_shape = contem_global_shape,
    # contem_global_scale = contem_global_scale
  )
  class(res) <- "ngspec"
  res
}

#' Dirichlet-Laplace Hyperparameter for Coefficients and Contemporaneous Coefficients
#'
#' `r lifecycle::badge("experimental")` Set DL hyperparameters for VAR or VHAR coefficient and contemporaneous coefficient.
#'
#' @param dir_grid Griddy gibbs grid size for Dirichlet hyperparameter
#' @param shape Inverse Gamma shape
#' @param scale Inverse Gamma scale
#' @return `dlspec` object
#' @references
#' Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015). *Dirichlet-Laplace Priors for Optimal Shrinkage*. Journal of the American Statistical Association, 110(512), 1479-1490.
#'
#' Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4), 230-354.
#' @order 1
#' @export
set_dl <- function(dir_grid = 100L, shape = .01, scale = .01) {
  if (!(length(dir_grid) == 1 && length(shape) == 1 && length(scale) == 1)) {
    stop("'dirichlet', 'contem_dirichlet', 'shape', and 'scale' should be length 1 numeric.")
  }
  if (dir_grid %% 1 != 0) {
    stop("Provide integer for 'dir_grid'.")
  }
  res <- list(
    process = "VAR",
    prior = "DL",
    grid_size = dir_grid,
    shape = shape,
    scale = scale
  )
  class(res) <- "dlspec"
  res
}

#' Generalized Double Pareto Shrinkage Hyperparameters for Coefficients and Contemporaneous Coefficients
#'
#' `r lifecycle::badge("experimental")` Set GDP hyperparameters for VAR or VHAR coefficient and contemporaneous coefficient.
#'
#' @param shape_grid Griddy gibbs grid size for Gamma shape hyperparameter
#' @param rate_grid Griddy gibbs grid size for Gamma rate hyperparameter
#' @return `gdpspec` object
#' @references
#' Armagan, A., Dunson, D. B., & Lee, J. (2013). *GENERALIZED DOUBLE PARETO SHRINKAGE*. Statistica Sinica, 23(1), 119–143.
#'
#' Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4), 230-354.
#' @order 1
#' @export
set_gdp <- function(shape_grid = 100L, rate_grid = 100L) {
  if (!(length(shape_grid) == 1 && length(rate_grid))) {
    stop("'shape_grid' and 'rate_grid' should be length 1 numeric.")
  }
  if (!(shape_grid %% 1 == 0 && rate_grid %% 1 == 0)) {
    stop("Provide integer for 'shape_grid' and 'rate_grid'.")
  }
  res <- list(
    process = "VAR",
    prior = "GDP",
    grid_shape = shape_grid,
    grid_rate = rate_grid
  )
  class(res) <- "gdpspec"
  res
}

#' Covariance Matrix Prior Specification
#'
#' `r lifecycle::badge("experimental")` Set prior for covariance matrix.
#'
#' @param ig_shape Inverse-Gamma shape of Cholesky diagonal vector.
#' For SV ([set_sv()]), this is for state variance.
#' @param ig_scl Inverse-Gamma scale of Cholesky diagonal vector.
#' For SV ([set_sv()]), this is for state variance.
#' @details
#' [set_ldlt()] specifies LDLT of precision matrix,
#' \deqn{\Sigma^{-1} = L^T D^{-1} L}
#' @order 1
#' @export
set_ldlt <- function(ig_shape = 3, ig_scl = .01) {
  if (!is.vector(ig_shape) ||
    !is.vector(ig_scl)) {
    stop("'ig_shape' and 'ig_scl' should be a vector.")
  }
  if ((length(ig_shape) != length(ig_scl))) {
    stop("'ig_shape' and 'ig_scl' should have same length.")
  }
  res <- list(
    process = "Homoskedastic",
    prior = "Cholesky",
    shape = ig_shape,
    scale = ig_scl
  )
  class(res) <- c("ldltspec", "covspec")
  res
}

#' @rdname set_ldlt
#' @param initial_mean Prior mean of initial state.
#' @param initial_prec Prior precision of initial state.
#' @details
#' [set_sv()] specifices time varying precision matrix under stochastic volatility framework based on
#' \deqn{\Sigma_t^{-1} = L^T D_t^{-1} L}
#' @references
#' Carriero, A., Chan, J., Clark, T. E., & Marcellino, M. (2022). *Corrigendum to “Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors” \[J. Econometrics 212 (1)(2019) 137-154\]*. Journal of Econometrics, 227(2), 506-512.
#'
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press.
#' @order 1
#' @export
set_sv <- function(ig_shape = 3, ig_scl = .01, initial_mean = 1, initial_prec = .1) {
  if (!is.vector(ig_shape) ||
    !is.vector(ig_scl) ||
    !is.vector(initial_mean)) {
    stop("'ig_shape', 'ig_scl', and 'initial_mean' should be a vector.")
  }
  if ((length(ig_shape) != length(ig_scl)) ||
    (length(ig_scl) != length(initial_mean))) {
    stop("'ig_shape', 'ig_scl', and 'initial_mean' should have same length.")
  }
  if (is.vector(initial_prec) && length(initial_prec) > 1) {
    initial_prec <- diag(initial_prec)
  }
  if (is.matrix(initial_prec)) {
    if ((length(ig_shape) != nrow(initial_prec))
        || (length(ig_shape) != ncol(initial_prec))) {
      stop("'initial_prec' should be symmetric matrix of same size with the other vectors.")
    }
  }
  res <- list(
    process = "SV",
    prior = "Cholesky",
    shape = ig_shape,
    scale = ig_scl,
    initial_mean = initial_mean,
    initial_prec = initial_prec
  )
  class(res) <- c("svspec", "covspec")
  res
}
