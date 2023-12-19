# bvhar 1.1.0

* For new research, add new features for shrinkage priors.

* Add Shrinkage priors SSVS and Horseshoe (`bvar_ssvs()`, `bvhar_ssvs()`, `bvar_horseshoe()`, and `bvhar_horseshoe()`).

* `bvar_sv()`, `bvhar_sv()` works with SSVS (`set_ssvs()`) and Horseshoe (`set_horseshoe()`).

* Update the shrinkage structure in the spirit of Minnesota. (`minnesota = TRUE`, `minnesota = c("no", "short", "longrun")`).

* Stochastic volatility models implement corrected triangular algorithm of Carriero et al. (2021).

# bvhar 1.0.2

* License has been changed to [GPLv3](https://choosealicense.com/licenses/gpl-3.0/).

* Remove unnecessary Rcpp plugins in source files.

# bvhar 1.0.1

* Fix `knitr::knit_print()` method export methods [(#2)](https://github.com/ygeunkim/bvhar/issues/2).

# bvhar 1.0.0

* "Bayesian Vector Heterogeneous Autoregressive Modeling" has been accepted in JSCS 🎉

* Update to major version before publication.

# bvhar 0.14.1

# bvhar 0.14.0

* Add Stochastic Search Variable Selection (SSVS) models for VAR and VHAR (`bvar_ssvs()` and `bvhar_ssvs()`)

* Can do corresponding variable selection (`summary.ssvsmod()`)

# bvhar 0.13.0

* Add stochastic volatility models VAR-SV and VHAR-SV (`bvar_sv()` and `bvhar_sv()`).

# bvhar 0.12.1

* Fix not working Hierarchical natural conjugate MNIW function (`bvar_niwhm()`).

* Use `posterior` package for `summary.normaliw()` to improve processing and printing.

# bvhar 0.12.0

* Now can use heavy-tailed distribution ([Multivariate t-distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution)) when generating VAR and VHAR process (`sim_var()` and `sim_vhar()`).

* Also provide independent MVT generation function (`sim_mvt()`).

# bvhar 0.11.0

* Added `method = c("nor", "chol", "qr")` option in VAR and VHAR fitting function to use cholesky and Householder QR method (`var_lm()` and `vhar_lm()`).

* Now `include_mean` works internally with `Rcpp`.

# bvhar 0.10.0

* Add partial t-test for each VAR and VHAR coefficient (`summary.varlse()` and `summary.vharlse()`).

* Appropriate print method for the updated summary method (`print.summary.varlse()` and `print.summary.vharlse()`).

# bvhar 0.9.0

* Can compute impulse response function for VAR (`varlse`) and VHAR (`vharlse`) models (`analyze_ir()`).

* Can draw impulse -> response plot in grid panels (`autoplot.bvharirf()`).

# bvhar 0.8.0

* Changed the way of specifying the lower and upper bounds of empirical bayes (`bound_bvhar()`).

* Added Empirical Bayes vignette.

# bvhar 0.7.1

* When simulation, asymmetric covariance error is caught now (`sim_mgaussian()`).

# bvhar 0.7.0

* Add one integrated function that can do empirical bayes (`choose_bayes()` and `bound_bvhar()`).

# bvhar 0.6.1

* Pre-process date column of `oxfordman` more elaborately (it becomes same with `etf_vix`).

# bvhar 0.6.0

* Added weekly and monthly order feature in VHAR family (`vhar_lm()` and `bvhar_minnesota()`).

* Other functions are compatible with har order option (`predict.vharlse()`, `predict.bvharmn()`, and `choose_bvhar()`)

# bvhar 0.5.2

* Added parallel option for empirical bayes (`choose_bvar()` and `choose_bvhar()`).

# bvhar 0.5.1

* Added facet feature for the loss plot and changed its name (`gg_loss()`).

# bvhar 0.5.0

* Added rolling window and expanding window features (`forecast_roll()` and `forecast_expand()`).

* Can compute loss for each rolling and expanding window method (`mse.bvharcv()`, `mae.bvharcv()`, `mape.bvharcv()`, and `mape.bvharcv()`).

# bvhar 0.4.1

* Fix Marginal likelihood form (`compute_logml()`).

* Optimize empirical bayes method using stabilized marginal likelihood function (`logml_stable()`).

# bvhar 0.4.0

* Change the way to compute the CI of BVAR and BVHAR (`predict.bvarmn()`, `predict.bvharmn()`, and `predict.bvarflat()`)

* Used custom random generation function - MN, IW, and MNIW based on RcppEigen

# bvhar 0.3.0

* Added Bayesian model specification functions and class (`bvharspec`).

* Replaced hyperparameters with model specification in Bayesian models (`bvar_minnesota()`, `bvar_flat()`, and `bvhar_minnesota()`).

# bvhar 0.2.0

* Added constant term choice in each function (`var_lm()`, `vhar_lm()`, `bvar_minnesota()`, `bvar_flat()`, and `bvhar_minnesota()`).

# bvhar 0.1.0

* Added a `NEWS.md` file to track changes to the package.
