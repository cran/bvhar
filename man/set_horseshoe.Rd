% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/hyperparam.R, R/print-bvharspec.R, R/member.R
\name{set_horseshoe}
\alias{set_horseshoe}
\alias{print.horseshoespec}
\alias{is.horseshoespec}
\alias{knit_print.horseshoespec}
\title{Horseshoe Prior Specification}
\usage{
set_horseshoe(local_sparsity = 1, group_sparsity = 1, global_sparsity = 1)

\method{print}{horseshoespec}(x, digits = max(3L, getOption("digits") - 3L), ...)

is.horseshoespec(x)

\method{knit_print}{horseshoespec}(x, ...)
}
\arguments{
\item{local_sparsity}{Initial local shrinkage hyperparameters}

\item{group_sparsity}{Initial group shrinkage hyperparameters}

\item{global_sparsity}{Initial global shrinkage hyperparameter}

\item{x}{Any object}

\item{digits}{digit option to print}

\item{...}{not used}
}
\description{
Set initial hyperparameters and parameter before starting Gibbs sampler for Horseshoe prior.
}
\details{
Set horseshoe prior initialization for VAR family.
\itemize{
\item \code{local_sparsity}: Initial local shrinkage
\item \code{group_sparsity}: Initial group shrinkage
\item \code{global_sparsity}: Initial global shrinkage
}

In this package, horseshoe prior model is estimated by Gibbs sampling,
initial means initial values for that gibbs sampler.
}
\references{
Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. Biometrika, 97(2), 465-480.

Makalic, E., & Schmidt, D. F. (2016). \emph{A Simple Sampler for the Horseshoe Estimator}. IEEE Signal Processing Letters, 23(1), 179-182.
}
