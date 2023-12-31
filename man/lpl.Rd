% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary-forecast.R
\name{lpl}
\alias{lpl}
\alias{lpl.predsv}
\title{Evaluate the Model Based on Log Predictive Likelihood}
\usage{
lpl(x, y, ...)

\method{lpl}{predsv}(x, y, ...)
}
\arguments{
\item{x}{Forecasting object}

\item{y}{Test data to be compared. should be the same format with the train data.}

\item{...}{not used}
}
\description{
This function computes LPL given prediction result versus evaluation set.
}
\references{
Cross, J. L., Hou, C., & Poon, A. (2020). \emph{Macroeconomic forecasting with large Bayesian VARs: Global-local priors and the illusion of sparsity}. International Journal of Forecasting, 36(3), 899–915.

Gruber, L., & Kastner, G. (2022). \emph{Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!} arXiv.
}
