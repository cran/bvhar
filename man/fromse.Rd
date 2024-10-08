% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary-sparse.R
\name{fromse}
\alias{fromse}
\alias{fromse.bvharsp}
\title{Evaluate the Estimation Based on Frobenius Norm}
\usage{
fromse(x, y, ...)

\method{fromse}{bvharsp}(x, y, ...)
}
\arguments{
\item{x}{Estimated model.}

\item{y}{Coefficient matrix to be compared.}

\item{...}{not used}
}
\value{
Frobenius norm value
}
\description{
This function computes estimation error given estimated model and true coefficient.
}
\details{
Consider the Frobenius Norm \eqn{\lVert \cdot \rVert_F}.
let \eqn{\hat{\Phi}} be nrow x k the estimates,
and let \eqn{\Phi} be the true coefficients matrix.
Then the function computes estimation error by
\deqn{MSE = 100 \frac{\lVert \hat{\Phi} - \Phi \rVert_F}{nrow \times k}}
}
\references{
Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global-local shrinkage priors. Journal of Multivariate Analysis, 167, 157-170.
}
