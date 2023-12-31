% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/doc-models.R
\name{var_vec_formulation}
\alias{var_vec_formulation}
\title{Vectorized OLS Formulation}
\description{
This page specifies the OLS formulation, which is vectorized of \link{var_design_formulation}.
Notation and format here will be used in this entire package document.
}
\section{Vector Autoregressive Model}{
Recall k-dim VAR model \eqn{Y_0 = X_0 A + Z_0}.
Applying \eqn{vec} operation, we have

\deqn{vec(Y_0) = (I_k \otimes X_0) vec(A) + vec(Z_0)}

Estimating \eqn{\alpha = vec(A)} is equivalent to estimating usual OLS.
}

\section{Vector Heterogeneous Autoregressive Model}{
Recall k-dim VHAR model \deqn{Y_0 = X_1 \Phi + Z_0 = (X_0 C_{HAR}^\intercal) \Phi + Z_0}.
Then
\deqn{vec(Y_0) = (I_k \otimes X_0 C_{HAR}^\intercal) vec(\Phi) + vec(Z_0) = (I_k \otimes X_1) vec(\Phi) + vec(Z_0)}
}

\references{
Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing.
}
\keyword{internal}
