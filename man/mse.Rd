% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary-forecast.R
\name{mse}
\alias{mse}
\alias{mse.predbvhar}
\alias{mse.bvharcv}
\title{Evaluate the Model Based on MSE (Mean Square Error)}
\usage{
mse(x, y, ...)

\method{mse}{predbvhar}(x, y, ...)

\method{mse}{bvharcv}(x, y, ...)
}
\arguments{
\item{x}{Forecasting object}

\item{y}{Test data to be compared. should be the same format with the train data.}

\item{...}{not used}
}
\value{
MSE vector corresponding to each variable.
}
\description{
This function computes MSE given prediction result versus evaluation set.
}
\details{
Let \eqn{e_t = y_t - \hat{y}_t}. Then
\deqn{MSE = mean(e_t^2)}
MSE is the most used accuracy measure.
}
\references{
Hyndman, R. J., & Koehler, A. B. (2006). \emph{Another look at measures of forecast accuracy}. International Journal of Forecasting, 22(4), 679-688.
}
