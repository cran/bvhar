% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary-forecast.R
\name{rmafe}
\alias{rmafe}
\alias{rmafe.predbvhar}
\alias{rmafe.bvharcv}
\title{Evaluate the Model Based on RMAFE}
\usage{
rmafe(x, pred_bench, y, ...)

\method{rmafe}{predbvhar}(x, pred_bench, y, ...)

\method{rmafe}{bvharcv}(x, pred_bench, y, ...)
}
\arguments{
\item{x}{Forecasting object to use}

\item{pred_bench}{The same forecasting object from benchmark model}

\item{y}{Test data to be compared. should be the same format with the train data.}

\item{...}{not used}
}
\value{
RMAFE vector corresponding to each variable.
}
\description{
This function computes RMAFE (Mean Absolute Forecast Error Relative to the Benchmark)
}
\details{
Let \eqn{e_t = y_t - \hat{y}_t}.
RMAFE is the ratio of L1 norm of \eqn{e_t} from forecasting object and from benchmark model.

\deqn{RMAFE = \frac{sum(\lVert e_t \rVert)}{sum(\lVert e_t^{(b)} \rVert)}}

where \eqn{e_t^{(b)}} is the error from the benchmark model.
}
\references{
Hyndman, R. J., & Koehler, A. B. (2006). \emph{Another look at measures of forecast accuracy}. International Journal of Forecasting, 22(4), 679-688.

Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1).

Ghosh, S., Khare, K., & Michailidis, G. (2018). \emph{High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models}. Journal of the American Statistical Association, 114(526).
}
