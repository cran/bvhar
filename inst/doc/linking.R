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

## ----eval=FALSE---------------------------------------------------------------
#  Sys.setenv("PKG_CPPFLAGS" = "-DUSE_RCPP")

## ----resetopts, include=FALSE-------------------------------------------------
options(orig_opts)

