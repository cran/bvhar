#ifndef BVHARDRAW_H
#define BVHARDRAW_H

#include "bvharmisc.h"
#include "bvharprob.h"

Eigen::VectorXd build_ssvs_sd(Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd mixture_dummy);

Eigen::VectorXd ssvs_chol_diag(Eigen::MatrixXd sse_mat, Eigen::VectorXd DRD, Eigen::VectorXd shape, Eigen::VectorXd rate, int num_design);

Eigen::VectorXd ssvs_chol_off(Eigen::MatrixXd sse_mat, Eigen::VectorXd chol_diag, Eigen::VectorXd DRD);

Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec);

Eigen::VectorXd ssvs_coef(Eigen::VectorXd prior_mean, Eigen::VectorXd prior_sd, Eigen::MatrixXd XtX, Eigen::VectorXd coef_ols, Eigen::MatrixXd chol_factor);

Eigen::VectorXd ssvs_dummy(Eigen::VectorXd param_obs, Eigen::VectorXd sd_numer, Eigen::VectorXd sd_denom, Eigen::VectorXd slab_weight);

Eigen::MatrixXd build_inv_lower(int dim, Eigen::VectorXd lower_vec);

Eigen::VectorXd varsv_regression(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec, Eigen::MatrixXd innov_prec);

Eigen::VectorXd varsv_ht(Eigen::VectorXd pj, Eigen::VectorXd muj, Eigen::VectorXd sigj, Eigen::VectorXd sv_vec, double init_sv, double sv_sig, Eigen::VectorXd latent_vec, int nthreads);

Eigen::VectorXd varsv_sigh(Eigen::VectorXd shp, Eigen::VectorXd scl, Eigen::VectorXd init_sv, Eigen::MatrixXd h1);

Eigen::VectorXd varsv_h0(Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec, Eigen::VectorXd init_sv, Eigen::VectorXd h1, Eigen::VectorXd sv_sig);

#endif
