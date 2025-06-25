#ifndef BVHAR_BAYES_SHRINKAGE_CONFIG_H
#define BVHAR_BAYES_SHRINKAGE_CONFIG_H

#include "../misc/draw.h"
#include "../../math/design.h"

namespace bvhar {

// Parameters
struct ShrinkageParams;
struct MinnParams;
struct HierminnParams;
struct SsvsParams;
// struct HorseshoeParams;
struct NgParams;
struct DlParams;
struct GdpParams;
// Initialization
struct ShrinkageInits;
struct HierminnInits;
struct SsvsInits;
struct GlInits;
struct HorseshoeInits;
struct NgInits;
struct GdpInits;

/**
 * @brief Hyperparameters for shrinkage priors `ShrinkageUpdater`
 * 
 */
struct ShrinkageParams {
	ShrinkageParams() {}
	ShrinkageParams(LIST& priors) {}
};

/**
 * @brief Hyperparameters for Minnesota prior `MinnUpdater`
 * 
 */
struct MinnParams : public ShrinkageParams {
	// Eigen::MatrixXd _prec_diag;
	Eigen::VectorXd _prior_prec, _prior_mean;
	
	MinnParams(LIST& priors)
	: ShrinkageParams(priors) {
		int lag = CAST_INT(priors["p"]); // append to bayes_spec, p = 3 in VHAR
		Eigen::VectorXd sigma = CAST<Eigen::VectorXd>(priors["sigma"]);
		// double lam = CAST_DOUBLE(priors["lambda"]);
		double lam;
		if (CAST_BOOL(priors["hierarchical"])) {
			lam = 1; // when Hierminn
		} else {
			lam = CAST_DOUBLE(priors["lambda"]); // when Minn
		}
		double eps = CAST_DOUBLE(priors["eps"]);
		int dim = sigma.size();
		Eigen::MatrixXd prec_diag = Eigen::MatrixXd::Zero(dim, dim);
		Eigen::VectorXd daily(dim);
		Eigen::VectorXd weekly(dim);
		Eigen::VectorXd monthly(dim);
		if (CONTAINS(priors, "delta")) {
			daily = CAST<Eigen::VectorXd>(priors["delta"]);
			weekly.setZero();
			monthly.setZero();
		} else {
			daily = CAST<Eigen::VectorXd>(priors["daily"]);
			weekly = CAST<Eigen::VectorXd>(priors["weekly"]);
			monthly = CAST<Eigen::VectorXd>(priors["monthly"]);
		}
		Eigen::MatrixXd dummy_response = build_ydummy(lag, sigma, lam, daily, weekly, monthly, false);
		Eigen::MatrixXd dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(lag, 1, lag),
			lam, sigma, eps, false
		);
		Eigen::MatrixXd prior_prec = dummy_design.transpose() * dummy_design;
		_prior_mean = prior_prec.llt().solve(dummy_design.transpose() * dummy_response).reshaped();
		prec_diag.diagonal() = 1 / sigma.array();
		_prior_prec = kronecker_eigen(prec_diag, prior_prec).diagonal();
	}

	MinnParams(LIST& priors, int num_lowerchol)
	: ShrinkageParams(priors),
		_prior_prec(Eigen::VectorXd::Ones(num_lowerchol)),
		_prior_mean(Eigen::VectorXd::Zero(num_lowerchol)) {}
};

/**
 * @brief Hyperparameters for hierarchical Minnesota prior `HierminnUpdater`
 * 
 */
struct HierminnParams : public MinnParams {
	double _shape, _rate;
	int _grid_size;
	// bool _minnesota;

	HierminnParams(LIST& priors)
	: MinnParams(priors),
		_shape(CAST_DOUBLE(priors["shape"])), _rate(CAST_DOUBLE(priors["rate"])), _grid_size(CAST_INT(priors["grid_size"])) {}
	
	HierminnParams(LIST& priors, int num_lowerchol)
	: MinnParams(priors, num_lowerchol),
		_shape(CAST_DOUBLE(priors["shape"])), _rate(CAST_DOUBLE(priors["rate"])), _grid_size(CAST_INT(priors["grid_size"])) {}
};

/**
 * @brief Hyperparameters for SSVS prior `SsvsUpdater`
 * 
 */
struct SsvsParams : public ShrinkageParams {
	Eigen::VectorXd _s1, _s2;
	double _slab_shape, _slab_scl;
	int _grid_size;

	SsvsParams(LIST& priors)
	: ShrinkageParams(priors),
		_s1(CAST<Eigen::VectorXd>(priors["s1"])), _s2(CAST<Eigen::VectorXd>(priors["s2"])),
		_slab_shape(CAST_DOUBLE(priors["slab_shape"])), _slab_scl(CAST_DOUBLE(priors["slab_scl"])),
		_grid_size(CAST_INT(priors["grid_size"])) {}
};

/**
 * @brief Hyperparameters for Normal-gamma prior `NgUpdater`
 * 
 */
struct NgParams : public ShrinkageParams {
	double _mh_sd, _group_shape, _group_scl, _global_shape, _global_scl;

	NgParams(LIST& priors)
	: ShrinkageParams(priors),
		_mh_sd(CAST_DOUBLE(priors["shape_sd"])),
		_group_shape(CAST_DOUBLE(priors["group_shape"])), _group_scl(CAST_DOUBLE(priors["group_scale"])),
		_global_shape(CAST_DOUBLE(priors["global_shape"])), _global_scl(CAST_DOUBLE(priors["global_scale"])) {}
};

/**
 * @brief Hyperparameters for Dirichlet-Laplace prior `DlUpdater`
 * 
 */
struct DlParams : public ShrinkageParams {
	int _grid_size;
	double _shape, _scl;

	DlParams(LIST& priors)
	: ShrinkageParams(priors), _grid_size(CAST_INT(priors["grid_size"])), _shape(CAST_DOUBLE(priors["shape"])), _scl(CAST_DOUBLE(priors["scale"])) {}
};

/**
 * @brief Hyperparameters for GDP prior `GdpUpdater`
 * 
 */
struct GdpParams : public ShrinkageParams {
	int _grid_shape, _grid_rate;

	GdpParams(LIST& priors)
	: ShrinkageParams(priors), _grid_shape(CAST_INT(priors["grid_shape"])), _grid_rate(CAST_INT(priors["grid_rate"])) {}
};

/**
 * @brief MCMC initial values for `ShrinkageUpdater`
 * 
 */
struct ShrinkageInits {
	ShrinkageInits() {}
	ShrinkageInits(LIST& init) {}
	ShrinkageInits(LIST& init, int num_design) {}
};

/**
 * @brief MCMC initial values for `HierminnUpdater`
 * 
 */
struct HierminnInits : public ShrinkageInits {
	double _own_lambda;
	double _cross_lambda;

	HierminnInits(LIST& init)
	: ShrinkageInits(init), _own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])) {}

	HierminnInits(LIST& init, int num_design)
	: ShrinkageInits(init, num_design), _own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])) {}
};

/**
 * @brief MCMC initial values for `SsvsUpdater`
 * 
 */
struct SsvsInits : public ShrinkageInits {
	Eigen::VectorXd _dummy, _weight, _slab;
	double _spike_scl;

	SsvsInits(LIST& init)
	: ShrinkageInits(init),
		_dummy(CAST<Eigen::VectorXd>(init["dummy"])),
		_weight(CAST<Eigen::VectorXd>(init["mixture"])),
		_slab(CAST<Eigen::VectorXd>(init["slab"])),
		_spike_scl(CAST_DOUBLE(init["spike_scl"])) {}
	
	SsvsInits(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_dummy(CAST<Eigen::VectorXd>(init["dummy"])),
		_weight(CAST<Eigen::VectorXd>(init["mixture"])),
		_slab(CAST<Eigen::VectorXd>(init["slab"])),
		_spike_scl(CAST_DOUBLE(init["spike_scl"])) {}
};

/**
 * @brief MCMC initial values for global-local shrinkage prior.
 * 
 */
struct GlInits : public ShrinkageInits {
	Eigen::VectorXd _local;
	double _global;

	GlInits(LIST& init)
	: ShrinkageInits(init),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_global(CAST_DOUBLE(init["global_sparsity"])) {}
	
	GlInits(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_global(CAST_DOUBLE(init["global_sparsity"])) {}
};

/**
 * @brief MCMC initial values for `HorseshoeUpdater`
 * 
 */
struct HorseshoeInits : public GlInits {
	Eigen::VectorXd _group;

	HorseshoeInits(LIST& init)
	: GlInits(init),
		_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
	
	HorseshoeInits(LIST& init, int num_design)
	: GlInits(init, num_design),
		_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
};

/**
 * @brief MCMC initial values for `NgUpdater`
 * 
 */
struct NgInits : public HorseshoeInits {
	Eigen::VectorXd _local_shape;

	NgInits(LIST& init)
	: HorseshoeInits(init),
		_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])) {}
	
	NgInits(LIST& init, int num_design)
	: HorseshoeInits(init, num_design),
		_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])) {}
};

/**
 * @brief MCMC initial values for `GdpUpdater`
 * 
 */
struct GdpInits : public ShrinkageInits {
	Eigen::VectorXd _local, _group_rate;
	double _gamma_shape, _gamma_rate;

	GdpInits(LIST& init)
	: ShrinkageInits(init),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _gamma_rate(CAST_DOUBLE(init["gamma_rate"])) {}
	
	GdpInits(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _gamma_rate(CAST_DOUBLE(init["gamma_rate"])) {}
};

} // namespace bvhar

#endif // BVHAR_BAYES_SHRINKAGE_CONFIG_H
