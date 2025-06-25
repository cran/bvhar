#ifndef BVHAR_BAYES_TRIANGULAR_REGFORECASTER_H
#define BVHAR_BAYES_TRIANGULAR_REGFORECASTER_H

#include "./forecaster.h"

namespace bvhar {

// Until updating cpp sources
using RegVarForecaster = CtaVarForecaster<RegForecaster>;
using RegVharForecaster = CtaVharForecaster<RegForecaster>;
using RegVarSelectForecaster = CtaVarSelectForecaster<RegForecaster>;
using RegVharSelectForecaster = CtaVharSelectForecaster<RegForecaster>;

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_REGFORECASTER_H
