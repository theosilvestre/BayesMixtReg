#include "src/MCMC_struct.hpp"
#include <armadillo>
#include <math.h>
#include <vector>
#include <iostream>
#include <thread>
// [[Rcpp::plugins(cpp20)]] 

const arma::rowvec createRowVec(const int n){ return arma::rowvec(n,arma::fill::zeros); }