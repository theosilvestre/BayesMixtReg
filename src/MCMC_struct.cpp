#include "MCMC_struct.hpp"
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <thread>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]


const arma::rowvec createRowVec(const int n){ return arma::rowvec(n,arma::fill::zeros); }