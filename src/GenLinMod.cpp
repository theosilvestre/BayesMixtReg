#include "GenLinMod.h"
#include <RcppArmadillo.h>
#include <math.h>
#include <vector>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp20)]]  

//' Logistic binomial regression / conditional inference with RWMH method 
//'
//' @description Performs logistic binomial regression using the RWMH method to infer the coefficients conditionally
//'
//' @param N the number of iterations post-adaptative step
//' @param nbAdaptMarche the number of adaptation steps, one step equals 100 iterations
//' @param y a vector of count values, from 0 to maxBinom[i] for y[i], i in 1...length(y)
//' @param X a matrix of covariates, with dim(X)[1] = length(y)
//' @param maxBinom a vector of maximum trials (in binomial fashion), length(maxBinom) = length(y)
//' @param curr a vector of initialization values corresponding to each of the covariates in the columns of X, ordered in the same way
//' @param SigMarch a covariance matrix for the Gaussian proposal distribution
//' @param prior a list of vectors for the priors' hyperparameters
//' @param warmup the number of iterations reserved to warmup (warmup >= 0 and warmup <= N)
//' @param invTemp a value to temper the likelihood
//' @return a list that includes a matrix of each coefficient posterior sample and the adapted covariance matrix for the Gaussian proposal distribution
// [[Rcpp::export(.logisticRWMHcond)]]
Rcpp::List logisticRWMHcond(int N, int nbAdaptMarche, Rcpp::NumericVector y, Rcpp::NumericMatrix X, Rcpp::NumericVector maxBinom, std::vector<double> curr, std::vector<double> SigMarch, Rcpp::List prior,const int warmup=0,const double invTemp=1)
{
  // /!\ Why use Rcpp::NumericVector instead of arma::vector? because these implicit conversions do not work in packages (see ALTREP)
  arma::vec ysafe = arma::vec(y.begin(), y.size(), false, true).eval();
  arma::mat Xsafe = arma::mat(X.begin(), X.nrow(), X.ncol(), false, true).eval();
  arma::vec maxBinomsafe = arma::vec(maxBinom.begin(), maxBinom.size(), false, true).eval();
  std::vector<double> currsafe(curr.begin(),curr.end());
  std::vector<double> SigMarchsafe(SigMarch.begin(),SigMarch.end());
  
  auto lambdaPrior = [&prior](){
    std::vector<NormalPrior> tmp;
    tmp.reserve(prior.size());
    for(int i=0;i<prior.size();++i){
      NumericVector vecPrior = prior[i];
      tmp.push_back(NormalPrior(vecPrior[0],vecPrior[1]));
    }
    return tmp;
  }();
  GlmRWMHcond<GlmLogit,NormalPrior> LOG(Data(ysafe,Xsafe,maxBinomsafe),AlgoRWMH(N,nbAdaptMarche,warmup,invTemp),currsafe,SigMarchsafe,lambdaPrior);

  RWMHCore(LOG);

  return Rcpp::List::create(Rcpp::Named("coeff") = [&LOG](){
                              arma::vec tmp(LOG.regCoeff.size());
                              for(int i=0;i<LOG.regCoeff.size();++i) tmp[i] = LOG.regCoeff[i].curr;
                              return tmp;
                            }(),
                            Rcpp::Named("SigMarch") = [&LOG](){
                              arma::vec tmp(LOG.regCoeff.size());
                              for(int i=0;i<LOG.regCoeff.size();++i) tmp[i] = LOG.regCoeff[i].sig;
                              return tmp;
                            }()
  );
}

//' Logistic binomial regression / joint inference with RWMH method 
//'
//' @description This function performs a logistic binomial regression using the RWMH method to jointly infer the coefficients.
//'
//' @param N the number of iterations post-adaptative step
//' @param nbAdaptMarche the number of adaptation steps, one step equals 100 iterations
//' @param y a vector of count values, from 0 to maxBinom[i] for y[i], i in 1...length(y)
//' @param X a matrix of covariates, with dim(X)[1] = length(y)
//' @param maxBinom a vector of maximum trials (in binomial fashion), length(maxBinom) = length(y)
//' @param curr a vector of initialization values corresponding to each of the covariates in the columns of X, ordered in the same way
//' @param SigMarch a covariance matrix for the Gaussian proposal distribution
//' @param prior a list of vectors for the priors' hyperparameters
//' @param warmup the number of iterations reserved to warmup (warmup >= 0 and warmup <= N)
//' @param invTemp a value to temper the likelihood
//' @return a list that includes a matrix of each coefficient posterior sample, the adapted covariance matrix for the Gaussian proposal distribution and some values to compute WAIC
// [[Rcpp::export(.logisticRWMHjoint)]]
Rcpp::List logisticRWMHjoint(int N, int nbAdaptMarche, Rcpp::NumericVector y, Rcpp::NumericMatrix X, Rcpp::NumericVector maxBinom, Rcpp::NumericVector curr, Rcpp::NumericMatrix SigMarch, Rcpp::List prior,const int warmup=0,const double invTemp=1)
{
  // /!\ Why use Rcpp::NumericVector instead of arma::vector? because these implicit conversions do not work in packages (see ALTREP)
  arma::vec ysafe = arma::vec(y.begin(), y.size(), false, true).eval();
  arma::mat Xsafe = arma::mat(X.begin(), X.nrow(), X.ncol(), false, true).eval();
  arma::vec maxBinomsafe = arma::vec(maxBinom.begin(), maxBinom.size(), false, true).eval();
  arma::vec currsafe = arma::vec(curr.begin(), curr.size(), false, true).eval();
  arma::mat SigMarchsafe = arma::mat(SigMarch.begin(), SigMarch.nrow(), SigMarch.ncol(), false, true).eval();
  arma::vec prior0 = Rcpp::as<arma::vec>(prior[0]);
  arma::mat prior1 = Rcpp::as<arma::mat>(prior[1]);
  GlmRWMHjoint<GlmLogit,MultivariateNormalPrior> LOG(Data(ysafe,Xsafe,maxBinomsafe),AlgoRWMH(N,nbAdaptMarche,warmup,invTemp),currsafe,SigMarchsafe,MultivariateNormalPrior(prior0,prior1));

  RWMHCore(LOG);

  return Rcpp::List::create(Rcpp::Named("coeff") = LOG.regCoeff.res,
                            Rcpp::Named("SigMarch") = LOG.regCoeff.sig,
                            Rcpp::Named("logDens") = LOG.algo.logDens,
                            Rcpp::Named("logDensCarre") = LOG.algo.logDensCarre,
                            Rcpp::Named("dens") = LOG.algo.dens,
                            Rcpp::Named("S") = LOG.algo.S);
}

const arma::vec likelihoodIndiv(const arma::vec& XprodCour,const arma::vec& y,const arma::vec& maxBinom,const arma::vec& tempCour){
  return XprodCour%y - maxBinom % tempCour;
}



