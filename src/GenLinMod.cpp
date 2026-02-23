#include "GenLinMod.hpp"
#include <RcppArmadillo.h>
#include <math.h>
#include <vector>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp20)]]  


// [[Rcpp::export]]
Rcpp::List logisticRWMHcond(int N, int nbAdaptMarche, arma::vec y, arma::mat X, arma::vec maxBinom, std::vector<double> curr, std::vector<double> SigMarch, std::vector<std::vector<double>> prior,const int warmup=0,const double invTemp=1)
{
  auto lambdaPrior = [&prior](){
    std::vector<NormalPrior> tmp;
    tmp.reserve(prior.size());
    for(int i=0;i<prior.size();++i) tmp.push_back(NormalPrior(prior[i][0],prior[i][1]));
    return tmp;
  };
  GlmRWMHcond<GlmLogit,NormalPrior> LOG(Data(y,X,maxBinom),AlgoMCMC(N,nbAdaptMarche,warmup,invTemp),curr,SigMarch,lambdaPrior());

  RWMHCore(LOG);

  // arma::mat Res(LOG.data.pSpat,LOG.algo.N);
  // for(int k=0;k<LOG.data.pSpat;++k) Res.row(k) = arma::trans(LOG.regCoeff[k].res);
  // arma::vec sigRes(LOG.data.pSpat);
  // for(int k=0;k<LOG.data.pSpat;++k) sigRes(k) = LOG.regCoeff[k].sig;
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

// [[Rcpp::export]]
Rcpp::List logisticRWMHjoint(int N, int nbAdaptMarche, arma::vec y, arma::mat X, arma::vec maxBinom, arma::vec curr, arma::mat SigMarch, std::vector<arma::vec> prior,const int warmup=0,const double invTemp=1)
{
  auto lambdaPrior = [&prior,&X](){
    arma::mat tmp(prior[1]);
    return MultivariateNormalPrior(prior[0],tmp.resize(arma::size(X)(1),arma::size(X)(1)));
  };
  GlmRWMHjoint<GlmLogit,MultivariateNormalPrior> LOG(Data(y,X,maxBinom),AlgoMCMC(N,nbAdaptMarche,warmup,invTemp),curr,SigMarch,lambdaPrior());

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

