#ifndef MCMC_STRUCT
#define MCMC_STRUCT

#include <src/data_algo.hpp>
#include <armadillo>
#include <math.h>
#include <vector>
#include <iostream>
#include <thread>

////////////////////////// Prototypes and Templates ////////////////////////// 
template<class T>
const arma::Col<T> initRes(const T& curr,const int N){ 
	return arma::zeros<arma::Col<T>>(N); 
}
template<class T>
const arma::Mat<T> initRes(const arma::Col<T>& curr,const int N){   //res is either a vector or a matrix
	int nbValCurr = arma::size(curr)(0);
  if(nbValCurr <= 10000) return arma::zeros<arma::Mat<T>>(nbValCurr,N); //res is a matrix - need to restrict the size
  else return arma::zeros<arma::Mat<T>>(nbValCurr,1);                  //if there are two many variables to stock, we opt for stocking statistics like sum or mean values.
}
template<class T>
const std::vector<arma::Mat<T>> initRes(const arma::Mat<T>& curr,const int N){
	return std::vector<arma::Mat<T>>(N,arma::zeros<T>(arma::size(curr)));
}
template<class T>
const arma::Mat<T> initRes(const std::vector<arma::vec>& curr,const int N){
	int taille = 0;
	for(int i=0;i<curr.size();++i) taille += arma::size(curr[i])(0);
	return arma::zeros<T>(taille,N);
}
template<class T>
const std::vector<std::vector<arma::Mat<T>>> initRes(const std::vector<arma::mat>& curr,const int N){
	return std::vector<std::vector<arma::Mat<T>>>(curr.size(),std::vector<arma::Mat<T>>(N,arma::zeros<T>(arma::size(curr[0]))));
}

const arma::rowvec createRowVec(const int n);
template<class T>
const arma::vec createVec(const T& curr){ return arma::vec(arma::size(curr)(0),arma::fill::zeros); }
template<class T>
const arma::vec createMat(const T& curr){ 
  int nbValCurr = arma::size(curr)(0);
  return arma::mat(nbValCurr,nbValCurr,arma::fill::zeros); 
}
template<class T>
const arma::vec createCube(const T& curr){ 
  int nbValCurr = arma::size(curr)(0);
  return arma::cube(nbValCurr,nbValCurr,nbValCurr,arma::fill::zeros); 
}

////////////////////////// Classes ////////////////////////// 
template<class T,class U>
struct MCMC
{
	T curr;  //current value : could be either vec of double or vec of int (classes)
	U res;
  template<class AlgoTmp>
	explicit MCMC(const AlgoTmp& algo,const T& curr) :
		curr(curr),res(initRes(curr,algo.N)) {}
};

template<class T>
void GibbsCore(T& obj)
{
  for(int i=0;i<obj.algo.N;++i){
    std::cout<<i<<"\n";
    obj.algo.ite = i;  
    GibbsUpdate(obj);
  }
}

template<class T,class U,class V>
struct Gibbs : MCMC<T,U>
{
  V params;
  template<class AlgoTmp>
  explicit Gibbs(const AlgoTmp& algo,const T& curr,const V& params) :
  	MCMC<T,U>(algo,curr), params(params) {}
};

template<class T>
void RWMHCore(T& obj)
{
  //Adaptation step
  for(int i=0;i<obj.algo.nbAdaptMarche;i++){
    std::cout<<"Adaptation "<<i<<"\n";
    obj.algo.ite = 0;  
    adaptRWMHUpdate(obj);
    std::cout<<"\n";
  }
  //Sampling step
  for(int i=0;i<obj.algo.N;i++){
    std::cout<<i<<"\n";
    obj.algo.ite = i;  
    RWMHUpdate(obj);
  }
}

template<class T,class U,class V,class W,std::array<double,2>  bdsTmp,class PropTmp,class RatioTmp,class P>
struct RWMH : MCMC<T,U>
{
  T cand;  //candidate value : could be either vec of double or int (classes)
  V sig;  //random walk variance-covariance matrix : could be either a matrix or a single double
  P prior;
  T driftCurr; //random walk drift : for current value
  T driftCand; //                    for candidate value
  std::array<double,2> bounds = bdsTmp; 
  double prop = 0;  //acceptance rate
  double logRatio = 0; //Metropolis-Hastings ratio of acceptance
  W tempCurr; //stocks temporary values to be used throughout iterations
  W tempCand; //stocks temporary values to be used throughout iterations
  int accepted=0;
  // int numInd=0;

  template<class AlgoTmp>
  explicit RWMH(const AlgoTmp& algo,const T& curr,const V& SigMarch,const P& prior,const W& tempCurr,const W& tempCand) :
    MCMC<T,U>(algo,curr), cand(curr), sig(SigMarch), prior(prior), driftCurr(curr), driftCand(curr), tempCurr(tempCurr), tempCand(tempCand) {}
  
  inline void driftMH() noexcept{ PropTmp::callDrift(*this); }
  inline void modifDrift(const double& val) noexcept{ PropTmp::callModDrift(*this,val); }
  inline void proposalMH() noexcept{ PropTmp::callGen(*this); }
  inline void logRatioProposal() noexcept{ PropTmp::callLogRatioProposal(*this); }
  template<class DataTmp,class AlgoTmp,typename... Args>
  inline void logRatioLikelihood(const DataTmp& data,const AlgoTmp& algo,Args& ...args) noexcept{ PropTmp::callLogRatioLikelihood(*this,data,algo,args...); }
};

template<class T,class U,class V,class W,class PropTmp,class RatioTmp,class P>
struct MALA : RWMH<T,U,V,W,{0.56,0.58},PropTmp,RatioTmp,P>
{
  arma::mat invSig;
  arma::vec gradCurr;
  arma::vec gradCand;
  
  template<class AlgoTmp>
  explicit MALA(const AlgoTmp& algo,const T& curr,const V& SigMarch) :
    RWMH<T,U,V,W,{0.56,0.58},PropTmp,RatioTmp,P>(algo,curr,SigMarch),invSig(createMat(curr)),gradCurr(createVec(curr)),gradCand(reateVec(curr)) {}
};

template<class T,class U,class V,class W,class PropTmp,class RatioTmp,class P>
struct MMALA : MALA<T,U,V,W,PropTmp,RatioTmp,P>
{
  arma::rowvec g1;    //support, no need to track nor update, used inline
  arma::rowvec dg1;   //support, no need to track nor update, used inline
  arma::mat g2;       //support, no need to track nor update, used inline
  arma::mat Gcurr;
  arma::mat Gcand;
  arma::cube dGcurr;
  arma::cube dGcand;
  arma::mat invGcurr;
  arma::mat invGcand;
  
  //set the size of the object according to the size the matrix X and the number of iterations N
  template<class AlgoTmp>
  explicit MMALA(const Data& data,const AlgoTmp& algo,const T& curr,const V& SigMarch) :
    MALA<T,U,V,W,PropTmp,RatioTmp,P>(algo,curr,SigMarch),g1(createRowVec(data.n[0])),dg1(createRowVec(data.n[0])),Gcurr(createMat(curr)),
    Gcurr(createMat(curr)),Gcand(createMat(curr)),dGcurr(createCube(curr)),dGcand(createCube(curr)),
    invGcurr(createMat(curr)),invGcand(createMat(curr)),g2(createMat(curr)) {}
};

//////////////////////// functions using classes //////////////////////// 
struct ModifDriftMH
{
  template<class T>
  static inline void callModDrift(T& obj,const double& val){
    obj.sig = obj.sig*val;  
  }
};

struct CandDriftMH {
  template<class T>
  static inline void callDrift(T& obj){
    obj.driftCand = obj.cand;
  }
};

struct NormPropMH : CandDriftMH, ModifDriftMH
{
  template<class T>
  static inline void callGen(T& obj){
    obj.cand = arma::as_scalar( arma::randn(1, arma::distr_param(obj.driftCurr,obj.sig)) );
  }
  template<class T>
  static inline void callLogRatioProposal(T& obj){
    obj.logRatio += -0.5*( pow(obj.curr-obj.driftCand,2) - pow(obj.cand-obj.driftCurr,2) )/obj.sig; //IN RWMH driftCand=cand and driftCurr=curr
  }
};
struct MultNormPropMH : CandDriftMH, ModifDriftMH
{
  template<class T>
  static inline void callGen(T& obj){
    obj.cand = arma::mvnrnd(obj.driftCurr,obj.sig,1);
  }
  template<class T>
  static inline void callLogRatioProposal(T& obj){
    obj.logRatio += -0.5*arma::sum( arma::trans(obj.curr-obj.driftCand)*arma::inv(obj.sig)*(obj.curr-obj.driftCand) - arma::trans(obj.cand-obj.driftCurr)*arma::inv(obj.sig)*(obj.cand-obj.driftCurr) ); //IN RWMH driftCand=cand and driftCurr=curr
  }
};

template<class T>
void majMH(T& obj)  //RWMH
{
  obj.curr = obj.cand;
  obj.driftCurr = obj.driftCand;
  obj.prop += 1;//only used for the variance adaption phase 
  obj.tempCurr = obj.tempCand;
}

template<class DataTmp,class AlgoTmp,class T,typename... Args>
void ratioMH(const DataTmp& data,const AlgoTmp& algo,T& obj,Args& ...args){
  obj.logRatio = 0;   
  obj.logRatioLikelihood(data,algo,args...); //must be first to compute WBIC
  obj.prior.logRatioDensity(obj);
  obj.logRatioProposal();
}

template<class DataTmp,class AlgoTmp,class T,typename... Args>
void MHstep(const DataTmp& data,const AlgoTmp& algo,T& obj,Args& ...args){
  obj.proposalMH();
  obj.driftMH();
  ratioMH(data,algo,obj,args...); 
  double tirage = arma::randu();
  if(obj.logRatio > log(tirage)){
    obj.accepted = 1;
    majMH(obj);
  }
  else obj.accepted = 0;
}

template<class AlgoTmp,class T>
void adaptationStep(const AlgoTmp& algo,T& obj,const int verbose=1){
  obj.prop /= algo.tau;
  if(obj.prop > obj.bounds[1]) obj.modifDrift(1.2); 
  else if(obj.prop < obj.bounds[0])  obj.modifDrift(0.8);  
  else obj.modifDrift(1.0);   //in most case: does nothing, for mala: obj.invSig = arma::inv(obj.sig); 
  if(verbose==1) std::cout<<"prop "<<obj.prop<<" ";
  obj.prop = 0;
}

#endif