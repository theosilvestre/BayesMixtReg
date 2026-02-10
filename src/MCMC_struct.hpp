#ifndef MCMC_STRUCT
#define MCMC_STRUCT

#include <data_algo.hpp>
#include <armadillo>
#include <math.h>
#include <vector>
#include <iostream>
#include <thread>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

////////////////////////// Prototypes and Templates ////////////////////////// 
template<class T>
const arma::Col<T> initRes(const T& curr,const int N){ 
	return arma::zeros<T>(N); 
}
template<class T>
const arma::Mat<T> initRes(const arma::Col<T>& curr,const int N){   //res is either a vector or a matrix
	int nbValCurr = arma::size(curr)(0);
  if(nbValCurr <= 10000) return arma::zeros<T>(nbValCurr,N); //res is a matrix - need to restrict the size
  else return arma::zeros<T>(nbValCurr,1);                  //if there are two many variables to stock, we opt for stocking statistics like sum or mean values.
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
  template<class AlgoTemplate>
	explicit MCMC(const AlgoTemplate& algo,const T& curr) :
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
  template<class AlgoTemplate>
  explicit Gibbs(const AlgoTemplate& algo,const T& curr,const V& params) :
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

template<class T,class U,class V,class W>
struct RWMH : MCMC<T,U>
{
  T cand;  //candidate value : could be either vec of double or int (classes)
  V sig;  //random walk variance-covariance matrix : could be either a matrix or a single double
  T driftCurr; //random walk drift : for current value
  T driftCand; //                    for candidate value
  std::vector<double> bounds = {0.22,0.24}; //acceptation-rejection boundaries
  double prop = 0;  //acceptance rate
  double ratio = 0; //Metropolis-Hastings ratio of acceptance
  W tempCurr; //stocks temporary values to be used throughout iterations
  W tempCand; //stocks temporary values to be used throughout iterations
  int accepted=0;
  // int numCoeff=0;
  // int numInd=0;

  template<class DataTemp,class AlgoTemplate>
  explicit RWMH(const DataTemp& data,const AlgoTemplate& algo,const T& curr,const V& SigMarche) :
    MCMC<T,U>(algo,curr), cand(curr), sig(SigMarche), driftCurr(curr), driftCand(curr) {}
};

template<class T,class U,class V,class W>
struct MALA : RWMH<T,U,V,W>
{
  arma::mat invSig;
  arma::vec gradCurr;
  arma::vec gradCand;
  std::vector<double> bounds = {0.56,0.58};
  
  template<class AlgoTemplate>
  explicit MALA(const Data& data,const AlgoTemplate& algo,const T& curr,const V& SigMarche) :
    RWMH<T,U,V,W>(data,algo,curr,SigMarche),invSig(createMat(curr)),gradCurr(createVec(curr)),gradCand(reateVec(curr)) {}
};

template<class T,class U,class V,class W>
struct MMALA : MALA<T,U,V,W>
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
  template<class AlgoTemplate>
  explicit MMALA(const Data& data,const AlgoTemplate& algo,const T& curr,const V& SigMarche) :
    MALA<T,U,V,W>(data,algo,curr,SigMarche),g1(createRowVec(data.n[0])),dg1(createRowVec(data.n[0])),Gcurr(createMat(curr)),
    Gcurr(createMat(curr)),Gcand(createMat(curr)),dGcurr(createCube(curr)),dGcand(createCube(curr)),
    invGcurr(createMat(curr)),invGcand(createMat(curr)),g2(createMat(curr)) {}
};

//////////////////////// functions using classes //////////////////////// 
template<class T,class U,class V,class W,class X>
void driftMH(T& objGen,RWMH<U,V,W,X>& obj){
  obj.driftCand = obj.cand;
}

template<class T>
void majGrad(T& obj) {} 

template<class T,class U,class V,class W>
void majGrad(struct MALA<T,U,V,W>& obj){ 
  obj.gradCurr = obj.gradCand;
}

template<class T>
void majMH(T& obj){         //RWMH
  obj.curr = obj.cand;
  obj.driftCurr = obj.driftCand;
  majGrad(obj);
  obj.prop += 1;//only used duriang the sig adaption phase 
  obj.tempCurr = obj.tempCand;
}

template<class T,class U>
void MHstep(T& objGen,U& obj){
  proposalMH(obj);
  driftMH(objGen,obj);
  ratioMH(objGen,obj); //rapport des proposals egal à 1
  double tirage = arma::randu();
  if(obj.ratio > log(tirage)){
    obj.accepted = 1;
    majMH(obj);
  }
  else obj.accepted = 0;
}

template<class T,class U,class V,class W>
void modifDrift(struct RWMH<T,U,V,W>& obj,const double& val){
  obj.sig = obj.sig*val;  
};

template<class T,class U,class V,class W>
void modifDrift(struct MALA<T,U,V,W>& obj,const double& val){
  obj.sig = obj.sig*val; 
  obj.driftCurr = (obj.driftCurr - obj.curr)*val + obj.curr; //sig is changed to the current value
  obj.driftCurr = obj.curr + 0.5*obj.sig(0,0)*obj.gradCurr;
  obj.invSig = arma::inv(obj.sig);
}

template<class T,class U>
void adaptationStep(T& objGen,U& obj,const int verbose=1){
  obj.prop /= objGen.algo.tau;
  if(obj.prop > obj.bounds[1]) modifDrift(obj,1.2); 
  else if(obj.prop < obj.bounds[0])  modifDrift(obj,0.8);  
  else modifDrift(obj,1.0);   //in most case: does nothing, for mala: obj.invSig = arma::inv(obj.sig); 
  if(verbose==1) std::cout<<"prop "<<obj.prop<<" ";
  obj.prop = 0;
}

#endif