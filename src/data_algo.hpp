#ifndef DATA_ALGO
#define DATA_ALGO

#include <armadillo>
#include <vector>
#include <iostream>
#include <thread>

//////////////////////////////////// Types ///////////////////////////////////
using voxVec = std::vector<arma::vec>;
using voxMat = std::vector<arma::mat>;

////////////////////////// Prototypes and Templates ////////////////////////// 
template<class T>
const int genericSize(const std::vector<T>& vec){ return vec.size(); }
template<class T>
const int genericSize(const arma::Col<T>& vec){ return arma::size(vec)(0); }
template<class T>
const int genericSize(const arma::Mat<T>& mat,const int index=0){ return arma::size(mat)(index); }
template<class T>
const int genericSize(const std::vector<arma::Mat<T>>& vecMat,const int index=0){ return arma::size(vecMat[0])(index); }

/////////////////////////////////// Classes ////////////////////////////////// 
template<class DataTmp,class AlgoTmp>
struct Base
{
  DataTmp data;
  AlgoTmp algo;
  
  explicit Base(const DataTmp& _data,const AlgoTmp& _algo) : data(_data), algo(_algo) {}
};

///////////////////////////////////// Data /////////////////////////////////// 
template<class T,class U,class V>
struct Data_template
{                   
  const T y;                // repsonse variable      
  const U X;                // covariates    
  const std::vector<int> n; // number of individual times 
  const std::vector<int> p; // number of variables     
  const V params;           // ex. max value binomial     

  explicit Data_template(T y=T(),U X=U(),std::vector<int> n=std::vector<int>(),std::vector<int> p=std::vector<int>(),V params=V()) : 
    y(y),X(X),n(n),p(p),params(params) {}
};

struct Data : Data_template<arma::vec,arma::mat,arma::vec>
{
  explicit Data(const arma::vec& y=arma::vec(),const arma::mat& X=arma::mat(),const arma::vec& params=arma::vec()) : 
    Data_template(y,X,std::vector<int>(1,genericSize(y)),std::vector<int>(1,genericSize(X,1)),params) {} 
};

struct DataVox : Data_template<voxVec,voxMat,voxVec>
{
  const int V = 2; // number of voxels
  
  explicit DataVox(const voxVec& y=voxVec(),const voxMat& X=voxMat(),const std::vector<int>& p=std::vector<int>(),const voxVec& params=voxVec()) : 
    Data_template(y,X,std::vector<int>(1,genericSize(y[0])),p,params), V(genericSize(y)) {}
};

///////////////////////////////////// Algo /////////////////////////////////// 
struct Algo_template
{
  const int N = 1;              //algo iteration number
  int ite = 0; //to keep trace of iterations
  const unsigned int nbThreads = std::thread::hardware_concurrency()-2;
  
  explicit Algo_template(const int N) : N(N) {
    std::cout<<"Number of threads: "<<this->nbThreads<<"\n";
  }
};

template<class T>
struct Algo_VBEM : Algo_template
{
  double stopCriterion = 0;
  double previousStopCriterion = 0;
};

template<class T>
struct Algo_MCMC_template : Algo_template
{
  const int nbAdaptMarche;  //MCMC number of adaptation iterations
  const double warmup;      //warmup iteration number
  const double invTemp = 0;
  const int tau;  //how many time MH is repeated with the same variance to calculate the ratio of acceptation
  std::vector<T> logDens;
  std::vector<T> logDensCarre;
  std::vector<T> dens;
  std::vector<double> S;
  
  explicit Algo_MCMC_template(const int N,const int nbAdaptMarche=1,const double& warmup=0,const double& invTemp = 1,const int tau = 100) : 
    Algo_template(N),nbAdaptMarche(nbAdaptMarche),warmup(warmup),invTemp(invTemp),tau(tau) 
  {
    int nbIndice = (int)(N-warmup) / 10000;
    if(nbIndice == 0) nbIndice = 1;
    this->logDens.resize(nbIndice);
    this->logDensCarre.resize(nbIndice);
    this->dens.resize(nbIndice);
    this->S.resize(nbIndice);
  }
};

using AlgoMCMC = Algo_MCMC_template<arma::vec>;
using AlgoMCMCVox = Algo_MCMC_template<std::vector<arma::vec>>;
using AlgoMCMCIndVox = Algo_MCMC_template<std::vector<std::vector<arma::vec>>>;

///////////////////////////////////// Priors ///////////////////////////////////
struct VecMat
{
  const arma::vec vec;
  const arma::mat mat;
  explicit VecMat(const arma::vec& vec) : vec(vec), mat(arma::mat()) {}
  explicit VecMat(const arma::mat& mat) : vec(arma::vec()), mat(mat) {}
};

template<class T>
struct Prior
{
  const T params;
  
  explicit Prior(const T& params) : params(params) {}
};

struct NormalPrior : Prior<std::vector<double>>
{
  explicit NormalPrior(const double& mu, const double& sig2) : Prior({mu,sig2,1/sig2}) {}
  template<class T>
  void logRatioDensity(T& obj){
    obj.logRatio += -0.5*params[2]*( pow(obj.cand,2)-pow(obj.curr,2) + 2*params[0]*(obj.curr-obj.cand) )  ;
  }
};

struct MultivariateNormalPrior : Prior<std::vector<VecMat>>
{
  explicit MultivariateNormalPrior(const arma::vec& mu, const arma::mat& Sig2) : Prior({VecMat(mu),VecMat(Sig2),VecMat(arma::mat(arma::inv(Sig2)))}) // inv(Sig2) is not of class mat, so the mat constructor of VecMat cannot directly be called, a mat object can be constructed with the result of inv(Sig2)
  {
    if(arma::size(params[0].vec)(0) != arma::size(params[1].mat)(0))throw std::invalid_argument("Prior: incompatible matrix dimensions: "+std::to_string(arma::size(params[0].vec)(0))+"x"+std::to_string(arma::size(params[0].vec)(1))+" and "+std::to_string(arma::size(params[1].mat)(0))+"x"+std::to_string(arma::size(params[1].mat)(1)));
  }
  template<class T>
  void logRatioDensity(T& obj){
    obj.logRatio += arma::as_scalar( -0.5*( arma::trans(obj.cand-params[0].vec)*params[2].mat*(obj.cand-params[0].vec) - arma::trans(obj.curr-params[0].vec)*params[2].mat*(obj.curr-params[0].vec) ) );
  }
};

struct SymDirichletPrior : Prior<double>
{
  explicit SymDirichletPrior(const double& alpha) : Prior({alpha}) {}
};


#endif