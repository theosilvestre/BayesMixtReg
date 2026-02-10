#ifndef DATA_ALGO
#define DATA_ALGO

#include <RcppArmadillo.h>
#include <vector>
#include <iostream>
#include <thread>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

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
template<class DataTemplate,class AlgoTemplate,class ModelTemplate>
struct Base
{
  DataTemplate data;
  AlgoTemplate algo;
  ModelTemplate model;
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

/*struct DataDose : Data_template<double,arma::vec,double>
{
  const int V;
  
  explicit DataDose(const arma::vec& X) : Data_template(X,genericSize(X)); 
};
*/
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
  const int ite = 0; //to keep trace of iterations
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
  T logDens;
  T logDensCarre;
  T dens;
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

using AlgoMCMC = Algo_MCMC_template<std::vector<arma::vec>>;
using AlgoMCMCVox = Algo_MCMC_template<std::vector<std::vector<arma::vec>>>;
using AlgoMCMCIndVox = Algo_MCMC_template<std::vector<std::vector<std::vector<arma::vec>>>>;


#endif