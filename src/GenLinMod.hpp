#ifndef GENLINMOD
#define GENLINMOD

#include "src/MCMC_struct.hpp"
#include <RcppArmadillo.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <type_traits>

//////////// model specification //////////// 
struct GlmUtil{
  template<class U,class V>
  const static inline arma::vec callEta(const U& X,const V& coeff,int v=0) { return X*coeff; }
};
struct GlmPoisson : GlmUtil 
{
  template<class U,class V>
  const static inline arma::vec callTempCompute(const U& X,const V& coeff,int v=0) noexcept { 
    return exp( GlmUtil::callEta(X,coeff) ); 
  }
  const static inline double callSumLikelihood(const arma::mat& X,const arma::vec& y,const arma::vec& maxBinom,const arma::vec& candMinusCour,const arma::vec& tempCand,const arma::vec& tempCurr){
    return arma::sum(X*candMinusCour%y - tempCand + tempCurr );
  }
};
struct GlmLogit : GlmUtil
{
  template<class U,class V>
  const static inline arma::vec callTempCompute(const U& X,const V& coeff,int v=0) noexcept { 
    return log1p(exp( GlmUtil::callEta(X,coeff) ) ); 
  }
  template<class xTmp,class coeffTmp>
  const static inline double callSumLikelihood(const xTmp& X,const arma::vec& y,const arma::vec& maxBinom,const coeffTmp& candMinusCour,const arma::vec& tempCand,const arma::vec& tempCurr){
    return arma::sum( (X*candMinusCour)%y - maxBinom%( tempCand - tempCurr ) );
  }
};
//////////// Utils //////////// 
template<typename T>
struct element_type {
  using type = T;
};
template<typename U, typename Alloc>
struct element_type<std::vector<U, Alloc>> {
  using type = U;
};
template<typename T>
using vecU_or_U = typename element_type<T>::type;
//////////////////////////////////////////////  

template<class DataTmp,class AlgoTmp>
struct BaseMCMC : Base<DataTmp,AlgoTmp>
{
  void WAICinit();
  
  explicit BaseMCMC(const DataTmp& _data,const AlgoTmp& _algo) : Base<DataTmp,AlgoTmp>(_data,_algo){
    WAICinit();
  }
  
  // void setParamsXiGibbs(const arma::mat& X,const CoeffGibbs& regCoeff,XiGibbs& xi) { xi.params = X*regCoeff.curr; }
};
template<class DataTmp,class AlgoTmp>
void BaseMCMC<DataTmp,AlgoTmp>::WAICinit(){
  for(int i=0;i<this->algo.logDens.size();++i){
    this->algo.logDens[i].zeros(this->data.n[0]);
    this->algo.logDensCarre[i].zeros(this->data.n[0]);
    this->algo.dens[i].zeros(this->data.n[0]);
  }
}

template<class T,class U,class V,class W,std::array<double,2>  bdsTmp,class PropTmp,class RatioTmp,class P>
struct GlmRWMH : RWMH<T,U,V,W,bdsTmp,PropTmp,RatioTmp,P>
{
  template<class xTmp,class CoeffTmp>
  const inline arma::vec eta(const xTmp& X,const CoeffTmp& coeff,int v=0) noexcept { return RatioTmp::callEta(X,coeff); }
  template<class xTmp,class CoeffTmp>
  const inline arma::vec tempCompute(const xTmp& X,const CoeffTmp& coeff) noexcept { return RatioTmp::callTempCompute(X,coeff); }
  
  using RWMH<T,U,V,W,bdsTmp,PropTmp,RatioTmp,P>::RWMH;
};

template<class RatioTmp,class P>
struct CoeffGlmRWMHcond : GlmRWMH<double,arma::vec,double,std::vector<arma::vec>,{0.43,0.45},NormPropMH,RatioTmp,P>
{
  int numCoeff=0;
  
  template<class DataTmp,class AlgoTmp,typename... Args>
  inline void logRatioLikelihood(const DataTmp& data,const AlgoTmp& algo,Args& ...args){  
    auto firstCoeff = std::forward_as_tuple(args...);
    std::get<0>(firstCoeff).tempCurr[1](this->numCoeff) = this->cand;
    std::get<0>(firstCoeff).tempCand[0] = this->tempCompute(data.X,std::get<0>(firstCoeff).tempCurr[1]);
    this->logRatio += RatioTmp::callSumLikelihood(data.X.col(numCoeff),data.y,data.params,this->cand-this->curr,std::get<0>(firstCoeff).tempCand[0],std::get<0>(firstCoeff).tempCurr[0]);
    this->logRatio *= algo.invTemp; // for WBIC if invTemp != 1
  }
  using GlmRWMH<double,arma::vec,double,std::vector<arma::vec>,{0.43,0.45},NormPropMH,RatioTmp,P>::GlmRWMH;
};
template<class RatioTmp,class P>
struct CoeffGlmRWMHjoint : GlmRWMH<arma::vec,arma::mat,arma::mat,arma::vec,{0.22,0.24},MultNormPropMH,RatioTmp,P>
{
  template<class DataTmp,class AlgoTmp,typename... Args>
  inline void logRatioLikelihood(const DataTmp& data,const AlgoTmp& algo,Args& ...args){  
    auto candMinusCour = this->cand-this->curr;
    this->tempCand = this->tempCompute(data.X,this->cand);
    this->logRatio += RatioTmp::callSumLikelihood(data.X,data.y,data.params,candMinusCour,this->tempCand,this->tempCurr);
    this->logRatio *= algo.invTemp; // for WBIC if invTemp != 1
  }
  using GlmRWMH<arma::vec,arma::mat,arma::mat,arma::vec,{0.22,0.24},MultNormPropMH,RatioTmp,P>::GlmRWMH;
};

template<class T,class BaseTemp>
struct GlmMH : BaseTemp
{
  using U = vecU_or_U<T>;
  
  T regCoeff;

  template<class P>
  explicit GlmMH(const Data& _data,const AlgoMCMC& _algo,const std::vector<double>& _curr,const std::vector<double>& _SigMarch,const std::vector<P>& _prior) :
    BaseTemp(_data,_algo), 
    regCoeff(T([&_data,&_algo,&_curr,&_SigMarch,&_prior](){
      T tmp;
      tmp.reserve(_curr.size());
      for(int i=0;i<_curr.size();++i){
        if(i==0) tmp.push_back(U(_algo,_curr[i],_SigMarch[i],_prior[i],{arma::vec(_data.n[0]),arma::vec(_data.p[0])},{arma::vec(_data.n[0]),arma::vec()}));
        else tmp.push_back(U(_algo,_curr[i],_SigMarch[i],_prior[i],std::vector<arma::vec>(),std::vector<arma::vec>()));
      }
      return tmp;
    }()))
  {
    // regCoeff.resize(BaseTemp::data.p[0]);
    for(int k=0;k<BaseTemp::data.p[0];++k) regCoeff[k].numCoeff = k;
    initializeTemporaryrGlmMH(regCoeff[0],regCoeff[0].tempCurr[0],regCoeff[0].tempCand[0],BaseTemp::data.X);
    regCoeff[0].tempCurr[0] = [&_data,*this](){
      arma::vec tmp(_data.p[0]);
      for(int i=0;i<_data.p[0];++i) tmp = regCoeff[i].curr;
      return tmp;
    }();
  }
  template<class P>
  GlmMH(const Data& data,const AlgoMCMC& algo,const arma::vec& curr,const arma::mat& SigMarch,const P& prior) :
    BaseTemp(data,algo), regCoeff(T(algo,curr,SigMarch,prior,arma::vec(data.n[0]),arma::vec(data.n[0])))
  {
    initializeTemporaryrGlmMH(regCoeff,regCoeff.tempCurr,regCoeff.tempCand,BaseTemp::data.X);
  }

  template<class Obj,class V,class W>
  void initializeTemporaryrGlmMH(Obj& obj,V& tempCurr,V& tempCand,const W& X)
  {
    tempCand.set_size(BaseTemp::data.n[0]);
    tempCurr.set_size(BaseTemp::data.n[0]);
    tempCurr = obj.tempCompute(X,obj.curr);
  }
};

template<class RatioTmp,class P>
using GlmRWMHcond = GlmMH<std::vector<CoeffGlmRWMHcond<RatioTmp,P>>,BaseMCMC<Data,AlgoMCMC>>;
template<class RatioTmp,class P>
using GlmRWMHjoint = GlmMH<CoeffGlmRWMHjoint<RatioTmp,P>,BaseMCMC<Data,AlgoMCMC>>;

template<class T>
void adaptRWMHUpdate(T& obj)   //for RWMHjoint, MALA, MMALA
{
  for(int t=0;t<obj.algo.tau;++t) MHstep(obj.data,obj.algo,obj.regCoeff);
  adaptationStep(obj.algo,obj.regCoeff);
} 
template<class T>
void RWMHUpdate(T& obj) //for RWMHjoint, MALA, MMALA, Poisson, Binom
{
  MHstep(obj.data,obj.algo,obj.regCoeff);
  obj.regCoeff.res.col(obj.algo.ite) = obj.regCoeff.curr;
  quantititiesForCriteria(obj.data,obj.regCoeff.curr,obj.regCoeff.tempCurr,obj.regCoeff,obj.algo);
}

template<class RatioTmp,class P>
void adaptRWMHUpdate(GlmRWMHcond<RatioTmp,P>& obj)
{
  for(int t=0;t<obj.algo.tau;++t){
    for(int k=0;k<obj.data.p[0];++k){
      MHstep(obj.data,obj.algo,obj.regCoeff[k],obj.regCoeff[0]);
      if(obj.regCoeff[k].accepted==1) obj.regCoeff[0].tempCurr[0] = obj.regCoeff[k].tempCand[0];
      else obj.regCoeff[0].tempCurr[1](obj.regCoeff[k].numCoeff) = obj.regCoeff[k].curr;
    }
  }
  for(int k=0;k<obj.data.p[0];++k) adaptationStep(obj.algo,obj.regCoeff[k]);
} 
template<class RatioTmp,class P>
void RWMHUpdate(GlmRWMHcond<RatioTmp,P>& obj)
{
  for(int k=0;k<obj.data.p[0];++k){
    MHstep(obj.data,obj.algo,obj.regCoeff[k],obj.regCoeff[0]);
    if(obj.regCoeff[k].accepted==1) obj.regCoeff[0].tempCurr[0] = obj.regCoeff[k].tempCand[0];
    else obj.regCoeff[0].tempCurr[1](obj.regCoeff[k].numCoeff) = obj.regCoeff[k].curr;
    obj.regCoeff[k].res(obj.algo.ite) = obj.regCoeff[k].curr;
  }
  quantititiesForCriteria(obj.data,obj.regCoeff[0].curr,obj.regCoeff[0].tempCurr[0],obj.regCoeff[0],obj.algo);
}

const arma::vec likelihoodIndiv(const arma::vec& XprodCour,const arma::vec& y,const arma::vec& maxBinom,const arma::vec& tempCurr);

template<class DataTmp,class T,class U,class V,class AlgoTmp> 
void quantititiesForCriteria(const DataTmp& data,const T& curr,const U& tempCurr,V& obj,AlgoTmp& algo)
{
  if(algo.ite >= algo.warmup){
    arma::vec logDensTemp = likelihoodIndiv(obj.eta(data.X,curr),data.y,data.params,tempCurr);
    int indice = (int)(algo.ite-algo.warmup) / 10000      ;
    algo.logDens[indice]      += logDensTemp              ;
    algo.logDensCarre[indice] += logDensTemp % logDensTemp;
    algo.dens[indice]         += exp(logDensTemp)         ;
    algo.S[indice]            += 1                        ;
  }
}


#endif