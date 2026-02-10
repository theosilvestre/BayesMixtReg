#ifndef EM_STRUCT
#define EM_STRUCT

#include "MCMC_struct.hpp"
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <thread>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

//////////////////////////  Classes  ////////////////////////// 
template<class V,class W>
struct Expectation
{
  V params;
  W moments; 
  explicit Expectation(const V& params,const W& moments):
  	params(params), moments(moments) {}
};

template<class T>
struct Maximisation
{
  T cour;  
  explicit Maximisation(const T& cour):
  	cour(cour) {}
};

////////////////////////// VBEM ////////////////////////// 

template<class T>
void VEMcore(T& obj,double& epsilon)
{
  int compt = 0;
  obj.algo.previousStopCriterion = epsilon + 1;
  
  while( (abs((obj.algo.stopCriterion - obj.algo.previousStopCriterion)/obj.algo.previousStopCriterion) > epsilon) && (compt<obj.algo.N) ){
    std::cout<<"iteration "<<compt<<" crit: "<<obj.algo.stopCriterion<<" weighted diff: "<<abs((obj.algo.stopCriterion - obj.algo.previousStopCriterion)/obj.algo.previousStopCriterion)<<"\n";
    obj.algo.previousStopCriterion = obj.algo.stopCriterion;
    obj.algo.stopCriterion = 0;
    
    VEMupdate(obj);
    ELBO(obj);
    
    compt++;
  }
  if(compt==obj.algo.N || isnan(obj.algo.stopCriterion)) std::cout<<"Limit has not been reached."<<"\n";
  else std::cout<<"Limit reached in "<<compt<<" iterations"<<" crit: "<<obj.algo.stopCriterion <<" weighted diff: "<<abs((obj.algo.stopCriterion - obj.algo.previousStopCriterion)/obj.algo.previousStopCriterion)<<"\n";
}

#endif