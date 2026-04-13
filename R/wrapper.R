

#Reconstructed non available function from coda
".safespec0" <- function (x) {
  result <- try(spectrum0.ar(x)$spec)
  ## R
  if (class(result) == "try-error") result <- NA
  ## S-Plus
  if (class(result) == "try") result <- NA
  result
}

.concatPlot <- function(N,var,minChain,maxChain,chains){
  plot(1:N,chains[[1]][var,],type="l",col=1,xlab="Iterations",ylab="",ylim=c(minChain,maxChain))
  if(length(chains)>1){
    for(i in 2:length(chains)) points(1:N,chains[[i]][var,],type="l",col=i,xlab="Iterations",ylab="")
  }
}

.concatDensityPlot <- function(var,chains){
  maxDense <-  0
  for(i in 1:length(chains)){
    cont <- max(density(chains[[i]][var,])[["y"]])
    if(cont > maxDense) maxDense <- cont
  }
  plot(density(chains[[1]][var,]),type="l",col=1,xlab="",ylab="",main="",ylim=c(0,maxDense))
  if(length(chains)>1){
    for(i in 2:length(chains)) points(density(chains[[i]][var,]),type="l",col=i)
  }
}

.summaryChains <- function(warmup,method,...,display=FALSE,thin=0){
  chainsList <- list( ... )
  N <- ncol(chainsList[[1]][[1]])
  if(is.null(N)) N <- length(chainsList[[1]][[1]])
  if(warmup >= N) stop("Warmup greater than sample number.")
  nbVar <- nrow(chainsList[[1]][[1]])
  if(is.null(nbVar)) nbVar <- 1
  nbChain <- length(chainsList[[1]])
  chains <- list()
  if(is.matrix(chainsList[[1]][[1]])){
    for(i in 1:nbChain) chains[[i]] <- chainsList[[1]][[i]][,(warmup+1):N]
  } else{
    for(i in 1:nbChain) chains[[i]] <- matrix(chainsList[[1]][[i]][(warmup+1):N],nrow=1)
  }
  if(thin > 0) for(i in 1:nbChain) chains[[i]] <- chains[[i]][,seq(1,length(chains[[i]][1,]),thin)]
  concatChains <- as.matrix(as.data.frame( chains ))
  nbIter <- ncol(chains[[1]])
  
  moy <- apply(concatChains, 1, mean)
  standDev <- apply(concatChains, 1, sd)
  se <- standDev / sqrt(nbIter*nbChain)
  quant <- apply(concatChains, 1, function(x) quantile(x,c(0.025,0.5,0.975)))
  ess <- apply(concatChains,1,coda::effectiveSize)
  tsSe <- rep(0,nbVar)
  for(i in 1:nbChain) tsSe <- tsSe + apply(chains[[i]], 1, safespec0)
  tsSe <- sqrt(tsSe/(nbChain**2*nbIter))
  resSummary <- rbind(rep(nbChain,nbVar),rep(nbIter,nbVar),moy,standDev,se,tsSe,quant,ess)
  rownames(resSummary) <- c("nbChain","nbIter","mean","sd","se","time-series se","2.5%","50%","97.5%","ESS")
  # if(nbChain==1) if(method=="RWMH" || method=="MALA" || method=="MMALA" || method=="ULA") resSummary <- rbind(resSummary,"sigMarche"=diag(chainsList[[1]][[2]]))
  if(display){
    par(mfrow=c(2,nbVar))
    for(i in 1:nbVar) concatPlot(nbIter,i,min(concatChains[i,],na.rm=TRUE),max(concatChains[i,],na.rm=TRUE),chains)
    for(i in 1:nbVar) concatDensityPlot(i,chains)
    par(mfrow=c(nbChain,nbVar))
    for(i in 1:nbVar) for(j in 1:nbChain) acf(chains[[j]][i,],main="")
  }
  # colnames(resSummary) <- c("coeff.","(Intercept)")
  colnames(resSummary) <- paste0("var",1:nbVar)
  
  return( resSummary )
}

########## Logistic ##########

#' Logistic binomial regression / joint inference with RWMH method 
#'
#' @description This function performs a logistic binomial regression using the RWMH method to jointly infer the coefficients.
#'
#' @param N the number of iterations post-adaptative step
#' @param nbAdaptMarche the number of adaptation steps, one step equals 100 iterations
#' @param y a vector of count values, from 0 to maxBinom[i] for y[i], i in 1...length(y)
#' @param X a matrix of covariates, with dim(X)[1] = length(y)
#' @param maxBinom a vector of maximum trials (in binomial fashion), length(maxBinom) = length(y)
#' @param curr a vector of initialization values corresponding to each of the covariates in the columns of X, ordered in the same way
#' @param SigMarch a covariance matrix for the Gaussian proposal distribution
#' @param prior a list of vectors for the priors' hyperparameters
#' @param warmup the number of iterations reserved to warmup (warmup >= 0 and warmup <= N)
#' @param invTemp a value to temper the likelihood
#' @return a list that includes a matrix of each coefficient posterior sample, the adapted covariance matrix for the Gaussian proposal distribution and some values to compute WAIC
#' 
#' @details ...
#' 
#' @examples ...
#' 
logisticRegression <- function(form,data,maxBinom=NULL,method="freq",curr=c(),prior=c(),N=5000,nbAdaptMarche=100,warmup=0,epsilon=10e-8,SigMarch=1,invTemp=1)
{
  data <- as.data.frame(data)
  outcome <- all.vars(form)[attr(terms(form),"response")]
  income <- all.vars(form)[-attr(terms(form),"response")]
  nbVar <- length(income) 
  y <- data[[outcome]]
  
  if(method=="freq"){
    if(attr(terms(form),"intercept")==1) nbVar <- nbVar + 1
    res <- glm(formula = form,data = data,family = binomial(link = "logit"))
    
    n <- nrow(data)
    statFreq <- summary(res)
    resSummary <- matrix(NA,ncol = nbVar,nrow=3)
    rownames(resSummary) <- c("2.5%","50%","97.5%")
    NAorNot <- !as.vector(statFreq$aliased)
    for(i in 1:nbVar){
      if(NAorNot[i]) try(resSummary[,i] <- statFreq$coefficients[i,1] + c(-1,0,1)*qnorm(0.975)*statFreq$coefficients[i,2],silent=TRUE)
      else resSummary[,i] <- NA
    }
    if(attr(terms(form),"intercept")==1) colnames(resSummary) <- c("(Intercept)",income)
    else colnames(resSummary) <- c(income)
    return(resSummary)
  }
  else{
    if(attr(terms(form),"intercept")==1){
      X <- as.matrix(data.frame("(Intercept)"=1,data[income]))
      nbVar <- nbVar + 1 
    }
    else X <- as.matrix(data.frame(data[income]))
    if(method=="RWMH"){
      if(attr(terms(form),"intercept")==1 & length(income) == length(curr)) curr <- c(rnorm(1),curr)
      res <- .logisticRWMHjoint(N, nbAdaptMarche, y, X, maxBinom, curr, SigMarch, prior, warmup, invTemp)
    }
    return(res)
  }
}

