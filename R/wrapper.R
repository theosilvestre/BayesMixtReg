

#Reconstructed non available function from coda
".safespec0" <- function (x) {
  result <- try(coda::spectrum0.ar(x)$spec)
  ## R
  if (class(result) == "try-error") result <- NA
  ## S-Plus
  if (class(result) == "try") result <- NA
  result
}

.concatPlot <- function(N,var,minChain,maxChain,chains){
  plot(1:N,chains[[1]][var,],type="l",col=1,xlab="Iterations",ylab="",ylim=c(minChain,maxChain))
  if(length(chains)>1){
    for(i in 2:length(chains)) graphics::points(1:N,chains[[i]][var,],type="l",col=i,xlab="Iterations",ylab="")
  }
}

.concatDensityPlot <- function(var,chains){
  maxDense <-  0
  for(i in 1:length(chains)){
    cont <- max(stats::density(chains[[i]][var,])[["y"]])
    if(cont > maxDense) maxDense <- cont
  }
  plot(stats::density(chains[[1]][var,]),type="l",col=1,xlab="",ylab="",main="",ylim=c(0,maxDense))
  if(length(chains)>1){
    for(i in 2:length(chains)) graphics::points(stats::density(chains[[i]][var,]),type="l",col=i)
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
  standDev <- apply(concatChains, 1, stats::sd)
  se <- standDev / sqrt(nbIter*nbChain)
  quant <- apply(concatChains, 1, function(x) stats::quantile(x,c(0.025,0.5,0.975)))
  ess <- apply(concatChains,1,coda::effectiveSize)
  tsSe <- rep(0,nbVar)
  for(i in 1:nbChain) tsSe <- tsSe + apply(chains[[i]], 1, .safespec0)
  tsSe <- sqrt(tsSe/(nbChain**2*nbIter))
  resSummary <- rbind(rep(nbChain,nbVar),rep(nbIter,nbVar),moy,standDev,se,tsSe,quant,ess)
  rownames(resSummary) <- c("nbChain","nbIter","mean","sd","se","time-series se","2.5%","50%","97.5%","ESS")
  # if(nbChain==1) if(method=="RWMH" || method=="MALA" || method=="MMALA" || method=="ULA") resSummary <- rbind(resSummary,"sigMarche"=diag(chainsList[[1]][[2]]))
  if(display){
    graphics::par(mfrow=c(2,nbVar))
    for(i in 1:nbVar) .concatPlot(nbIter,i,min(concatChains[i,],na.rm=TRUE),max(concatChains[i,],na.rm=TRUE),chains)
    for(i in 1:nbVar) .concatDensityPlot(i,chains)
    graphics::par(mfrow=c(nbChain,nbVar))
    for(i in 1:nbVar) for(j in 1:nbChain) stats::acf(chains[[j]][i,],main="")
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
#' @param form an object of class "formula" (as in "lm" or "glm") 
#' @param data a data.frame containing response variables and covariates
#' @param maxBinom a vector of maximum trials (in binomial fashion), length(maxBinom) = nrow(data)
#' @param method chosen method for the inference : "freq", "RWMH".
#' @param curr a vector of initialization values corresponding to each of the covariates in the chosen columns of data, ordered in the same way
#' @param prior a list of vectors for the priors' hyperparameters
#' @param N the number of iterations post-adaptative step
#' @param nbAdaptMarche the number of adaptation steps, one step equals 100 iterations
#' @param warmup the number of iterations reserved to warmup (warmup >= 0 and warmup <= N)
#' @param epsilon the number of iterations reserved to warmup (warmup >= 0 and warmup <= N)
#' @param SigMarch a covariance matrix for the Gaussian proposal distribution (potential only a scalar)
#' @param invTemp a value to temper the likelihood
#' @return a list that includes a matrix of each coefficient posterior sample, the adapted covariance matrix for the Gaussian proposal distribution and some values to compute WAIC
#' 
#' @details ...
#' 
#' @examples ...
#' 
logisticRegression <- function(form,data,maxBinom=NULL,method="freq",curr=NULL,prior=NULL,N=5000,nbAdaptMarche=100,warmup=0,epsilon=10e-8,SigMarch=1,invTemp=1)
{
  if(!inherits(form,"formula")) stop("Please provide a correct formula of the type y ~ x .")
  if(N < 0 || nbAdaptMarche < 0 || invTemp < 0 || warmup < 0) stop("Either N or nbAdaptMarche or warmup or invTemp is strictly inferior to 0.")
  data <- as.data.frame(data)
  outcome <- all.vars(form)[attr(stats::terms(form),"response")]
  income <- all.vars(form)[-attr(stats::terms(form),"response")]
  nbVar <- length(income) 
  y <- data[[outcome]]
  
  if(method=="freq"){
    res <- stats::glm(formula = form,data = data,family = stats::binomial(link = "logit"))
    
    n <- nrow(data)
    statFreq <- summary(res)
    if(attr(stats::terms(form),"intercept")==1) nbVar <- nbVar + 1
    resSummary <- matrix(NA,ncol = nbVar,nrow=3)
    rownames(resSummary) <- c("2.5%","50%","97.5%")
    NAorNot <- !as.vector(statFreq$aliased)
    for(i in 1:nbVar){
      if(NAorNot[i]) try(resSummary[,i] <- statFreq$coefficients[i,1] + c(-1,0,1)*stats::qnorm(0.975)*statFreq$coefficients[i,2],silent=TRUE)
      else resSummary[,i] <- NA
    }
    if(attr(stats::terms(form),"intercept")==1) colnames(resSummary) <- c("(Intercept)",income)
    else colnames(resSummary) <- c(income)
    return(resSummary)
  }
  else{
    if(length(maxBinom) != nrow(data)) stop("Please provide a vector maxBinom of size ",nrow(data),".")
    if(is.null(prior)) stop("Please provide priors' hyperparameters.")
    if(is.null(curr) || (length(income) == length(curr) && length(income) == length(curr)-1)) stop(paste0("Please provide curr vector of size ",length(income)," or size ",length(income)+1," if an intercept is included."))
    if(attr(stats::terms(form),"intercept")==1){
      if(length(income) == length(curr)) curr <- c(stats::rnorm(1),curr)
      X <- as.matrix(data.frame("(Intercept)"=1,data[income]))
      nbVar <- nbVar + 1 
    }
    else{
      if(length(income) != length(curr)) stop(paste0("An intercept is not included, please provide curr vector of size ",length(income),"."))
      X <- as.matrix(data.frame(data[income]))
    }
    if(method=="RWMH"){
      if(is.null(SigMarch) || nrow(SigMarch) != ncol(SigMarch) || (length(income) != ncol(SigMarch) && length(income) != ncol(SigMarch)-1)) stop(paste0("Please provide SigMarch matrix of size ",length(income),"x",length(income)," or size ",length(income)+1,"x",length(income)+1," if an intercept is included."))
      if(attr(stats::terms(form),"intercept")==1){ 
        if(length(income) == nrow(SigMarch)){ 
          SigMarch <- rbind(rep(0,ncol(SigMarch)),SigMarch)
          SigMarch <- cbind(rep(0,nrow(SigMarch)),SigMarch)
          SigMarch[1,1] <- 1 
        }
      } else{ if(length(income) != length(curr)) stop(paste0("An intercept is not included, please provide SigMarch matrix of square size ",length(income),".")) }
      res <- .logisticRWMHjoint(N, nbAdaptMarche, y, X, maxBinom, curr, SigMarch, prior, warmup, invTemp)
    }
    return(res)
  }
}

