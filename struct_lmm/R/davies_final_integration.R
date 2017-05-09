#install.packages('CompQuadForm', dependencies=TRUE, repos='http://cran.rstudio.com/')
library('CompQuadForm')

SKAT_Optimal_PValue_Davies<-function(pmin.q, MuQ, VarQ, KerQ, lambda, VarRemain, Df, tau_rho, r_all, pmin=NULL){

  #re<-try(integrate(SKAT_Optimal_Integrate_Func_Davies, lower=0, upper=30, subdivisions=500, pmin.q=pmin.q,param.m=param.m,r_all=r_all,abs.tol = 10^-15), silent = TRUE)

  #print('davies')
  re<-try(integrate(SKAT_Optimal_Integrate_Func_Davies, lower=0, upper=40, subdivisions=1000, pmin.q=pmin.q, MuQ=MuQ, VarQ=VarQ, KerQ=KerQ, lambda = lambda,  VarRemain= VarRemain,  Df= Df, tau = tau_rho, r_all=r_all, abs.tol = 10^-25), silent = TRUE)
  #print(re)
  # Might want to add this back in
  if(class(re) == "try-error"){
    #print('reverting liu')
    re<-SKAT_Optimal_PValue_Liu(pmin.q, MuQ, VarQ, KerQ, lambda, VarRemain,  Df, tau_rho, r_all, pmin)
    #print(re)
    return(re)
  } 
  #print(pmin)
  pvalue<-1-re[[1]]
  #print(length(r_all))
  #print(pvalue)
  if(!is.null(pmin)){
    if(pmin *length(r_all) < pvalue){
      pvalue = pmin *length(r_all)
    }
  }
  #print(pvalue)
  return(pvalue)

}


SKAT_Optimal_Integrate_Func_Davies<-function(x, pmin.q, MuQ, VarQ, KerQ, lambda, VarRemain, Df, tau, r_all){
  
  n.r<-length(r_all)
  n.x<-length(x)
  #print(x)

  temp1<-tau %x% t(x)
  #print(temp1[0:5])
  #print(temp1)

  #print(pmin.q[0:5])
  temp<-(pmin.q - temp1)/(1-r_all)
  #print(temp)
  temp.min<-apply(temp,2,min)
  #print(temp.min)

  re<-rep(0,length(x))
  #print(length(x))
  for(i in 1:length(x)){
    #a1<<-temp.min[i]
    min1<-temp.min[i]
    #print(min1)
    #print(sum(lambda))
    if(min1 > sum(lambda) * 10^4){
      #print('greater')
      temp<-0
      #print(temp)
    } else {
      #print('less')
      min1.temp<- min1 - MuQ   
      sd1<-sqrt(VarQ - VarRemain)/sqrt(VarQ)
      min1.st<-min1.temp *sd1 + MuQ
      dav.re<-davies(min1.st,lambda,acc=10^(-6))
      temp<-dav.re$Qq
      if(dav.re$ifault != 0){
        stop("dav.re$ifault is not 0")
      }
    }
    if(temp > 1){
      #print('too_large')
      temp=1
    }
    #lambda.record<<-param.m$lambda
    #print(c(min1,temp,dav.re$ifault,sum(param.m$lambda)))
    re[i]<-(1-temp) * dchisq(x[i],df=1)
  }
  return(re)

}


SKAT_Optimal_PValue_Liu<-function(pmin.q, MuQ, VarQ, KerQ, lambda, VarRemain, Df, tau, r_all, pmin=NULL){

  re<-integrate(SKAT_Optimal_Integrate_Func_Liu, lower=0, upper=40, subdivisions=2000, pmin.q=pmin.q, MuQ=MuQ, VarQ=VarQ, KerQ=KerQ, lambda = lambda,  VarRemain= VarRemain,  Df= Df, tau = tau, r_all=r_all,abs.tol = 10^-25)
  
  pvalue<-1-re[[1]]
  
  if(!is.null(pmin)){
    if(pmin *length(r_all) < pvalue){
      pvalue = pmin *length(r_all)
    }
  }
  
  return(pvalue)

}


SKAT_Optimal_Integrate_Func_Liu<-function(x,pmin.q, MuQ, VarQ, KerQ, lambda, VarRemain, Df, tau, r_all){
  
  #x<-1
  #print(length(x))
  #print(x)
  #X1<<-x
  #x<-X1

  n.r<-length(r_all)
  n.x<-length(x)

  temp1<-tau %x% t(x)

  temp<-(pmin.q - temp1)/(1-r_all)
  temp.min<-apply(temp,2,min)

  temp.q<-(temp.min - MuQ)/sqrt(VarQ)*sqrt(2*Df) + Df
  re<-pchisq(temp.q ,df=Df) * dchisq(x,df=1)
  
  return(re)

}
