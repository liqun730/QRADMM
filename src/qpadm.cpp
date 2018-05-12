#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <chrono>
using namespace Rcpp;
using namespace arma;

arma::vec shrinkcpp1(arma::vec u, arma::vec v){
    arma::vec w=(1+sign(u-v))/2%(u-v)-(1+sign(-u-v))/2%(-u-v);
    return w;
}

arma::vec deriv(arma::vec beta, double a, double lambda, String penalty){
    int p = beta.n_elem;
    arma::vec df = arma::zeros(p);
    if(penalty=="scad"){
        for(int j=0;j<p;j++){
            if(fabs(beta(j))<=lambda) df(j)=lambda;
            else if(fabs(beta(j))<=a*lambda) df(j)=(a*lambda-fabs(beta(j)))/(a-1);
        }
    }else if(penalty=="mcp"){
         for(int j=0;j<p;j++){
             if(fabs(beta(j))<=a*lambda) df(j)=lambda-fabs(beta(j))/a;
         }
    }else{
      for(int j=0;j<p;j++){
        df(j)=lambda;
      }
    }
    return df;
}

//[[Rcpp::export]]
arma::vec QRADM(arma::mat xr,arma::vec yr,double ta,double rhor,double lambdar,int iter,bool intercept,int M,String penalty,double a){

    int maxit=(iter);
    double tau=(ta),rho=(rhor),lambda=(lambdar),alpha=1.7;
    arma::mat x=(xr);
    arma::vec y=(yr);
    int n=x.n_rows, p=x.n_cols;
    int ni = n/M;
    lambda = lambda/n, rho=rho/n;

    arma::vec r=arma::zeros(n),df,u=arma::zeros(n),beta,xbetai;
    arma::vec betaold=beta, beta_avg, eta_avg,comparev;
    if(intercept){
      p += 1;
      x.insert_cols(0,arma::ones(n));
      comparev=arma::zeros(3);
      beta=arma::zeros(p);
    }
    else{
      comparev=arma::zeros(2);
      beta=arma::zeros(p);
    }
    r = y-x*beta;

    double ABSTOL = 1e-7,RELTOL = 1e-4;
    double rnorm,epspri,snorm,epsdual;
    arma::mat betai=arma::zeros(p,M),etai=arma::zeros(p,M);
    arma::cube dat=arma::zeros<arma::cube>(p,p,M);
    for(int i=0;i<M;i++){
        arma::mat tmp,xi=x.rows(ni*i,ni*i+ni-1);
        if(ni>p && intercept) tmp = (xi.t()*xi+arma::eye(p,p)).i();
        else tmp = arma::eye(p,p)-xi.t()*(xi*xi.t()+arma::eye(ni,ni)).i()*xi;
        dat.slice(i)=tmp;
    }

    int iteration=0;
    while(iteration<maxit){
        beta_avg=mean(betai,1);
        eta_avg=mean(etai,1);
        betaold=beta;
        //update beta
        if(intercept){
          df = deriv(beta.subvec(1,p-1),a,lambda,penalty);
          beta(0) = beta_avg(0) + eta_avg(0)/rho;
          beta.subvec(1,p-1)=shrinkcpp1(beta_avg.subvec(1,p-1)+eta_avg.subvec(1,p-1)/rho,df/(rho*M));
        }else{
          df = deriv(beta,a,lambda,penalty);
          beta=shrinkcpp1(beta_avg+eta_avg/rho,df/(rho*M));
        }
        for(int i=0;i<M;i++){
            arma::vec yi=y.subvec(ni*i,ni*i+ni-1),ui=u.subvec(ni*i,ni*i+ni-1);
            arma::vec beta_i=betai.col(i),ri = r.subvec(ni*i,ni*i+ni-1);
            arma::mat xi=x.rows(ni*i,ni*i+ni-1);
            xbetai=alpha*xi*beta_i+(1-alpha)*(yi-ri);
            r.subvec(ni*i,ni*i+ni-1)=shrinkcpp1(ui/rho+yi-xbetai-.5*(2*tau-1)/(n*rho),.5*arma::ones<arma::vec>(ni)/(n*rho));
            //update betai
            betai.col(i)=dat.slice(i)*(xi.t()*(yi-r.subvec(ni*i,ni*i+ni-1)+ui/rho)-etai.col(i)/rho+beta);
            //update u and eta
            u.subvec(ni*i,ni*i+ni-1)=u.subvec(ni*i,ni*i+ni-1)+rho*(yi-xbetai-r.subvec(ni*i,ni*i+ni-1));
            etai.col(i)=etai.col(i)+rho*(betai.col(i)-beta);
        }

        if(intercept){
          rnorm = sqrt(accu(square(y-x*beta-r)));
          snorm = sqrt(accu(square(rho*x.cols(1,p-1)*(beta.subvec(1,p-1)-betaold.subvec(1,p-1)))));
          comparev(0)=sqrt(accu(square(x.cols(1,p-1)*beta.subvec(1,p-1))));
          comparev(1)=sqrt(accu(square(-r)));
          comparev(2)=sqrt(accu(square(y-beta(0))));
          epspri = sqrt(n)*ABSTOL + RELTOL*arma::max(comparev);
          epsdual = sqrt(n)*ABSTOL + RELTOL*sqrt(accu(square(u)));
        }
        else{
          rnorm = sqrt(accu(square(y-x*beta-r)));
          snorm = sqrt(accu(square(rho*x*(beta-betaold))));
          comparev(0)=sqrt(accu(square(x*beta)));
          comparev(1)=sqrt(accu(square(-r)));
          epspri = sqrt(n)*ABSTOL + RELTOL*arma::max(comparev);
          epsdual = sqrt(n)*ABSTOL + RELTOL*sqrt(accu(square(u)));
        }

        if (rnorm < epspri && snorm < epsdual)
          iteration = maxit+1;
        else
          iteration = iteration + 1;
    }

    arma::vec betanew=beta;
    return betanew;
}
