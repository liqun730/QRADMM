// The QRADMM function for solving quantile regression with lasso, elastic net, MCP, and SCAD penalties

#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

//the soft thresholding function
arma::vec shrink(arma::vec u, arma::vec v){

    arma::vec w=(1+sign(u-v))/2%(u-v)-(1+sign(-u-v))/2%(-u-v);
    return w;

}


//CD for lasso penalized least square
arma::vec lselasso(arma::mat x,arma::vec y, arma::vec beta,int m,bool intercept,double rho,double lambda){

    int maxit=m;
    int iteration=1;
    int n=x.n_rows,p=x.n_cols;
    double error, ABSTOL = 1e-6;
    arma::vec xjrj0;
    double xjrj,xjsq;
    arma:: vec betaold=beta;
    arma::vec xj,rj,bjcand,loss,sort_cand;
    arma::uvec idx;
    arma::vec r = y-x*beta;

    while (iteration<=maxit){

        betaold=beta;
        for (int j=0;j<p;j++){

            if(j==0&&intercept){
                  beta(0)=accu(y-x.cols(1,p-1)*beta.subvec(1,p-1))/n;
                  r = r + (betaold(0)-beta(0))*x.col(0);
            }
            else{
                xj = x.col(j);
                rj=r+xj*beta(j);
                xjrj0=xj.t()*rj;
                xjrj=xjrj0(0);
                xjsq=accu(square(xj));
                bjcand=arma::zeros(3);
                loss=arma::zeros(3);

                bjcand(0)=0;
                bjcand(1)=(xjrj-lambda/rho)/xjsq;
                bjcand(2)=(xjrj+lambda/rho)/xjsq;

                loss(0)=.5*accu(square(xj*bjcand(0)-rj));

                if(bjcand(1)>0)
                    loss(1)=.5*accu(square(xj*bjcand(1)-rj))+fabs(bjcand(1))*lambda/rho;
                else
                    loss(1)=loss(0)+1;

                if(bjcand(2)<0)
                    loss(2)=.5*accu(square(xj*bjcand(2)-rj))+fabs(bjcand(2))*lambda/rho;
                else
                    loss(2)=loss(0)+1;

                idx=sort_index(loss);
                sort_cand = bjcand(idx);
                beta(j)=sort_cand(0);

                r = r + (betaold(j)-beta(j))*xj;
            }

        }

        error=accu(abs(beta-betaold));
        if(error<ABSTOL)
            iteration=maxit+1;
        else
            iteration++;

    }

    arma::vec betanew = beta;
    return betanew;

}


//CD for elastic net penalized least square
arma::vec lsenet(arma::mat x,arma::vec y, arma::vec beta,int m,bool intercept,double rho,double lambda,double lambda1,double lambda2){

    int maxit=(m);
    int iteration=1;
    int n=x.n_rows,p=x.n_cols;
    double error, ABSTOL = 1e-6;
    arma::vec xjrj0;
    double xjrj,xjsq;
    arma:: vec betaold=beta;
    arma::vec xj,rj;
    arma::vec r = y-x*beta;

    while (iteration<=maxit){

        betaold=beta;
        for (int j=0;j<p;j++){

            if(j==0&&intercept){
                beta(0)=accu(y-x.cols(1,p-1)*beta.subvec(1,p-1))/n;
                r = r + (betaold(0)-beta(0))*x.col(0);
            }
            else{
                xj = x.col(j);
                rj=r+xj*beta(j);
                xjrj0=xj.t()*rj;
                xjrj=xjrj0(0);
                xjsq=accu(square(xj));

                if(fabs(xjrj)<=lambda*lambda1/rho)
                    beta(j)=0;
                else if(xjrj>0)
                    beta(j)=(xjrj-lambda*lambda1/rho)/(xjsq+2*lambda*lambda2/rho);
                else
                    beta(j)=(xjrj+lambda*lambda1/rho)/(xjsq+2*lambda*lambda2/rho);

                r = r + (betaold(j)-beta(j))*xj;
            }

        }

        error=accu(abs(beta-betaold));
        if(error<ABSTOL)
            iteration=maxit+1;
        else
            iteration++;

    }

    arma::vec betanew = beta;
    return betanew;

}


//The MCP penalty for x>=0
double mcp(double lambda,double a,double x){

    if(x>=a*lambda)
        return .5*a*lambda*lambda;
    else
        return lambda*x-.5*x*x/a;

}


//The SCAD penalty for x>=0
double scad(double lambda,double a,double x){

    if(x>=a*lambda)
        return .5*(a*a-1)*lambda*lambda/(a-1);
    else if(x>lambda)
        return (a*lambda*x-.5*(x*x+lambda*lambda))/(a-1);
    else
        return lambda*x;

}


//CD for least square with mcp penalty with initial beta value=beta
arma::vec lsemcp(arma::mat x,arma::vec y, arma::vec beta, int m,bool intercept,double rho,double lambda, double a){

    int maxit=m;
    int n=x.n_rows,p=x.n_cols;
    arma::vec xjrj0;
    double xjrj,xjsq;
    double error,ABSTOL = 1e-6;
    arma:: vec betaold=beta;
    arma::vec xj,rj,bjcand=arma::zeros(4),loss=arma::zeros(4),sort_cand;
    arma::uvec idx;
    int iteration=1;
    arma::vec r = y-x*beta;

    while (iteration<=maxit){

        betaold=beta;
        for (int j=0;j<p;j++){

            if(j==0&&intercept){
                  beta(0)=accu(y-x.cols(1,p-1)*beta.subvec(1,p-1))/n;
                  r = r + (betaold(0)-beta(0))*x.col(0);
            }
            else{
                xj = x.col(j);
                rj=r+xj*beta(j);
                xjrj0=xj.t()*rj;
                xjsq=accu(square(xj));
                xjrj=xjrj0(0);
                loss=arma::zeros(4);

                bjcand(0)=0;
                bjcand(1)=xjrj/xjsq;
                bjcand(2)=(xjrj+lambda/rho)/(xjsq-1/(rho*a));
                bjcand(3)=(xjrj-lambda/rho)/(xjsq-1/(rho*a));

                loss(0)=.5*rho*accu(square(xj*bjcand(0)-rj));

                if(fabs(bjcand(1))>=a*lambda){
                    loss(1)=.5*rho*accu(square(xj*bjcand(1)-rj));
                    loss(1)=loss(1)+mcp(lambda,a,fabs(bjcand(1)));
                }
                else
                    loss(1)=loss(0)+1;

                if(bjcand(2)>-a*lambda&&bjcand(2)<0){
                    loss(2)=.5*rho*accu(square(xj*bjcand(2)-rj));
                    loss(2)=loss(2)+mcp(lambda,a,fabs(bjcand(2)));
                }
                else
                    loss(2)=loss(0)+1;

                if(bjcand(3)<a*lambda&&bjcand(3)>lambda){
                    loss(3)=.5*rho*accu(square(xj*bjcand(3)-rj));
                    loss(3)=loss(3)+mcp(lambda,a,fabs(bjcand(3)));
                }
                else
                    loss(3)=loss(0)+1;

                idx=sort_index(loss);
                sort_cand = bjcand(idx);
                beta(j)=sort_cand(0);
                if(fabs(beta(j))<1e-6)
                    beta(j) = 0;

                r = r + (betaold(j)-beta(j))*xj;
            }

        }

        error=accu(abs(beta-betaold));
        if(error < ABSTOL)
            iteration = maxit + 1;
        else
            iteration=iteration+1;

    }

    arma::vec betanew = beta;
    return betanew;

}


//CD for least square with scad penalty with initial beta value=beta, m is the maxit
arma::vec lsescad(arma::mat x,arma::vec y, arma::vec beta, int m,bool intercept,double rho,double lambda, double a){

    int maxit=m;
    int n=x.n_rows,p=x.n_cols;
    arma::vec xjrj0;
    double xjrj,xjsq;
    double error,ABSTOL = 1e-6;
    arma:: vec betaold=beta;
    arma::vec xj,rj,bjcand=arma::zeros(6),loss=arma::zeros(6),sort_cand;
    arma::uvec idx;
    int iteration=1;
    arma::vec r = y-x*beta;

    while (iteration<=maxit){

        betaold=beta;
        for (int j=0;j<p;j++){

            if(j==0&&intercept){
                  beta(0)=accu(y-x.cols(1,p-1)*beta.subvec(1,p-1))/n;
                  r = r + (betaold(0)-beta(0))*x.col(0);
            }
            else{
                xj = x.col(j);
                rj=r+xj*beta(j);
                xjrj0=xj.t()*rj;
                xjsq=accu(square(xj));
                xjrj=xjrj0(0);
                loss=arma::zeros(6);

                bjcand(0)=0;
                bjcand(1)=(rho*xjrj+lambda)/(rho*xjsq);
                bjcand(2)=(rho*xjrj-lambda)/(rho*xjsq);
                bjcand(3)=(rho*xjrj+a*lambda/(a-1))/(rho*xjsq-1/(a-1));
                bjcand(4)=(rho*xjrj-a*lambda/(a-1))/(rho*xjsq-1/(a-1));
                bjcand(5)=xjrj/xjsq;

                loss(0)=.5*rho*accu(square(xj*bjcand(0)-rj));

                if(bjcand(1)>-lambda&&bjcand(1)<0){
                    loss(1)=.5*rho*accu(square(xj*bjcand(1)-rj));
                    loss(1)=loss(1)+scad(lambda,a,fabs(bjcand(1)));
                }
                else
                    loss(1)=loss(0)+1;

                if(bjcand(2)<lambda&&bjcand(2)>0){
                    loss(2)=.5*rho*accu(square(xj*bjcand(2)-rj));
                    loss(2)=loss(2)+scad(lambda,a,fabs(bjcand(2)));
                }
                else
                    loss(2)=loss(0)+1;

                if(bjcand(3)>-a*lambda&&bjcand(3)<-lambda){
                    loss(3)=.5*rho*accu(square(xj*bjcand(3)-rj));
                    loss(3)=loss(3)+scad(lambda,a,fabs(bjcand(3)));
                }
                else
                    loss(3)=loss(0)+1;

                if(bjcand(4)<a*lambda&&bjcand(4)>lambda){
                    loss(4)=.5*rho*accu(square(xj*bjcand(4)-rj));
                    loss(4)=loss(4)+scad(lambda,a,fabs(bjcand(4)));
                }
                else
                    loss(4)=loss(0)+1;

                if(bjcand(5)>a*lambda||bjcand(5)<-a*lambda){
                    loss(5)=.5*rho*accu(square(xj*bjcand(5)-rj));
                    loss(5)=loss(5)+scad(lambda,a,fabs(bjcand(5)));
                }
                else
                    loss(5)=loss(0)+1;

                idx=sort_index(loss);
                sort_cand = bjcand(idx);
                beta(j)=sort_cand(0);
                if(fabs(beta(j))<1e-6)
                    beta(j) = 0;

                r = r + (betaold(j)-beta(j))*xj;
            }

        }

    error=accu(abs(beta-betaold));
    if(error < ABSTOL)
        iteration = maxit + 1;
    else
        iteration=iteration+1;

    }

    arma::vec betanew = beta;
    return betanew;

}


//The beta update function for QRADMM with different penalties
arma::vec betaupdate(arma::mat ima,arma::mat x,arma::vec y, arma::vec beta,int m,bool intercept,double rho,double lambda, double a, double lambda1, double lambda2, std::string penalty){

    if(penalty=="none")
        beta = ima*y;
    else if(penalty=="lasso")
        beta = lselasso(x,y,beta,m,intercept,rho,lambda);
    else if(penalty=="enet")
        beta = lsenet(x,y,beta,m,intercept,rho,lambda,lambda1,lambda2);
    else if(penalty=="scad")
        beta = lsescad(x,y,beta,m,intercept,rho,lambda,a);
    else
        beta = lsemcp(x,y,beta,m,intercept,rho,lambda,a);

    return beta;

}

//[[Rcpp::export]]
arma::vec QRADMMCPP(arma::mat x,arma::vec y,double tau,double rho,double lambda,int iter, bool intercept,std::string penalty, double a,double lambda1,double lambda2){

    int maxit=(iter);
    int n=x.n_rows, p=x.n_cols;
    rho = rho/n, lambda=lambda/n;
    arma::uvec ix;
    arma:: vec r,beta,betaold,comparev;

    if(intercept){
        x.insert_cols(0,arma::ones(n));
        comparev=arma::zeros(3);
        beta=arma::zeros(p+1);
    }
    else{
        comparev=arma::zeros(2);
        beta=arma::zeros(p);
    }

    double ABSTOL = 1e-7,RELTOL = 1e-4;
    double rnorm,epspri,snorm,epsdual,alpha=1.0;
    int iteration=1;
    arma::vec u=arma::zeros(n),xbeta;
    r=y-x*beta;
    betaold=beta;
    arma::mat ima = (x.t()*x).i()*x.t();

    while (iteration<=maxit){

        //updata r- residual
        xbeta=alpha*x*beta+(1-alpha)*(y-r);
        r = shrink(u/rho+y-xbeta-.5*(2*tau-1)/(n*rho),.5*arma::ones<arma::vec>(n)/(n*rho));
        // ix = (r==0);
        // r = r + ix%1e-4;

        //update beta
        betaold=beta;
        beta=betaupdate(ima,x,u/rho+y-r,beta,maxit,intercept,rho,lambda,a,lambda1,lambda2,penalty);

        //updata u
        u = u +rho*(y-xbeta-r);

        //termination check
        if(intercept){
                rnorm = sqrt(accu(square(y-x*beta-r)));
                snorm = sqrt(accu(square(rho*x.cols(1,p)*(beta.subvec(1,p)-betaold.subvec(1,p)))));
                comparev(0)=sqrt(accu(square(x.cols(1,p)*beta.subvec(1,p))));
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
