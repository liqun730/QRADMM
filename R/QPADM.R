#' Penalized quantile regression with ADMM
#'
#' Currently support penalized quantile regression with the Lasso, MCP and SCAD penalties.
#' This is a pseudo-parallel implementation. The beta-update is solved with EM algorithm
#' that results in a closed-form solution. See the paper
#' \url{https://amstat.tandfonline.com/doi/full/10.1080/10618600.2017.1328366#.WsMVH9PwafU}
#' for details.
#'
#' @seealso \code{\link{QRADMM}}
#' @param X The design matrix (without intercept)
#' @param y The response vector
#' @param tau The quantile of interest
#' @param rho The augmentation parameter for the ADMM
#' @param lambda The penalization parameter
#' @param iter Maximum number of iterations allowed
#' @param intercept Whether to include the intercept into the model, default is TRUE
#' @param M Number of subsets (split the data into M blocks)
#' @param penalty Name of the penalty to use, currently support ("none","lasso","enet","scad","mcp")
#' @param a The shape parameter for the MCP/SCAD penalty
#' @return The coefficient estimation of the linear quantile regression model
#' @examples
#' require(MASS)
#' n=3000; p=100
#' Sig = matrix(0, nrow = p, ncol = p)
#' for(i1 in 1:p){
#'   for(j in 1:p){
#'     Sig[i1,j] = 0.5^abs(i1-j)
#'   }
#' }
#' X = mvrnorm(n, mu = rep(0,p), Sigma = Sig, tol = 1e-6)
#' X[,1]=pnorm(X[,1])
#' u = rnorm(n)
#' y=X[,3] + X[,4] + X[,5] + X[,1]*u
#' beta = QPADM(X, y, .9, 1, 20, penalty="mcp", intercept=FALSE, M=10)
#' @export
#'
QPADM <- function(X,y,tau,rho,lambda,iter=500,intercept=TRUE,M=1,penalty="lasso",a=3.7){

  if(!(penalty %in% c("lasso","scad","mcp"))){
    warning(paste0('Penalty "',penalty,'" is not supported, Lasso estimation is returned'))
    penalty = "lasso"
  }

  if(lambda <= 0 && n<p)
    warning("Dimension is larger than sample size, penalization is strongly recommended")

  .Call('_QRADMM_QRADM', PACKAGE = 'QRADMM', X, y, tau, rho, lambda, iter, intercept, M, penalty, a)
}
