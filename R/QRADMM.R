#' Penalized quantile regression with ADMM
#'
#' This is a non-parallel implementation. Currently support unpenalized quantile regression and
#' penalized quantile regression with the Lasso, elastic net, MCP and SCAD penalties. See the paper
#' \url{https://onlinelibrary.wiley.com/doi/abs/10.1111/insr.12221} for details.
#'
#' @seealso \code{\link{QPADM}}
#' @param X The design matrix (without intercept)
#' @param y The response vector
#' @param tau The quantile of interest
#' @param rho The augmentation parameter for the ADMM
#' @param lambda The penalization parameter
#' @param iter Maximum number of iterations allowed
#' @param intercept Whether to include the intercept into the model, default is TRUE
#' @param penalty Name of the penalty to use, currently support ("none","lasso","enet","scad","mcp")
#' @param a The shape parameter for the MCP/SCAD penalty
#' @param lambda1 Extra penalization parameter for the elastic net. Specifically, the l1 part of the elastic net has penalization coefficient lambda*labda1
#' @param lambda2 Extra penalization parameter for the elastic net. Specifically, the l2 part of the elastic net has penalization coefficient lambda*labda2
#' @return The coefficient estimation of the linear quantile regression model
#' @examples
#' require(MASS)
#' n=300;p=1000
#' Sig = matrix(0, nrow = p, ncol = p)
#' for(i1 in 1:p){
#'   for(j in 1:p){
#'     Sig[i1,j] = 0.5^abs(i1-j)
#'   }
#' }
#' X = mvrnorm(n, mu = rep(0,p), Sigma = Sig, tol = 1e-6)
#' X[,1] = pnorm(X[,1])
#' u = rnorm(n)
#' y=X[,3] + X[,4] + X[,5] + X[,1]*u
#' beta = QRADMM(X, y, .9, 1, 20, penalty="scad", intercept=FALSE)
#' @export
#'
QRADMM <- function(X,y, tau, rho, lambda, iter=100, intercept=TRUE,penalty="none",a=3.7,lambda1=1,lambda2=1){

  if(!(penalty %in% c("none","lasso","enet","scad","mcp"))){
    warning(paste0('Penalty "',penalty,'" is not supported, unpenalized estimation is returned'))
    penalty = "none"
  }

  if(penalty=="none" && n<p)
    warning("Dimension is larger than sample size, penalization is strongly recommended")

  .Call('_QRADMM_QRADMMCPP', PACKAGE = 'QRADMM', X, y, tau, rho, lambda, iter, intercept, penalty, a, lambda1, lambda2)

}
