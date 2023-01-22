##### dependencies #####
##dyn.load("OptimGRC_C.so")
OptimGRC_C <- function (X, U, A, info) {
  .Call("OptimGRC_C",
        as.double(X),
        as.double(U),
        as.double(A),
        as.double(info), ##rho1, rho2 are included
        package = "grc")
}


##### main #####
## N.comp1 <- 2; N.comp2 <- 0; N.clust <- 3; nstart <- 100; rho1 <- 1; rho2 <- 0; N.random <- 1; show.random.ite <- FALSE; mc.cores <- 7; maxit <- 100; eps <- 1e-05; A.first <- NULL
GRC <- function(X, N.comp1, N.comp2, N.clust, rho1=1, rho2=0, N.random=1, nstart=100, show.random.ite=FALSE, maxit=100, eps=1e-05, A.first=NULL, mc.cores=1)
{
  N.sub <- dim(X)[1]
  N.var <- dim(X)[2]
  N.comp <- N.comp1 + N.comp2 ## dimension for the whole subspace sp(A1) + sp(A2)

  if (rho1 <= rho2)
    stop("GRC WARNING: check the constraint for rho1 and rho2.")

  if(!is.null(A.first))
    N.random <- 1

  ## for parallel package
  if (mc.cores > 1) {
    if(nchar(system.file(package="parallel")) == 0)
      stop("Package 'parallel' is required to calculate using multicores.")
    require(parallel, quietly=TRUE)

    ##----- multiple ramdom initial starts -----
    temp.solution <- mclapply(1:N.random, FUN = function(n.random) {
      if (show.random.ite)
        if (n.random %% (N.random / 10) == 0)
          cat(paste(n.random, "; ", sep=""))

      set.seed(n.random)

      if (is.null(A.first)) {
        A <- qr.Q(qr(matrix(rnorm(N.var * N.comp), N.var, N.comp))) ## initial values
      } else {
        A <- A.first
      }
      val.lossfunc <- Inf
      conv <- 0
      U <- matrix(0, N.sub, N.clust)
      cluster <- sample(1:N.clust, N.sub, replace=TRUE)

      ## Optimization by K-means + EVD, or Optimization by K-means + GP algorithm
      info <- c(N.sub, N.var, N.comp1, N.comp2, N.clust, nstart, maxit, eps, rho1, rho2)
      solution <- try(OptimGRC_C(X, U, A, info), silent=FALSE)

      ans <- list()
      if (class(solution) == "try-error") {
        ans$loss <- Inf
      } else {
        ans$A <- solution[[1]]
        ans$U <- solution[[2]]
        ans$cluster <- c(ans$U %*% c(1:N.clust))
        ans$n.ite <- solution[[3]][1]
        ans$loss <- solution[[3]][2]
      }

      invisible({rm(list=c("solution")); gc(); gc(); gc()})

      return(ans)
    }, mc.cores=mc.cores)

    ## get results from temp.solution (obtained by parallel processing)
    best.lossfunc <- sapply(temp.solution,
                            FUN = function(x) x$loss
                            )
    cluster.all <- sapply(temp.solution,
                          FUN = function(x) x$cluster
                          )
    if (all(is.na(best.lossfunc)))
      stop("\nmywarning-->Could not find a feasible starting point...exiting\n",
           call. = FALSE)
    nb <- which(best.lossfunc == min(best.lossfunc, na.rm = TRUE))[1]
    solution <- temp.solution[[nb]]
    A <- solution$A
    cluster <- solution$cluster
    conv <- solution$conv
    val.lossfunc <- solution$loss
    n.ite <- solution$n.ite

    invisible({rm(list=c("temp.solution")); gc(); gc()})


    ## for not use parallel package
  } else {
    ret.val.lossfunc <- Inf
    best.lossfunc <- NULL

    ##----- multiple ramdom initial starts -----
    for (n.random in 1:N.random) {
      set.seed(n.random)
      if (show.random.ite)
        if (n.random %% (N.random / 10) == 0)
          cat(paste(n.random, "; ", sep=""))

      if (is.null(A.first)) {
        A <- qr.Q(qr(matrix(rnorm(N.var * N.comp), N.var, N.comp))) ## initial values
      } else {
        A <- A.first
      }
      conv <- 0
      U <- matrix(0, N.sub, N.clust)
      cluster <- sample(1:N.clust, N.sub, replace=TRUE)
      U[col(U) == cluster] <- 1

      ## Optimization by K-means + EVD, or Optimization by K-means + GP algorithm
      info <- c(N.sub, N.var, N.comp1, N.comp2, N.clust, nstart, maxit, eps, rho1, rho2)
      solution <- OptimGRC_C(X, U, A, info)

      best.lossfunc <- c(best.lossfunc, solution[[3]][2])

      ## check the values of loss functions
      if (solution[[3]][2] < ret.val.lossfunc) {
        ret.val.lossfunc <- solution[[3]][2]
        A.ret <- solution[[1]]
        U.ret <- solution[[2]]
        cluster.ret <- c(U.ret %*% c(1:N.clust))
        n.ite.ret <- solution[[3]][1]
        val.lossfunc.ret <- solution[[3]][2]
      }
    }

    A <- A.ret
    U <- U.ret
    cluster <- cluster.ret
    n.ite <- n.ite.ret
    val.lossfunc <- val.lossfunc.ret
  }


  if (N.comp == N.comp1) {
    A1 <- A
    A2 <- NULL
  } else {
    A1 <- A[, 1:N.comp1]
    A2 <- A[, (N.comp1 + 1):N.comp]
  }
  F <- X %*% A1
  U <- matrix(0, N.sub, N.clust)
  U[col(U) == cluster] <- 1


  ## calculate cluster centroids
  NullCluster <- 0
  if (any(diag(t(U) %*% U) == 0)) {
    F.mean <- NULL
    NullCluster <- 1
  } else {
    Pu <- U %*% solve(t(U) %*% U) %*% t(U)
    F.mean <- solve(t(U) %*% U) %*% t(U) %*% F
  }

  if (show.random.ite)
    cat("\n")

##  class(A) <- "loadings"
##  class(A1) <- "loadings"
##  if (!is.null(A2))
##    class(A2) <- "loadings"

  return(list("A1"=A1, "A2"=A2, "A"=A, "F"=F, "U"=U, "cluster"=cluster, "lossfunc"=val.lossfunc, "n.ite"=n.ite, "F.mean"=F.mean, "NullCluster"=NullCluster, "lossfunc.vec"=best.lossfunc))
}
