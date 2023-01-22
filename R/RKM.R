RKM <- function (X, N.comp, N.clust, N.random=1, nstart=100, show.random.ite=FALSE, maxit=100, eps=1e-05, mc.cores=1)
{
  GRC(X, N.comp, 0, N.clust, 1, 0, N.random, nstart, show.random.ite, maxit, eps, mc.cores)
}
