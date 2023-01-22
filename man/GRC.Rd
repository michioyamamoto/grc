\name{GRC}
\alias{GRC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
	Generalized Reduced Clustering Analysis
}
\description{
	This function conducts the generalized reduced clustering analysis proposed
	by Yamamoto and Hwang (2014).
}
\usage{
GRC(X, N.comp1, N.comp2, N.clust, rho1=1, rho2=0, N.random=1, nstart=100,
show.random.ite=FALSE, maxit=100, eps=1e-05, A.first=NULL, mc.cores=1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
		Centered data matrix (N * D), where N denotes sample size and D denotes the
		number of variables.
	}
  \item{N.comp1}{
		The number of components, which correspond to the subspace for a
		cluster structure.
	}
  \item{N.comp2}{
		The number of components, which correspond to the subspace for irrelevant features.
	}
  \item{N.clust}{
		The number of clusters.
	}
  \item{rho1}{
		A tuning parameter for the within-group variability in a subspace.
	}
  \item{rho2}{
		A tuning parameter for the between-group variability in a subspace.
	}
  \item{N.random}{
		The number of random sets of parmeters for initial random starts.
	}
  \item{nstart}{
		The number of random starts for the k-means clustering in each
		optimization of U.
	}
  \item{show.random.ite}{
		If \code{"TRUE"}, the number of each iteration is shown.
	}
	\item{maxit}{
		A max number of iterations for ALS algorithm.
	}
	\item{eps}{
		A criterion for the convergence of the loss function.
	}
	\item{A.first}{
		An initial value for the loading matrix A.
	}
  \item{mc.cores}{
		\code{GRC} adopts a multithread process for multiple initial
		random starts by using the \code{parallel} package. If
		\code{"mc.cores"=1}, a single core process is conducted. Since the
		\code{parallel} package is not available on Windows, \code{mc.cores}
		must always be set to 1.
	}
}
% \details{
% %%  ~~ If necessary, more details than the description above ~~
% }
\value{
	\item{A1}{Estimated loading matrix which corresponds to a subspace for
	a cluster structure.}
	\item{A2}{Estimated loading matrix which corresponds to a subspace for
	irrelevant features.}
	\item{A}{Estimated loading matrix.}
	\item{F}{Estimated component score matrix which corresponds to a
		subspace for a cluster structure.}
	\item{U}{The cluster assignment matrix (N * \code{N.clust}).}
	\item{cluster}{A vector of cluster assignment parameters.}
	\item{lossfunc}{The value of loss function.}
	\item{n.ite}{The number of iteration needed for convergence.}
	\item{F.mean}{A component score matrix of cluster centroids.}
}
\references{
	Yamamoto, M. and Hwang, H. (2014). A general formulation of cluster
	analysis with dimension reduction and subspace
	separation. Behaviormetrika, \bold{41}, 1, 115--129.
}
\author{
	Michio Yamamoto\cr
	\email{michio.koko@gmail.com}
}
% \note{
% %%  ~~further notes~~
% }

% %% ~Make other sections like Warning with \section{Warning }{....} ~

% \seealso{
% %% ~~objects to See Also as \code{\link{help}}, ~~~
% }
\examples{
## Toy example
## 300 subjects and 5 variables
## Consider three clusters in a one-dimensional subspace
N.sub.k <- 100
X1 <- rnorm(N.sub.k, -10, 1)
X2 <- rnorm(N.sub.k, 0, 1)
X3 <- rnorm(N.sub.k, 10, 1)
X <- cbind(c(X1, X2, X3), matrix(rnorm(300 * 4), 300, 4))
X <- scale(X)
res <- GRC(X, 1, 0, 3)
res$cluster
res$A1

## multiple random initial starts may work
# res <- GRC(X, 1, 0, 3, N.random=100)
# res$cluster
# res$A1
}

