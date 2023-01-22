\name{FKM}
\alias{FKM}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
	Factorial K-means Clustering Analysis
}
\description{
	This function conducts the factorial k-means clustering analysis proposed
	by Vichi and Kiers (2001). The loss function of FKM is
	minimized by running rho1 as 2 and rho2 as 1 in the \code{GRC}
	function. Thus, it may give slightly different results from the
	original FKM algorithm.
}
\usage{
FKM(X, N.comp, N.clust, N.random=1, nstart=100, show.random.ite=FALSE, maxit=100, eps=1e-05, mc.cores=1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
		Centered data matrix (N * D), where N denotes sample size and D denotes the
		number of variables.
	}
  \item{N.comp}{
		The number of components
	}
  \item{N.clust}{
		The number of clusters.
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
	Vichi, M. and Kiers, H.A.L. (2001). Factorial k-means analysis for
	two-way data. Computational Statistics & Data Analysis, \bold{37}, 49--64.
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
res <- FKM(X, 1, 3)
res$cluster
res$A1
}
