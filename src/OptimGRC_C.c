//��������������������������������������������������������
//   Generalized Reduced Clustering�κ�Ŭ��
//
//  �ե�����̾��OptimGRC_C.c
//  �ե��������ơ�
//  �����ԡ�YAMAMOTO, Michio
//  ��������2013ǯ09��16��
//  �ǽ���������2023ǯ8��3��
//  �����ȡ�N.comp==N.comp1 or not��ʬ�����Ƥ���
//           K-means����ʬ�ϤȤꤢ����Lloyd�Υ��르�ꥺ������Ѥ��뤳�Ȥˤ���
//           k-means�ΤȤ����n_random_kmeans==1�ξ��Υ��饹�����濴�������ʬ����������150214��
//           package�����Τ���˺ǽ���ǧ (221210)
//           F77_CALL�� FCONE FCONE���ɵ����� (230803)
//��������������������������������������������������������
#include <stdlib.h>
#include <time.h> /* time�ˤ������ν�����Τ��� */
#include <R.h>
#include <Rdefines.h>
#include <R_ext/Parse.h>
#include <R_ext/Lapack.h>

#ifndef FCONE
# define FCONE
#endif
  
void f_identity(int N, double *I);
void f_grad(double *X, int row1, int col1, double *U, int row2, int col2, double *A, int row3, int col3, int N_comp1, int N_comp2, double rho1, double rho2, double *G);
double f_lossfunc(double *X, int row1, int col1, double *U, int row2, int col2, double *A, int row3, int col3, int N_comp1, int N_comp2, double rho1, double rho2);
void kmeans_Lloyd(double *x, int *pn, int *pp, double *cen, int *pk, int *cl, int *pmaxiter, int *nc, double *wss);
void kmeans_MacQueen(double *x, int *pn, int *pp, double *cen, int *pk, int *cl, int *pmaxiter, int *nc, double *wss);
void GPAlgorithm(double *X, double *U, double *A, int N_sub, int N_var, int N_comp1, int N_comp2, int N_clust, double rho1, double rho2);


SEXP OptimGRC_C (SEXP X, SEXP U, SEXP A, SEXP INFO)
{
  // R����ΰ��Ϥ�
  int     N_sub = (int) REAL(INFO)[0];
  int     N_var = (int) REAL(INFO)[1];
  int     N_comp1 = (int) REAL(INFO)[2];
  int     N_comp2 = (int) REAL(INFO)[3];
  int     N_clust = (int) REAL(INFO)[4];
  int     N_random_kmeans = (int) REAL(INFO)[5];
  int     N_ite = (int) REAL(INFO)[6];
  double  eps = (double) REAL(INFO)[7];
  double  rho1 = (double) REAL(INFO)[8];
  double  rho2 = (double) REAL(INFO)[9];
  int     N_comp = N_comp1 + N_comp2;

  // �ѿ����
  double *A_current;
  double *A_new;
  double *A1;
  /* double *A2; */
  int    *aRandArray; /* ��������Ǽ���ѿ� */
  int     aRandValArray[N_clust]; /* ����μ������ѿ� */
  double  blas_one = 1.0;
  double  blas_zero = 0.0;
  int    *cl_vec;
  int    *cl_vec_temp;
  int    *nc;
  int     Counter; /* �롼�ץ��������ѿ� */
  double  diff_loss;
  double *eigen_value;
  double *F1;
  double *Fc;
  int     i, j, k, l;
  int     info; /* for dsyev_�ؿ� */
  double  loss = 1.0E+100, loss_old;
  int     lwork = N_var * N_sub; /* ��ͭ��ʬ���work space */
  double *Mat_cc;
  double *Mat_cs;
  double *Mat_cv;
  double *Mat_ss;
  double *Mat_vs;
  double *Mat_vv;
  int     N_clust_clust = N_clust * N_clust;
  int     N_clust_sub = N_clust * N_sub;
  int     N_clust_var = N_clust * N_var;
  int     N_clust_comp1 = N_clust * N_comp1;
  /* double  N_clust_double = (double) N_clust; */
  int     n_ite = 0;
  int     N_ite_kmeans = 1000;
  /* int     N_max_alpha = 15; */
  int     n_random_kmeans;
  int     N_sub_clust = N_sub * N_clust;
  int     N_sub_comp1 = N_sub * N_comp1;
  int     N_sub_sub = N_sub * N_sub;
  int     N_var_comp = N_var * N_comp;
  int     N_var_comp1 = N_var * N_comp1;
  /* int     N_var_comp2 = N_var * N_comp2; */
  int     N_var_sub   = N_var * N_sub;
  int     N_var_var   = N_var * N_var;
  int     NumRand; /* �Ĥ��������� */
  int     RandKey; /* �������������ѿ� */
  double *U_old;
  double *work;
  double *wss; /* kmeans�����Ѥ��� */
  double  wss_best; /* kmeans�����Ѥ��� */
  double  wss_sum = 1.0E+100; /* kmeans�����Ѥ��� */

  // �ǽ�Ū������ͤ������PROTECT
  SEXP ans;
  SEXP ans_A;
  SEXP ans_U;
  SEXP ans_ind;
  /* SEXP ans_hoge; /\* �ǽ�Ū�˺�����Ƥ������� *\/ */

  PROTECT(ans = allocVector(VECSXP, 4));
  PROTECT(ans_A = allocMatrix(REALSXP, N_var, N_comp));
  PROTECT(ans_U = allocMatrix(REALSXP, N_sub, N_clust));
  PROTECT(ans_ind = allocVector(REALSXP, 2));
  /* PROTECT(ans_hoge = allocMatrix(REALSXP, N_var, N_comp)); /\* �ǽ�Ū�˺�����Ƥ������� *\/ */


  // malloc �ˤ�����������
  aRandArray  = (int    *) malloc(sizeof(int)    * N_sub);
  A1          = (double *) malloc(sizeof(double) * N_var_comp1);
  A_current   = (double *) malloc(sizeof(double) * N_var_comp);
  A_new       = (double *) malloc(sizeof(double) * N_var_comp);
  cl_vec      = (int    *) malloc(sizeof(int)    * N_sub);
  cl_vec_temp = (int    *) malloc(sizeof(int)    * N_sub);
  eigen_value = (double *) malloc(sizeof(double) * N_var);
  F1          = (double *) malloc(sizeof(double) * N_sub_comp1);
  Fc          = (double *) malloc(sizeof(double) * N_clust_comp1);
  Mat_cc      = (double *) malloc(sizeof(double) * N_clust_clust);
  Mat_cs      = (double *) malloc(sizeof(double) * N_clust_sub);
  Mat_cv      = (double *) malloc(sizeof(double) * N_clust_var);
  Mat_ss      = (double *) malloc(sizeof(double) * N_sub_sub);
  Mat_vs      = (double *) malloc(sizeof(double) * N_var_sub);
  Mat_vv      = (double *) malloc(sizeof(double) * N_var_var);
  nc          = (int    *) malloc(sizeof(int)    * N_clust);
  U_old       = (double *) malloc(sizeof(double) * N_sub_clust);
  work        = (double *) malloc(sizeof(double) * lwork);
  wss         = (double *) malloc(sizeof(double) * N_clust);

  /****************************/
  /*   A2��̵ͭ�ˤ��ʬ��      */
  /****************************/
  /* A2��̵����� */
  if (N_comp == N_comp1) {

	/****************************/
	/*   ALS algorithm start    */
	/****************************/
	do {
	  n_ite++;
	  loss_old = loss;

	  /*****************************/
	  /*          U�ι���          */
	  /*****************************/
	  /* F1��׻����Ƥ���*/
	  // dgemm_("N", "N", &N_sub, &N_comp1, &N_var, &blas_one, REAL(X), &N_sub, REAL(A), &N_var, &blas_zero, F1, &N_sub); // F1 = X %*% A1
    F77_CALL(dgemm)("N", "N", &N_sub, &N_comp1, &N_var, &blas_one, REAL(X), &N_sub, REAL(A), &N_var, &blas_zero, F1, &N_sub FCONE FCONE); // F1 = X %*% A1

	  // ʣ������ͤ��Ѥ���kmeans_Lloyd�¹Ԥ���
	  n_random_kmeans = 0;
	  wss_best = 1.0E+100;
	  do {
	  	n_random_kmeans++;

	  	/* ����ˤ�륯�饹�����濴�˳���������ֹ�μ��� */
		srand(time(NULL));
		//srand(n_random_kmeans * n_ite);
	  	for (Counter = 0; Counter < N_sub; Counter++)
	  	  aRandArray[Counter] = Counter + 1;
	  	NumRand = N_sub;
		//		printf("OK1\r"); fflush(stdout);
	  	for (Counter = 0; Counter < N_clust; Counter++) {
	  	  /* �������� */
	  	  RandKey = rand();
	  	  /* �������������Ĥ�����ǳ�ä�;���������� */
	  	  RandKey %= NumRand;
	  	  /* �������Ȥ�����Ȥ��Ƽ������� */
	  	  aRandValArray[Counter] = aRandArray[RandKey];
	  	  /* ���Ѥ��������̤���Ѥ�������֤������� */
	  	  aRandArray[RandKey] = aRandArray[NumRand - 1];
	  	  /* ���������ĸ��餹 */
	  	  --NumRand;
	  	}
	  	/* ���饹�����濴����Fc�κ��� */
		if (n_ite > 1 && n_random_kmeans == 1) {
		  F77_CALL(dgemm)("T", "N", &N_clust, &N_var, &N_sub, &blas_one, REAL(U), &N_sub, REAL(X), &N_sub, &blas_zero, Mat_cv, &N_clust FCONE FCONE); //t(U) %*% X
		  F77_CALL(dgemm)("N", "N", &N_clust, &N_comp1, &N_var, &blas_one, Mat_cv, &N_clust, REAL(A), &N_var, &blas_zero, Fc, &N_clust FCONE FCONE); //t(U) %*% X %*% A
		  F77_CALL(dgemm)("T", "N", &N_clust, &N_clust, &N_sub, &blas_one, REAL(U), &N_sub, REAL(U), &N_sub, &blas_zero, Mat_cc, &N_clust FCONE FCONE); //t(U) %*% U
		  for (i = 0; i < N_clust; i++) //solve(t(U) %*% U)
			  Mat_cc[i + i * N_clust] = 1 / Mat_cc[i + i * N_clust];
		  F77_CALL(dgemm)("N", "N", &N_clust, &N_comp1, &N_clust, &blas_one, Mat_cc, &N_clust, Fc, &N_clust, &blas_zero, Mat_cv, &N_clust FCONE FCONE); //solve(t(U) %*% U) %*% t(U) %*% X %*% A
		  for (k = 0; k < N_clust; k++)
			  for (l = 0; l < N_comp; l++)
			    Fc[k + N_clust * l] = Mat_cv[k + N_clust * l];
		}
		else {
		  for (k = 0; k < N_clust; k++)
			  for (l = 0; l < N_comp; l++)
			    Fc[k + N_clust * l] = F1[aRandValArray[k] + N_sub * l];
		}

	  	//x: �ǡ�������, m: N.sub, p: N.var, centers: ���饹��������͹���, k: N.clust, c1: integer(N.sub), iter: N.ite, nc: N.clust, wss: double(k)
		kmeans_Lloyd(F1, &N_sub, &N_comp1, Fc, &N_clust, cl_vec_temp, &N_ite_kmeans, nc, wss);
	  	//kmeans_MacQueen(F1, &N_sub, &N_comp1, Fc, &N_clust, cl_vec_temp, &N_ite_kmeans, nc, wss);
	  	/* WSS���¤�׻����� */
	  	wss_sum = 0.0;
	  	for (k = 0; k < N_clust; k++) {
	  	  wss_sum = wss_sum + wss[k];
		}

	  	/* �Ǿ���WSS����ĥ��饹������̤���¸���� */
	  	if (wss_sum < wss_best) {
	  	  wss_best = wss_sum;
	  	  for (i = 0; i < N_sub; i++)
	  		cl_vec[i] = cl_vec_temp[i];
	  	}
	  } while (n_random_kmeans < N_random_kmeans); /* End of kmeans */

	  /* ����줿���饹������٥뤫�����U�ι��� */
	  for (i = 0; i < N_sub_clust; i++)
	  	REAL(U)[i] = 0.0;
	  for (i = 0; i < N_sub; i++)
	  	REAL(U)[i + N_sub * (cl_vec[i] - 1)] = 1.0;


	  /*****************************/
	  /*          A�ι���          */
	  /*****************************/
	  /* ��ͭ��ʬ��ˤ�빹�� */
	  F77_CALL(dgemm)("T", "N", &N_clust, &N_clust, &N_sub, &blas_one, REAL(U), &N_sub, REAL(U), &N_sub, &blas_zero, Mat_cc, &N_clust FCONE FCONE); //t(U) %*% U
	  for (i = 0; i < N_clust; i++) //solve(t(U) %*% U)
	  	Mat_cc[i + i * N_clust] = 1 / Mat_cc[i + i * N_clust];
	  F77_CALL(dgemm)("N", "T", &N_clust, &N_sub, &N_clust, &blas_one, Mat_cc, &N_clust, REAL(U), &N_sub, &blas_zero, Mat_cs, &N_clust FCONE FCONE); //solve(t(U) %*% U) %*% t(U)
	  F77_CALL(dgemm)("N", "N", &N_sub, &N_sub, &N_clust, &blas_one, REAL(U), &N_sub, Mat_cs, &N_clust, &blas_zero, Mat_ss, &N_sub FCONE FCONE); //U %*% solve(t(U) %*% U) %*% t(U)

	  /* (1 - rho1) * I_N + (rho1 - rho2) * P_U */
	  for (i = 0; i < N_sub; i++) {
	  	for (j = 0; j < N_sub; j++) {
	  	  if (i == j) {
	  		Mat_ss[i + N_sub * i] = (1.0 - rho1) + (rho1 - rho2) * Mat_ss[i + N_sub * i];
	  	  } else {
	  		Mat_ss[i + N_sub * j] = (rho1 - rho2) * Mat_ss[i + N_sub * j];
	  	  }
	  	}
	  }

	  F77_CALL(dgemm)("T", "N", &N_var, &N_sub, &N_sub, &blas_one, REAL(X), &N_sub, Mat_ss, &N_sub, &blas_zero, Mat_vs, &N_var FCONE FCONE); // X' %*% ((1 - rho1) * I_N + (rho1 - rho2) * P_U)
	  F77_CALL(dgemm)("N", "N", &N_var, &N_var, &N_sub, &blas_one, Mat_vs, &N_var, REAL(X), &N_sub, &blas_zero, Mat_vv, &N_var FCONE FCONE); // X' %*% ((1 - rho1) * I_N + (rho1 - rho2) * P_U) %*% X
	  F77_CALL(dsyev)("V", "U", &N_var, Mat_vv, &N_var, eigen_value, work, &lwork, &info FCONE FCONE); /* ��ͭ�ͤ�����Ǥ��뤳�Ȥ���� */
	  if (info != 0) {
	  	//	break;
	  	error(("error code %d from Lapack routine '%s'"), info, "dsyev_");
	  }

	  /* A���ͭ�٥��ȥ���֤����� */
	  for (j = 0; j < N_var; j++)
	  	for (l = 0; l < N_comp; l++) //��N_comp==N_comp1�Ǥ���
	  	  REAL(A)[j + N_var * l] = Mat_vv[j + N_var * (N_var - l - 1)];

  	  /* ��«Ƚ�� */
	  loss = f_lossfunc(REAL(X), N_sub, N_var, REAL(U), N_sub, N_clust, REAL(A), N_var, N_comp, N_comp1, N_comp2, rho1, rho2);
	  diff_loss = fabs(loss_old - loss);

	} while (diff_loss > eps && n_ite < N_ite); // End of ALS algorithm
  }



  /* A2��ͭ���� */
  else {

	/****************************/
	/*   ALS algorithm start    */
	/****************************/
	do {
	  n_ite++;
	  loss_old = loss;

	  /*****************************/
	  /*          U�ι���          */
	  /*****************************/
	  /* F1��׻����Ƥ���*/
	  for (i = 0; i < N_var; i++)
		for (j = 0; j < N_comp; j++)
		  if (j < N_comp1)
			A1[i + j * N_var] = REAL(A)[i + j * N_var];
	  F77_CALL(dgemm)("N", "N", &N_sub, &N_comp1, &N_var, &blas_one, REAL(X), &N_sub, A1, &N_var, &blas_zero, F1, &N_sub FCONE FCONE); // F1 = X %*% A1
	  // ʣ������ͤ��Ѥ���kmeans_Lloyd�¹Ԥ���
	  n_random_kmeans = 0;
	  wss_best = 1.0E+100;
	  do {
	  	n_random_kmeans++;

	  	/* ����ˤ�륯�饹�����濴�˳���������ֹ�μ��� */
		//		srand(time(NULL));
		srand(n_random_kmeans);
	  	for (Counter = 0; Counter < N_sub; Counter++)
	  	  aRandArray[Counter] = Counter + 1;
	  	NumRand = N_sub;
	  	for (Counter = 0; Counter < N_clust; Counter++) {
	  	  /* �������� */
	  	  RandKey = rand();
	  	  /* �������������Ĥ�����ǳ�ä�;���������� */
	  	  RandKey %= NumRand;
	  	  /* �������Ȥ�����Ȥ��Ƽ������� */
	  	  aRandValArray[Counter] = aRandArray[RandKey];
	  	  /* ���Ѥ��������̤���Ѥ�������֤������� */
	  	  aRandArray[RandKey] = aRandArray[NumRand - 1];
	  	  /* ���������ĸ��餹 */
	  	  --NumRand;
	  	}
	  	/* ���饹�����濴����Fc�κ��� */
		if (n_ite > 1 && n_random_kmeans == 1) {
		  F77_CALL(dgemm)("T", "N", &N_clust, &N_var, &N_sub, &blas_one, REAL(U), &N_sub, REAL(X), &N_sub, &blas_zero, Mat_cv, &N_clust FCONE FCONE); //t(U) %*% X
		  F77_CALL(dgemm)("N", "N", &N_clust, &N_comp1, &N_var, &blas_one, Mat_cv, &N_clust, A1, &N_var, &blas_zero, Fc, &N_clust FCONE FCONE); //t(U) %*% X %*% A1
		  F77_CALL(dgemm)("T", "N", &N_clust, &N_clust, &N_sub, &blas_one, REAL(U), &N_sub, REAL(U), &N_sub, &blas_zero, Mat_cc, &N_clust FCONE FCONE); //t(U) %*% U
		  for (i = 0; i < N_clust; i++) //solve(t(U) %*% U)
			  Mat_cc[i + i * N_clust] = 1 / Mat_cc[i + i * N_clust];
		  F77_CALL(dgemm)("N", "N", &N_clust, &N_comp1, &N_clust, &blas_one, Mat_cc, &N_clust, Fc, &N_clust, &blas_zero, Mat_cv, &N_clust FCONE FCONE); //solve(t(U) %*% U) %*% t(U) %*% X %*% A
		  for (k = 0; k < N_clust; k++)
			  for (l = 0; l < N_comp1; l++)
			    Fc[k + N_clust * l] = Mat_cv[k + N_clust * l];
		}
		else {
		  for (k = 0; k < N_clust; k++)
			for (l = 0; l < N_comp1; l++)
			  Fc[k + N_clust * l] = F1[aRandValArray[k] + N_sub * l];
		}

	  	//x: �ǡ�������, m: N.sub, p: N.var, centers: ���饹��������͹���, k: N.clust, c1: integer(N.sub), iter: N.ite, nc: N.clust, wss: double(k)
		kmeans_Lloyd(F1, &N_sub, &N_comp1, Fc, &N_clust, cl_vec_temp, &N_ite_kmeans, nc, wss);
	  	//kmeans_MacQueen(F1, &N_sub, &N_comp1, Fc, &N_clust, cl_vec_temp, &N_ite_kmeans, nc, wss);
	  	/* WSS���¤�׻����� */
	  	wss_sum = 0.0;
	  	for (k = 0; k < N_clust; k++) {
	  	  wss_sum = wss_sum + wss[k];
		}

	  	/* �Ǿ���WSS����ĥ��饹������̤���¸���� */
	  	if (wss_sum < wss_best) {
	  	  wss_best = wss_sum;
	  	  for (i = 0; i < N_sub; i++)
	  		cl_vec[i] = cl_vec_temp[i];
	  	}
	  } while (n_random_kmeans < N_random_kmeans); /* End of kmeans */

	  /* ����줿���饹������٥뤫�����U�ι��� */
	  for (i = 0; i < N_sub_clust; i++)
	  	REAL(U)[i] = 0.0;
	  for (i = 0; i < N_sub; i++)
	  	REAL(U)[i + N_sub * (cl_vec[i] - 1)] = 1.0;


	  /*****************************/
	  /*          A�ι���          */
	  /*****************************/
	  /* for (k = 0; k < N_var_comp; k++) */
	  /* 	REAL(ans_hoge)[k] = REAL(A)[k]; /\* �ǽ�Ū�˺�����Ƥ������� *\/ */

	  /* GP algorithm�ˤ���Ŭ�� */
	  GPAlgorithm(REAL(X), REAL(U), REAL(A), N_sub, N_var, N_comp1, N_comp2, N_clust, rho1, rho2);

  	  /* ��«Ƚ�� */
	  loss = f_lossfunc(REAL(X), N_sub, N_var, REAL(U), N_sub, N_clust, REAL(A), N_var, N_comp, N_comp1, N_comp2, rho1, rho2);
	  diff_loss = fabs(loss_old - loss);

	  /* if (n_ite == 2) */
	  /* 	goto out; */

	} while (diff_loss > eps && n_ite < N_ite); // End of ALS algorithm
  }

  /* ��μ����Ϥ� */
  //printf("hoge\n");
 /* out: */
  for (i = 0; i < N_var_comp; i++)
   	REAL(ans_A)[i] = REAL(A)[i];
  for (i = 0; i < N_sub_clust; i++)
	REAL(ans_U)[i] = REAL(U)[i];
  REAL(ans_ind)[0] = (double) n_ite;
  REAL(ans_ind)[1] = loss;

  /* for (k = 0; k < N_var_comp; k++) */
  /* 	REAL(ans_hoge)[k] = RE[k]; /\* �ǽ�Ū�˺�����Ƥ������� *\/ */

  SET_VECTOR_ELT(ans, 0, ans_A);
  SET_VECTOR_ELT(ans, 1, ans_U);
  SET_VECTOR_ELT(ans, 2, ans_ind);
  /* SET_VECTOR_ELT(ans, 3, ans_hoge); /\* �ǽ�Ū�˺�����Ƥ������� *\/ */

  // ���곫��
  /* UNPROTECT(5); */
  UNPROTECT(4);

  // malloc�ǳ�����Ƥ�����γ���
  free(A1);
  free(A_current);
  free(A_new);
  free(cl_vec);
  free(cl_vec_temp);
  free(eigen_value);
  free(F1);
  free(Fc);
  free(Mat_cc);
  free(Mat_cs);
  free(Mat_cv);
  free(Mat_ss);
  free(Mat_vs);
  free(Mat_vv);
  free(U_old);
  free(work);
  free(wss);

  return(ans);

}


/* GP algorithm�ˤ��A�κ�Ŭ�� */
void GPAlgorithm(double *X, double *U, double *A, int N_sub, int N_var, int N_comp1, int N_comp2, int N_clust, double rho1, double rho2)
{
  /* �����μ����Ϥ� */
  int N_comp = N_comp1 + N_comp2;

  /* �ѿ���� */
  double *A_new;
  double *A_current;
  double *A_target;
  double  alpha;
  double  alpha_ini = 1.0;
  double  blas_one = 1.0;
  double  blas_zero = 0.0;
  double *C;
  double  diff_cr;
  double  eps = 0.00001;
  double *G;
  double *I_comp;
  /* int     i, j; */
  int     i;
  int     info;
  char    jobu = 'S', jobvt = 'S';
  double  lossfunc_current;
  double  lossfunc_new;
  int     lwork = N_sub * N_var;
  int     N_comp_comp = N_comp * N_comp;
  int     n_ite, N_ite = 100;
  int     n_ite_alpha; //����double���������Ƥ������ѹ����Ƥߤ�
  int     N_max_alpha = 15;
  int     N_sub_var = N_sub * N_var;
  int     N_var_comp = N_var * N_comp;
  /* int     N_var_comp1 = N_var * N_comp1; */
  /* int     N_var_comp2 = N_var * N_comp2; */
  int     N_var_var = N_var * N_var;
  double  scalar;
  double *singular_value;
  double *U_svd;
  double *Vt_svd;
  double *work;

  A_current  = (double *) malloc(sizeof(double) * N_var_comp);
  A_new      = (double *) malloc(sizeof(double) * N_var_comp);
  A_target   = (double *) malloc(sizeof(double) * N_var_comp);
  C          = (double *) malloc(sizeof(double) * N_comp_comp);
  G          = (double *) malloc(sizeof(double) * N_var_comp);
  I_comp     = (double *) malloc(sizeof(double) * N_comp_comp);
  singular_value = (double *) malloc(sizeof(double) * N_var);
  U_svd      = (double *) malloc(sizeof(double) * N_sub_var);
  Vt_svd     = (double *) malloc(sizeof(double) * N_var_var);
  work       = (double *) malloc(sizeof(double) * N_sub_var);

  f_identity(N_comp, I_comp); // ñ�̹������Ƥ���

  /* GP algorithm */
  n_ite = 0;
  for (i = 0; i < (N_var * N_comp); i++)
	A_new[i] = A[i];

  do {
	n_ite++;

	for (i = 0; i < (N_var * N_comp); i++)
	  A_current[i] = A_new[i];

	//Calculation of Gradient of Loss function at A
	f_grad(X, N_sub, N_var, U, N_sub, N_clust, A_current, N_var, N_comp, N_comp1, N_comp2, rho1, rho2, G);

	/* ��Ŭ��alpha��õ�� */
	// ���ߤ���Ū�ؿ����ͤ�׻����Ƥ���
	n_ite_alpha = 0;
	alpha = 2 * alpha_ini;
	lossfunc_current = f_lossfunc(X, N_sub, N_var, U, N_sub, N_clust, A_current, N_var, N_comp, N_comp1, N_comp2, rho1, rho2);

	do {
	  n_ite_alpha++;
	  alpha = alpha / 2;
	  scalar = -1 * alpha;

	  /* A�ι��� */
	  for (i = 0; i < (N_var * N_comp); i++)
		A_target[i] = A_current[i];

	  F77_CALL(dgemm)("N", "N", &N_var, &N_comp, &N_comp, &scalar, G, &N_var, I_comp, &N_comp, &blas_one, A_target, &N_var FCONE FCONE); // A - alpha * G
	  F77_CALL(dgesvd)(&jobu, &jobvt, &N_var, &N_comp, A_target, &N_var, singular_value, U_svd, &N_var, Vt_svd, &N_comp, work, &lwork, &info FCONE FCONE);
	  if (info != 0)
		  error(("error code %d from Lapack routine '%s'"), info, "dgesvd");
	  F77_CALL(dgemm)("N", "N", &N_var, &N_comp, &N_comp, &blas_one, U_svd, &N_var, Vt_svd, &N_comp, &blas_zero, A_new, &N_var FCONE FCONE); // A.new = U %*% t(V)

	  /* �����ͤ���Ū�ؿ����ͤ�׻����Ƥ��� */
	  lossfunc_new = f_lossfunc(X, N_sub, N_var, U, N_sub, N_clust, A_new, N_var, N_comp, N_comp1, N_comp2, rho1, rho2);

	} while (lossfunc_new > lossfunc_current && n_ite_alpha < N_max_alpha);

	//alpha = 0�ξ�硤����A���������Ƥ���
	if (n_ite_alpha == N_max_alpha) {
	  alpha = 0;
	  for (i = 0; i < (N_var * N_comp); i++)
		A_new[i] = A_current[i];
	  lossfunc_new = lossfunc_current;
	}

	/* ��«Ƚ�� */
	// �����ͤ���Ū�ؿ����ͤ�׻����Ƥ���
	//f_lossfunc(X, N_sub, N_var, U, N_sub, N_clust, A_new, N_var, N_comp, N_comp1, N_comp2, rho1, rho2, lossfunc_new);
	diff_cr = lossfunc_current - lossfunc_new;

	} while (diff_cr > eps && n_ite < N_ite); // GP algorithm��λ

  /* ��μ����Ϥ� */
  for (i = 0; i < (N_var * N_comp); i++)
	A[i] = A_new[i];

  free(A_current);
  free(A_new);
  free(A_target);
  free(C);
  free(G);
  free(I_comp);
  free(singular_value);
  free(U_svd);
  free(Vt_svd);
  free(work);
}


// R internal function
void kmeans_Lloyd(double *x, int *pn, int *pp, double *cen, int *pk, int *cl, int *pmaxiter, int *nc, double *wss)
{
    int n = *pn, k = *pk, p = *pp, maxiter = *pmaxiter;
    int iter, i, j, c, it, inew = 0;
    double best, dd, tmp;
    Rboolean updated;

    for(i = 0; i < n; i++) cl[i] = -1;
    for(iter = 0; iter < maxiter; iter++) {
	updated = FALSE;
	for(i = 0; i < n; i++) {
	    /* find nearest centre for each point */
	    best = R_PosInf;
	    for(j = 0; j < k; j++) {
		dd = 0.0;
		for(c = 0; c < p; c++) {
		    tmp = x[i+n*c] - cen[j+k*c];
		    dd += tmp * tmp;
		}
		if(dd < best) {
		    best = dd;
		    inew = j+1;
		}
	    }
	    if(cl[i] != inew) {
		updated = TRUE;
		cl[i] = inew;
	    }
	}
	if(!updated) break;
	/* update each centre */
	for(j = 0; j < k*p; j++) cen[j] = 0.0;
	for(j = 0; j < k; j++) nc[j] = 0;
	for(i = 0; i < n; i++) {
	    it = cl[i] - 1; nc[it]++;
	    for(c = 0; c < p; c++) cen[it+c*k] += x[i+c*n];
	}
	for(j = 0; j < k*p; j++) cen[j] /= nc[j % k];
    }

    *pmaxiter = iter + 1;
    for(j = 0; j < k; j++) wss[j] = 0.0;
    for(i = 0; i < n; i++) {
	it = cl[i] - 1;
	for(c = 0; c < p; c++) {
	    tmp = x[i+n*c] - cen[it+k*c];
	    wss[it] += tmp * tmp;
	}
    }
}

// R internal function
void kmeans_MacQueen(double *x, int *pn, int *pp, double *cen, int *pk,
		     int *cl, int *pmaxiter, int *nc, double *wss)
{
    int n = *pn, k = *pk, p = *pp, maxiter = *pmaxiter;
    int iter, i, j, c, it, inew = 0, iold;
    double best, dd, tmp;
    Rboolean updated;

    /* first assign each point to the nearest cluster centre */
    for(i = 0; i < n; i++) {
	best = R_PosInf;
	for(j = 0; j < k; j++) {
	    dd = 0.0;
	    for(c = 0; c < p; c++) {
		tmp = x[i+n*c] - cen[j+k*c];
		dd += tmp * tmp;
	    }
	    if(dd < best) {
		best = dd;
		inew = j+1;
	    }
	}
	if(cl[i] != inew) cl[i] = inew;
    }
   /* and recompute centres as centroids */
    for(j = 0; j < k*p; j++) cen[j] = 0.0;
    for(j = 0; j < k; j++) nc[j] = 0;
    for(i = 0; i < n; i++) {
	it = cl[i] - 1; nc[it]++;
	for(c = 0; c < p; c++) cen[it+c*k] += x[i+c*n];
    }
    for(j = 0; j < k*p; j++) cen[j] /= nc[j % k];

    for(iter = 0; iter < maxiter; iter++) {
	updated = FALSE;
	for(i = 0; i < n; i++) {
	    best = R_PosInf;
	    for(j = 0; j < k; j++) {
		dd = 0.0;
		for(c = 0; c < p; c++) {
		    tmp = x[i+n*c] - cen[j+k*c];
		    dd += tmp * tmp;
		}
		if(dd < best) {
		    best = dd;
		    inew = j;
		}
	    }
	    if((iold = cl[i] - 1) != inew) {
		updated = TRUE;
		cl[i] = inew + 1;
		nc[iold]--; nc[inew]++;
		/* update old and new cluster centres */
		for(c = 0; c < p; c++) {
		    cen[iold+k*c] += (cen[iold+k*c] - x[i+n*c])/nc[iold];
		    cen[inew+k*c] += (x[i+n*c] - cen[inew+k*c])/nc[inew];
		}
	    }
	}
	if(!updated) break;
    }

    *pmaxiter = iter + 1;
    for(j = 0; j < k; j++) wss[j] = 0.0;
    for(i = 0; i < n; i++) {
	it = cl[i] - 1;
	for(c = 0; c < p; c++) {
	    tmp = x[i+n*c] - cen[it+k*c];
	    wss[it] += tmp * tmp;
	}
    }
}


/* ��Ū�ؿ����ͤ���� */
double f_lossfunc(double *X, int row1, int col1, double *U, int row2, int col2, double *A, int row3, int col3, int N_comp1, int N_comp2, double rho1, double rho2)
{
  // �ѿ����
  double *A1;
  double  blas_one = 1;
  double  blas_zero = 0;
  double  blas_minus_one = -1;
  double *G1;
  double *G2;
  int     i, j;
  /* int     inc_dscal = 1; */
  double *I_N;
  double  lossfunc;
  double *Mat_cc;
  double *Mat_cs;
  double *Mat_ss;
  double *Mat_vs;
  double *Mat_vv;
  double *Mat_sa1;
  double *Mat_sa1_2;
  double *Mat_a1a1;
  int     N_clust = col2;
  int     N_comp = col3;
  int     N_sub = row1;
  int     N_var = col1;
  /* int     N_var_comp = N_var * N_comp; */
  double  rho1_local = rho1;
  double  rho2_local = rho2;
  double *temp_X;
  double  term1 = 0;
  double  term2 = 0;
  double  term3 = 0;
  double *Z1;
  double *Z2;
  double *Z3;

  // malloc �ˤ�����������
  A1        = (double *) malloc(sizeof(double) * N_var * N_comp1);
  G1        = (double *) malloc(sizeof(double) * N_var * N_comp1);
  G2        = (double *) malloc(sizeof(double) * N_var * N_comp2);
  I_N       = (double *) malloc(sizeof(double) * N_sub * N_sub);
  Mat_cc    = (double *) malloc(sizeof(double) * N_clust * N_clust);
  Mat_cs    = (double *) malloc(sizeof(double) * N_clust * N_sub);
  Mat_ss    = (double *) malloc(sizeof(double) * N_sub * N_sub);
  Mat_vs    = (double *) malloc(sizeof(double) * N_var * N_sub);
  Mat_vv    = (double *) malloc(sizeof(double) * N_var * N_var);
  Mat_sa1   = (double *) malloc(sizeof(double) * N_sub * N_comp1);
  Mat_sa1_2 = (double *) malloc(sizeof(double) * N_sub * N_comp1);
  Mat_a1a1  = (double *) malloc(sizeof(double) * N_comp1 * N_comp1);
  temp_X    = (double *) malloc(sizeof(double) * N_sub * N_var);
  Z1        = (double *) malloc(sizeof(double) * N_sub * N_var);
  Z2        = (double *) malloc(sizeof(double) * N_sub * N_comp1);
  Z3        = (double *) malloc(sizeof(double) * N_sub * N_comp1);


  /* A1����Ƥ��� */
  for (i = 0; i < N_var; i++) {
  	for (j = 0; j < N_comp; j++) {
  	  if (j < N_comp1)
  	  	A1[i + j * N_var] = A[i + j * N_var];
  	}
  }

  /* ñ�̹���I_N����Ƥ��� */
  f_identity(N_sub, I_N);

  /* Z1 = X - X %*% A %*% t(A) ����� */
  for (i = 0; i < (N_sub * N_var); i++)
	temp_X[i] = X[i];
  F77_CALL(dgemm)("N", "T", &N_var, &N_var, &N_comp, &blas_one, A, &N_var, A, &N_var, &blas_zero, Mat_vv, &N_var FCONE FCONE); // A %*% t(A)
  F77_CALL(dgemm)("N", "N", &N_sub, &N_var, &N_var, &blas_minus_one, X, &N_sub, Mat_vv, &N_var, &blas_one, temp_X, &N_sub FCONE FCONE); // X - X %*% A %*% t(A)
  for (i = 0; i < (N_sub * N_var); i++) // Z1 = X - X %*% A %*% t(A)
	  Z1[i] = temp_X[i];

  /* Z2 = X %*% A1 - U %*% solve(t(U) %*% U) %*% t(U) %*% X %*% A1 ����� */
  F77_CALL(dgemm)("N", "N", &N_sub, &N_comp1, &N_var, &blas_one, X, &N_sub, A1, &N_var, &blas_zero, Mat_sa1, &N_sub FCONE FCONE); //X %*% A1
  for (i = 0; i < (N_sub * N_comp1); i++)
	  Mat_sa1_2[i] = Mat_sa1[i];
  F77_CALL(dgemm)("T", "N", &N_clust, &N_clust, &N_sub, &blas_one, U, &N_sub, U, &N_sub, &blas_zero, Mat_cc, &N_clust FCONE FCONE); //t(U) %*% U
  for (i = 0; i < N_clust; i++) //solve(t(U) %*% U)
  	Mat_cc[i + i * N_clust] = 1 / Mat_cc[i + i * N_clust];
  F77_CALL(dgemm)("N", "T", &N_clust, &N_sub, &N_clust, &blas_one, Mat_cc, &N_clust, U, &N_sub, &blas_zero, Mat_cs, &N_clust FCONE FCONE); //solve(t(U) %*% U) %*% t(U)
  F77_CALL(dgemm)("N", "N", &N_sub, &N_sub, &N_clust, &blas_one, U, &N_sub, Mat_cs, &N_clust, &blas_zero, Mat_ss, &N_sub FCONE FCONE); //U %*% solve(t(U) %*% U) %*% t(U)
  F77_CALL(dgemm)("N", "N", &N_sub, &N_comp1, &N_sub, &blas_minus_one, Mat_ss, &N_sub, Mat_sa1, &N_sub, &blas_one, Mat_sa1_2, &N_sub FCONE FCONE); //X %*% A1 - U %*% solve(t(U) %*% U) %*% t(U) %*% X %*% A1
  for (i = 0; i < (N_sub * N_comp1); i++)  //Z2 = X %*% A1 - U %*% solve(t(U) %*% U) %*% t(U) %*% X %*% A1
	  Z2[i] = Mat_sa1_2[i];

  /* Z3 = U %*% solve(t(U) %*% U) %*% t(U) %*% X %*% A1 */
  F77_CALL(dgemm)("N", "N", &N_sub, &N_comp1, &N_var, &blas_one, X, &N_sub, A1, &N_var, &blas_zero, Mat_sa1, &N_sub FCONE FCONE); // X %*% A1
  F77_CALL(dgemm)("N", "N", &N_sub, &N_comp1, &N_sub, &blas_one, Mat_ss, &N_sub, Mat_sa1, &N_sub, &blas_zero, Mat_sa1_2, &N_sub FCONE FCONE); // U %*% solve(t(U) %*% U) %*% t(U) %*% X %*% A1
  for (i = 0; i < (N_sub * N_comp1); i++) // Z3 = U %*% solve(t(U) %*% U) %*% t(U) %*% X %*% A1
	  Z3[i] = Mat_sa1_2[i];

  /* value of loss function ����� */
  F77_CALL(dgemm)("T", "N", &N_var, &N_var, &N_sub, &blas_one, Z1, &N_sub, Z1, &N_sub, &blas_zero, Mat_vv, &N_var FCONE FCONE); // t(Z1) %*% Z1
  for (i = 0; i < N_var; i++)
	  term1 = term1 + Mat_vv[i + i * N_var];

  F77_CALL(dgemm)("T", "N", &N_comp1, &N_comp1, &N_sub, &blas_one, Z2, &N_sub, Z2, &N_sub, &blas_zero, Mat_a1a1, &N_comp1 FCONE FCONE); // t(Z2) %*% Z2
  for (i = 0; i < N_comp1; i++)
	  term2 = term2 + Mat_a1a1[i + i * N_comp1];

  F77_CALL(dgemm)("T", "N", &N_comp1, &N_comp1, &N_sub, &blas_one, Z3, &N_sub, Z3, &N_sub, &blas_zero, Mat_a1a1, &N_comp1 FCONE FCONE); // t(Z3) %*% Z3
  for (i = 0; i < N_comp1; i++)
	  term3 = term3 + Mat_a1a1[i + i * N_comp1];

  lossfunc = term1 + rho1_local * term2 + rho2_local * term3;

  // malloc�ǳ�����Ƥ�����γ���
  free(A1);
  free(G1);
  free(G2);
  free(I_N);
  free(Mat_cc);
  free(Mat_cs);
  free(Mat_ss);
  free(Mat_vs);
  free(Mat_vv);
  free(Mat_sa1);
  free(Mat_sa1_2);
  free(Mat_a1a1);
  free(temp_X);
  free(Z1);
  free(Z2);
  free(Z3);

  return(lossfunc);
}


/* ñ�̹���I_N�򻻽Ф��� */
void f_identity(int N, double *I)
{
  int i, j;
  for (i = 0; i < N; i++) {
  	for (j = 0; j < N; j++) {
	    if (i == j) {
    		I[j + i * N] = 1;
	    } else {
    		I[j + i * N] = 0;
	    }
  	}
  }
}


/* Gradient G����� */
//����Ū�ˡ�row1, col1, row2, col2, row3, col3���ͤ��������Υ����å����ɲä��Ƥ��ɤ�
void f_grad(double *X, int row1, int col1, double *U, int row2, int col2, double *A, int row3, int col3, int N_comp1, int N_comp2, double rho1, double rho2, double *G)
{
  // �ѿ����
  double *A1;
  double *A2;
  double  blas_one = 1;
  double  blas_zero = 0;
  double  coef1 = 1 - rho1;
  double  coef2 = rho1 - rho2;
  double *G1;
  double *G2;
  int     i, j;
  double *I_N;
  int     inc_dscal = 1;
  int     N_clust = col2;
  int     N_comp = col3;
  int     N_sub = row1;
  int     N_var = col1;
  int     N_var_comp = N_var * N_comp;
  double  scal_G = -2;
  double *temp1;
  double *temp2;
  double *temp3;
  double *temp4;
  double *temp5;
  double *temp6;

  // malloc �ˤ�����������
  A1    = (double *) malloc(sizeof(double) * N_var * N_comp1);
  A2    = (double *) malloc(sizeof(double) * N_var * N_comp2);
  G1    = (double *) malloc(sizeof(double) * N_var * N_comp1);
  G2    = (double *) malloc(sizeof(double) * N_var * N_comp2);
  I_N   = (double *) malloc(sizeof(double) * N_sub * N_sub);
  temp1 = (double *) malloc(sizeof(double) * N_clust * N_clust);
  temp2 = (double *) malloc(sizeof(double) * N_clust * N_sub);
  temp3 = (double *) malloc(sizeof(double) * N_sub * N_sub);
  temp4 = (double *) malloc(sizeof(double) * N_var * N_sub);
  temp5 = (double *) malloc(sizeof(double) * N_var * N_var);
  temp6 = (double *) malloc(sizeof(double) * N_var * N_var);

  /* A1��A2����Ƥ��� */
  for (i = 0; i < N_var; i++) {
  	for (j = 0; j < N_comp; j++) {
  	  if (j < N_comp1) {
  	  	A1[i + j * N_var] = A[i + j * N_var];
	  } else {
  	  	A2[i + (j - N_comp1) * N_var] = A[i + j * N_var];
	  }
  	}
  }

  /* ñ�̹���I_N����Ƥ��� */
  f_identity(N_sub, I_N);

  /* U����٥��ȥ뤬ĥ����֤ؤμͱƹ������� */
  F77_CALL(dgemm)("T", "N", &N_clust, &N_clust, &N_sub, &blas_one, U, &N_sub, U, &N_sub, &blas_zero, temp1, &N_clust FCONE FCONE); //t(U) %*% U
  for (i = 0; i < N_clust; i++) //solve(t(U) %*% U)
  	temp1[i + i * N_clust] = 1 / temp1[i + i * N_clust];
  F77_CALL(dgemm)("N", "T", &N_clust, &N_sub, &N_clust, &blas_one, temp1, &N_clust, U, &N_sub, &blas_zero, temp2, &N_clust FCONE FCONE); //solve(t(U) %*% U) %*% t(U)
  F77_CALL(dgemm)("N", "N", &N_sub, &N_sub, &N_clust, &blas_one, U, &N_sub, temp2, &N_clust, &blas_zero, temp3, &N_sub FCONE FCONE); //U %*% solve(t(U) %*% U) %*% t(U)

  /* G1����� */
  F77_CALL(dgemm)("N", "N", &N_sub, &N_sub, &N_sub, &coef1, I_N, &N_sub, I_N, &N_sub, &coef2, temp3, &N_sub FCONE FCONE); //(1 - rho1) * diag(N.sub) + (rho1 - rho2) * U %*% solve(t(U) %*% U) %*% t(U)
  F77_CALL(dgemm)("T", "N", &N_var, &N_sub, &N_sub, &blas_one, X, &N_sub, temp3, &N_sub, &blas_zero, temp4, &N_var FCONE FCONE); //t(X) %*% ((1 - rho1) * diag(N.sub) + (rho1 - rho2) * U %*% solve(t(U) %*% U) %*% t(U))
  F77_CALL(dgemm)("N", "N", &N_var, &N_var, &N_sub, &blas_one, temp4, &N_var, X, &N_sub, &blas_zero, temp5, &N_var FCONE FCONE); //t(X) %*% ((1 - rho1) * diag(N.sub) + (rho1 - rho2) * U %*% solve(t(U) %*% U) %*% t(U)) %*% X
  F77_CALL(dgemm)("N", "N", &N_var, &N_comp1, &N_var, &blas_one, temp5, &N_var, A1, &N_var, &blas_zero, G1, &N_var FCONE FCONE); //t(X) %*% ((1 - rho1) * diag(N.sub) + (rho1 - rho2) * U %*% solve(t(U) %*% U) %*% t(U)) %*% X %*% A1

  /* G2����� */
  F77_CALL(dgemm)("T", "N", &N_var, &N_var, &N_sub, &blas_one, X, &N_sub, X, &N_sub, &blas_zero, temp5, &N_var FCONE FCONE); //t(X) %*% X
  F77_CALL(dgemm)("N", "N", &N_var, &N_comp2, &N_var, &blas_one, temp5, &N_var, A2, &N_var, &blas_zero, G2, &N_var FCONE FCONE); //t(X) %*% X %*% A2

  /* G����� */
  for (i = 0; i < N_var; i++) { //G = -2 * cbind(temp1, temp2)
  	for (j = 0; j < N_comp; j++) {
  	  if (j < N_comp1) {
	    	G[i + j * N_var] = G1[i + j * N_var];
	    } else {
    		G[i + j * N_var] = G2[i + (j - N_comp1) * N_var];
	    }
  	}
  }
  dscal_(&N_var_comp, &scal_G, G, &inc_dscal);

  // malloc�ǳ�����Ƥ�����γ���
  free(A1);
  free(A2);
  free(G1);
  free(G2);
  free(I_N);
  free(temp1);
  free(temp2);
  free(temp3);
  free(temp4);
  free(temp5);
  free(temp6);
}
