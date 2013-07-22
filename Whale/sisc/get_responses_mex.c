#include "math.h"
#include "mex.h"
#include <stdio.h>
#include <string.h>
#include "sisc_lib.h"

int patch_M;
int patch_N;
int num_channels;
int basis_M;
int basis_N;
int num_bases;

const mxArray* X;
double* X_real;
double* A_real;
const mxArray* A;
const mxArray* A_freq;
double* A_freq_real;
double* A_freq_imag;
mxArray* S;
double* S_real;

const mxArray* AtA;
const double* AtA_real;

int max_iter;
double beta;
int exact;
int num_coords;
int verbosity;

mxArray* dummy_real;
mxArray* dummy_real2;
mxArray* dummy_real3;
mxArray* dummy_complex;
double* zeros;

dPrintf(char* str){
  /*mexPrintf(str);*/
  return 0;
}

const int CLOCK_REC = 0;
const int CLOCK_CORR_FREQ = 1;
const int CLOCK_CORR = 2;
const int CLOCK_QUAD = 3;
const int CLOCK_SOLVE = 4;
const int CLOCK_ALL = 5;
bool keep_time;
const bool RESET_TIME = true;

bool compute_stats;
double* times_all;
double* fobj_all;

int* get_active_set(const double* corr, int n, double beta, int *num_active)
{
  int i,j;
  double max_elt;
  double cutoff;
  double delta;
  int overflow;
  int num_nonzero = 0;
  int* elems = mxCalloc(patch_M*patch_N*num_bases,sizeof(int));
  int *active_set;
  int r, c, b;

  /* Find the maximum value, and exit if it is zero. */
  max_elt = 0;
  for (r = 0; r <= patch_M - basis_M; r++){
    for (c = 0; c <= patch_N - basis_N; c++){
      for (b=0; b<num_bases; b++){
        i = b*patch_M*patch_N + c*patch_M + r;
        if (2*absdbl(corr[i]) > beta){
          elems[num_nonzero++] = i;
          /*mexPrintf("active_set row %d col %d basis %d\n", r, c, b);*/
          if (absdbl(corr[i]) > max_elt){
            max_elt = absdbl(corr[i]);
          }
        }
      }
    }
  }

  /* If the number of nonzero terms is no bigger than n,
     simply return that set. */
  if (num_nonzero <= n){
    *num_active = num_nonzero;
    /*mexPrintf("here... num_active: %d\n", *num_active);*/
    return elems;
  }

  active_set = mxCalloc(num_nonzero, sizeof(int));
  
  /* Do a binary search for a good value */
  cutoff = max_elt/2;
  delta = max_elt/4;
  for (j=1; j<30; j++){
    *num_active = 0;
    overflow = 0;

    for (i=0; i<num_nonzero; i++){
      if (absdbl(corr[elems[i]]) > cutoff){
        if (*num_active < n){
          active_set[*num_active] = elems[i];
          (*num_active)++;
        } else {
          overflow = 1;
          break;
        }
      }
    }

    if (overflow){
      cutoff += delta;
      delta /= 2;
    } else if (*num_active<=n && *num_active>=n/2) {
      break;
    } else {
      cutoff -= delta;
      delta /= 2;
    }
  }

  /*mexPrintf("num_active: %d\n", *num_active);*/

  return elems;
}


int* get_active_set_graft(const double* S_real, const double* corr, int n, double beta, int *num_active)
{
  double* dummy;
  int i,curr;
  int* active_set_zero;
  int num_nonzero;
  int* active_set_all;
  int npicked;
  int num_elems;
  
  num_nonzero = 0;

  /* Find the zero coefficients with the largest gradients. */
  num_elems = patch_M*patch_N*num_bases;
  dummy = mxCalloc(num_elems,sizeof(double));
  memcpy(dummy,corr,num_elems*sizeof(double));
  for (i=0; i<num_elems; i++){
    if (S_real[i] != 0){
      dummy[i] = 0;
      num_nonzero++;
    }
  }
  active_set_zero = get_active_set(dummy,n,beta,&npicked);

  active_set_all = mxCalloc(npicked+num_nonzero, sizeof(int));
  for (i=0; i<npicked; i++)
    active_set_all[i] = active_set_zero[i];
  
  curr = npicked;
  for (i=0; i<num_elems; i++)
    if (S_real[i] != 0)
      active_set_all[curr++] = i;

  *num_active = npicked + num_nonzero;

  /*mexPrintf("there... num_active: %d\n", *num_active);*/

  mxFree(dummy);
  return active_set_all;
}

double* get_quadratic(const int *active_set, int num_active)
{
  int i;
  int j;

  int AtA_M = 2*basis_M-1;
  int AtA_N = 2*basis_N-1;
  
  int ai, r1, c1, b1;
  double *fs_A;
  int aj, r2, c2, b2;
  double val;
  int temp1, temp2;
  
  fs_A = mxCalloc(num_active*num_active, sizeof(double));

  for (i=0; i<num_active; i++){
    ai = active_set[i];
    r1 = ai%patch_M;
    c1 = (ai/patch_M)%patch_N;
    b1 = ai/(patch_M*patch_N);

    /*mexPrintf("get_quadratic row %d col %d basis %d\n", r1, c1, b1);*/

    for (j=0; j<=i; j++){
      aj = active_set[j];
      r2 = aj%patch_M;
      c2 = (aj/patch_M)%patch_N;
      b2 = aj/(patch_M*patch_N);

      temp1 = r2-r1+basis_M-1;
      temp2 = c2-c1+basis_N-1;
      if (temp1 >= 0 && temp1 < AtA_M && temp2 >= 0 && temp2 < AtA_N){
        if (b1 <= b2){
          val = AtA_real[AtA_M*AtA_N*tri(b1,b2) +
                         AtA_M*(c2-c1+basis_N-1) +
                         (r2-r1+basis_M-1)];
        } else {
          val = AtA_real[AtA_M*AtA_N*tri(b2,b1) +
                         AtA_M*(c1-c2+basis_N-1) +
                         (r1-r2+basis_M-1)];
        }
      } else {
        val = 0;
      }
      
      fs_A[i*num_active+j] = val;
      fs_A[j*num_active+i] = val;     
    }
  }

  return fs_A;
}


mxArray* solve_fs(const mxArray* fs_A, const mxArray* fs_b, const mxArray* fs_x, double beta, int num_active)
{
  mxArray* result;
  int error;
  const mxArray* args[4];
  int i;
  
  args[0] = fs_A;
  args[1] = fs_b;
  args[2] = fs_x;
  args[3] = mxCreateDoubleMatrix(num_active,1,mxREAL);
  for (i=0; i<num_active; i++){
    mxGetPr(args[3])[i] = beta;
  }
  error = mexCallMATLAB(1,&result,4,(mxArray**)args,"solve_fs");
  mxAssert(!error,"Something wrong in solve_fs.");
  mxDestroyArray((mxArray*)args[3]);
  return result;
}


void get_responses()
{
  int i, j, k, basis, chan, elt;
  double *err_real, *err_imag;
  mxArray* err;
  mwSize dims[3];
  int finished = 0;
  int iter;
  mxArray *rec;
  double *rec_real, *rec_imag;
  int ind;
  double fres, fspars, fobj;
  double *corr_real, *corr_imag;
  int *active_set;
  int num_active;
  int sh_count;
  int num_nonzero;
  int c, r, b;
  mxArray* fs_x_init;
  mxArray* fs_b;
  double *fs_A_;
  mxArray *fs_A;
  mxArray* corr;
  mxArray* args[2];
  mxArray* temp;
  mxArray *fs_x;
    
  if (keep_time){
    if (RESET_TIME){
      reset_clock(CLOCK_REC);
      reset_clock(CLOCK_CORR_FREQ);
      reset_clock(CLOCK_CORR);
      reset_clock(CLOCK_QUAD);
      reset_clock(CLOCK_SOLVE);
    }
  }

/*mexPrintf("A\n");*/
  
  if (compute_stats){
    reset_clock(CLOCK_ALL);
    start_clock(CLOCK_ALL);
  }

  for (iter=1; iter<=max_iter; iter++){
    if (keep_time){
      start_clock(CLOCK_REC);
    }
    if (compute_stats){
      times_all[iter-1] = clock_time(CLOCK_ALL);
    }

    /* Compute the reconstruction. */
    rec = get_reconstruction(A_freq, S);
    get_real_imag(&rec_real, &rec_imag, rec);
    
    /* Get the residual. */
    dims[0] = patch_M;
    dims[1] = patch_N;
    dims[2] = num_channels;
    err = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX);
    get_real_imag(&err_real, &err_imag, err);
    for (chan = 0; chan < num_channels; chan++){
      for (elt = 0; elt < patch_M*patch_N; elt++){
        ind = chan*patch_M*patch_N + elt;
        err_real[ind] = X_real[ind] - rec_real[ind];
      }
    }

    /*mexPrintf("B\n");*/
    
    /* Compute the objective function. */
    if (compute_stats || verbosity >= 2){
      get_objective_function(&fres, &fspars, &fobj, err, S, beta);
      
      if (verbosity >= 2)
        mexPrintf("fres = %1.15f, fspars = %1.15f, fobj = %1.15f\n", 
          fres, fspars, fobj);
      
      if (compute_stats)
        fobj_all[iter-1] = fobj;
    }

    if (keep_time){
      stop_clock(CLOCK_REC);
      start_clock(CLOCK_CORR);
    }
    
    /*mexPrintf("C\n");*/

    /* Compute the correlations of bases with the error. */
    corr = get_correlation(A_freq, err);
    get_real_imag(&corr_real, &corr_imag, corr);
    /*mexPrintf("C1\n");*/
    if (keep_time){
      stop_clock(CLOCK_CORR);
      start_clock(CLOCK_QUAD);
    }
    /*mexPrintf("C2\n");*/
    sh_count = 0;
    num_nonzero = 0;
    for (c = 0; c <= patch_N - basis_N; c++){
      for (r = 0; r <= patch_M - basis_M; r++){
        for (b = 0; b < num_bases; b++){
          int i = b*patch_M*patch_N + c*patch_M + r;
          if ((2*corr_real[i]>beta || 2*corr_real[i]<-beta) && S_real[i]==0)
            sh_count++;
          if (S_real[i] != 0)
            num_nonzero++;
        }
      }
    }

    /*mexPrintf("D\n");*/

    /* Compute the active set. If we're in exact mode, use grafting,
       otherwise use coordinate descent. */
    if (exact){
      active_set = get_active_set_graft(S_real,corr_real,num_coords,beta,&num_active);
    } else {
      active_set = get_active_set(corr_real,num_coords,beta,&num_active);
    }

    /*mexPrintf("D1\n");*/

    /* Solve for those coefficients. */
    fs_x_init = mxCreateDoubleMatrix(num_active,1,mxREAL);
    fs_b = mxCreateDoubleMatrix(num_active,1,mxREAL);
    for (i=0; i<num_active; i++){
      ind = active_set[i];
      mxGetPr(fs_x_init)[i] = S_real[ind];
      mxGetPr(fs_b)[i] = -2*corr_real[ind];
    }
    
    /*mexPrintf("E\n");*/

    if (verbosity >= 2)
      mexPrintf("Active Set: %4d, Number Nonzero: %4d, Zero with Nonzero Gradient: %4d\n", num_active, num_nonzero, sh_count);

    fs_A_ = get_quadratic(active_set, num_active);
    fs_A = mxCreateDoubleMatrix(num_active, num_active, mxREAL);
    memcpy(mxGetPr(fs_A), fs_A_, num_active*num_active*sizeof(double));

    if (keep_time){
      stop_clock(CLOCK_QUAD);
      start_clock(CLOCK_SOLVE);
    }

    args[0] = fs_A;
    args[1] = fs_x_init;
    mexCallMATLAB(1,&temp,2,args,"mtimes");
    for (i=0; i<num_active; i++)
      mxGetPr(fs_b)[i] -= 2*mxGetPr(temp)[i];

    fs_x = solve_fs(fs_A,fs_b,fs_x_init,beta,num_active);

    for (i=0; i<num_active; i++){
      ind = active_set[i];
      S_real[ind] = mxGetPr(fs_x)[i];
    }

    if (keep_time){
      stop_clock(CLOCK_SOLVE);
    }

    /*mexPrintf("F\n");*/

    mxDestroyArray(rec);
    mxDestroyArray(err);
    mxDestroyArray(corr);
    mxDestroyArray(fs_A);
    mxDestroyArray(fs_x);
    mxDestroyArray(fs_x_init);
    mxDestroyArray(fs_b);
    mxFree(active_set);
    mxFree(fs_A_);

    if (exact && sh_count == 0){
      finished = 1;
      break;
    }
    
  }
  if (verbosity >= 1){
    if (finished)
      mexPrintf("Found exact solution in %d iterations.\n", iter);
    else
      mexPrintf("Stopped after %d iterations.\n",iter-1);
    if (keep_time){
      mexPrintf("Elapsed time:\n");
      print_time(CLOCK_REC, "Reconstruction");
      print_time(CLOCK_CORR_FREQ, "Correlation, frequency");
      print_time(CLOCK_CORR, "Correlation");
      print_time(CLOCK_QUAD, "Quadratic");
      print_time(CLOCK_SOLVE, "Feature Sign");
    }
  }
  if (compute_stats){
    times_all[iter] = clock_time(CLOCK_ALL);
    stop_clock(CLOCK_ALL);
    
    /* Compute the reconstruction. */
    rec = get_reconstruction(A_freq, S);
    rec_real = mxGetPr(rec);
    
    /* Get the residual. */
    dims[0] = patch_M;
    dims[1] = patch_N;
    dims[2] = num_channels;
    err = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX);
    err_real = mxGetPr(err);
    for (chan = 0; chan < num_channels; chan++){
      for (elt = 0; elt < patch_M*patch_N; elt++){
        ind = chan*patch_M*patch_N + elt;
        err_real[ind] = X_real[ind] - rec_real[ind];
      }
    }

    get_objective_function(&fres, &fspars, &fobj, err, S, beta);
    fobj_all[iter] = fobj;
    
    mxDestroyArray(rec);
    mxDestroyArray(err);
  }
  
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  const mwSize* dims;
  mwSize dims_[3];
  const double* dummy_p_old1, *dummy_p_old2, *dummy_p_old3;
  int i,j, k;
  const double* S_old;

  tisc_init();
  
  mxAssert(nlhs==1 || nlhs==3,"Expects one or three output arguments.\n");
  mxAssert(nrhs==10,"Expects 10 input arguments.\n");

  mxAssert(mxIsDouble(prhs[0]), "get_responses_mex: X must be a double array.\n");
  mxAssert(mxGetNumberOfDimensions(prhs[0])==2||mxGetNumberOfDimensions(prhs[0])==3,
           "Image must be 2 or 3 dimensions.");
  dims = mxGetDimensions(prhs[0]);
  patch_M = dims[0];
  patch_N = dims[1];
  if (mxGetNumberOfDimensions(prhs[0])==3)
    num_channels = dims[2];
  else
    num_channels = 1;
  X_real = mxGetPr(prhs[0]);
  X = prhs[0];

  mxAssert(mxIsDouble(prhs[1]), "get_responses_mex: A must be a double array.\n");
  mxAssert(mxGetNumberOfDimensions(prhs[1])==4,"Bases must be 4 dimensions.");
  dims = mxGetDimensions(prhs[1]);
  basis_M = dims[0];
  basis_N = dims[1];
  mxAssert(dims[2]==num_channels,"Dimension 3 of bases must equal num_channels.");
  num_bases = dims[3];
  A_real = mxGetPr(prhs[1]);
  A = prhs[1];

  mxAssert(mxIsDouble(prhs[2]), "get_responses_mex: A_freq must be a double array.\n");
  mxAssert(mxGetNumberOfDimensions(prhs[2])==4,"A_freq must be 4 dimensions.");
  dims = mxGetDimensions(prhs[2]);
  mxAssert(dims[0]==patch_M,"Dimension 1 of A_freq must equal patch_M.");
  mxAssert(dims[1]==patch_N,"Dimension 2 of A_freq must equal patch_N.");
  mxAssert(dims[2]==num_channels,"Dimension 3 of bases must equal num_channels.");
  mxAssert(dims[3]==num_bases,"Dimension 4 of bases must equal num_bases.");
  A_freq = prhs[2];
  get_real_imag(&A_freq_real, &A_freq_imag, A_freq);

  mxAssert(patch_M>0 && patch_N>0, "Bad dimensions.");

  max_iter = (int)(mxGetScalar(prhs[3]));
  
  beta = mxGetScalar(prhs[4]);

  exact = (int)(mxGetScalar(prhs[5]));

  num_coords = (int)(mxGetScalar(prhs[6]));

  verbosity = (int)(mxGetScalar(prhs[7]));
  keep_time = (verbosity >= 2);

  dims_[0] = patch_M;
  dims_[1] = patch_N;
  dims_[2] = num_bases;

  plhs[0] = mxCreateNumericArray(3,dims_,mxDOUBLE_CLASS,mxREAL);
  S = plhs[0];
  S_real = mxGetPr(S);

  S_old = mxGetPr(prhs[8]);

  AtA = prhs[9];
  AtA_real = mxGetPr(AtA);

  memcpy(S_real,S_old,patch_M*patch_N*num_bases*sizeof(double));

  zeros = mxCalloc(patch_M*patch_N*num_bases,sizeof(double));
  for (i=0; i<patch_M*patch_N*num_bases; i++)
    zeros[i] = 0;
  
  compute_stats = (nlhs==3);
  if (compute_stats){
    plhs[1] = mxCreateDoubleMatrix(max_iter+1, 1, mxREAL);
    times_all = mxGetPr(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(max_iter+1, 1, mxREAL);
    fobj_all = mxGetPr(plhs[2]);
  }

  get_responses();


}


