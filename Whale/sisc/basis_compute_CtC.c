

#include "math.h"
#include "mex.h"
#include "assert.h"
#include <stdio.h>
#include <string.h>
#include "sisc_lib.h"



const double* X_freq_real;
const double* X_freq_imag;
int patch_M;
int patch_N;
int num_channels;
int num_patches;

const double* s;
const mwIndex* s_sparse_rows;
const mwIndex* s_sparse_cols;
int* s_row;
int* s_col;
int* s_basis;
int* s_patch;
int s_num_elems;
int num_bases;

double* exp_real;
double* exp_imag;

mxArray* CtC_all;
double* CtC_allr;
double* CtC_alli;
mxArray* Ctd_all;
double* Ctd_allr;
double* Ctd_alli;

double c;

void get_CtC(){
  int row,col,chan,patch,elt,i,j;
  mxArray* Ct;
  double* Ct_real;
  double* Ct_imag;
  mxArray* d;
  double* d_real;
  double* d_imag;

  double stats1, stats2, stats3, stats4, stats5, stats6;
  int basis_elts;
  int row_elts;

  stats1 = stats2 = stats3 = stats4 = stats5 = stats6 = 0;
  
  basis_elts = num_bases*(num_bases+1)/2;
  row_elts = patch_M/2 + 1;

  Ct = mxCreateDoubleMatrix(num_bases,num_patches,mxCOMPLEX);
  Ct_real = mxGetPr(Ct);
  Ct_imag = mxGetPi(Ct);
  
  d = mxCreateDoubleMatrix(num_patches,1,mxCOMPLEX);
  d_real = mxGetPr(d);
  d_imag = mxGetPi(d);

  for (col = 0; col < patch_N; col++){
    for (row = 0; row <= patch_M/2; row++){
      mxArray* arguments[2];
      mxArray* CtC;
      mxArray* C;

      for (i=0; i<num_patches*num_bases; i++){
        Ct_real[i] = 0;
        Ct_imag[i] = 0;
      }
      
      for (elt=0; elt<s_num_elems; elt++){
        double real, imag;
        int ind;
        int temp;
        
        ind = (s_row[elt]*row)%patch_M+patch_M*((s_col[elt]*col)%patch_N);
        
        real = s[elt]*exp_real[ind];
        imag = s[elt]*exp_imag[ind];
        
        temp = s_patch[elt]*num_bases+s_basis[elt];

        Ct_real[temp] += real;
        Ct_imag[temp] -= imag;
      }

      arguments[0] = Ct;
      mexCallMATLAB(1,&C,1,arguments,"ctranspose");
      arguments[0] = Ct;
      arguments[1] = C;
      mexCallMATLAB(1,&CtC,2,arguments,"mtimes");

      /* Copy CtC into CtC_all */
      for (j = 0; j < num_bases; j++){
        for (i = 0; i <= j; i++){
          CtC_allr[col*row_elts*basis_elts + row*basis_elts + tri(i,j)] = mxGetPr(CtC)[j*num_bases + i];
        }
      }
      if (mxIsComplex(CtC)){
        for (j = 0; j < num_bases; j++){
          for (i = 0; i <= j; i++){
            CtC_alli[col*row_elts*basis_elts + row*basis_elts + tri(i,j)] = mxGetPi(CtC)[j*num_bases + i];
          }
        }
      } 

      for (chan=0; chan<num_channels; chan++){
        mxArray* Ctd;

        /* d = reshape(X_freq_all(row,col,chan,:),num_patches,1); */
        for (patch=0; patch<num_patches; patch++){
          int ind = patch*patch_M*patch_N*num_channels
            +chan*patch_M*patch_N
            +col*patch_M
            +row;
          
          d_real[patch] = X_freq_real[ind];
          d_imag[patch] = X_freq_imag[ind];
        }

        /* C'*d */
        arguments[0] = Ct;
        arguments[1] = d;
        mexCallMATLAB(1,&Ctd,2,arguments,"mtimes");

        /* Copy Ctd into Ctd_all */
        memcpy(&Ctd_allr[chan*num_bases + row*num_channels*num_bases + col*row_elts*num_channels*num_bases],
               mxGetPr(Ctd), num_bases*sizeof(double));
        if (mxIsComplex(Ctd)){
          memcpy(&Ctd_alli[chan*num_bases + row*num_channels*num_bases + col*row_elts*num_channels*num_bases],
                 mxGetPi(Ctd), num_bases*sizeof(double));
        }

        mxDestroyArray(Ctd);
      }

      mxDestroyArray(C);
      mxDestroyArray(CtC);
      
    }

  }

  mxDestroyArray(Ct);
  mxDestroyArray(d);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
  const mwSize* dims;
  mwSize dims_[4];
  int i;
  int s_patch_;
  char* X_freq_err;
  char* s_all_err;

  tisc_init();
  
  mxAssert(nrhs==4,"Expects four input arguments.\n");

  mxAssert(mxIsDouble(prhs[0]), "X_freq must be a double array.\n");
  X_freq_err = "X_freq must be patch_M x patch_N x num_channels x num_patches.\n";
  mxAssert(mxGetNumberOfDimensions(prhs[0])==4, X_freq_err);
  dims = mxGetDimensions(prhs[0]);
  patch_M = dims[0];
  mxAssert(patch_M%2 == 0, "X_freq must have an even number of rows.\n");
  patch_N = dims[1];
  num_channels = dims[2];
  num_patches = dims[3];
  X_freq_real = mxGetPr(prhs[0]);
  if (mxIsComplex(prhs[0]))
    X_freq_imag = mxGetPi(prhs[0]);
  else
    X_freq_imag = mxCalloc(patch_M*patch_N*num_channels*num_patches,sizeof(double));

  mxAssert(mxIsDouble(prhs[1]), "s_all must be a double array.\n");
  s_all_err = "s_all must be (patch_M * patch_N * num_bases) x num_patches.\n";
  mxAssert(mxIsSparse(prhs[1]),"s_all must be sparse.\n");
  mexPrintf("N = %d, should be %d\n", (int)mxGetN(prhs[1]), (int)num_patches);
  mxAssert(mxGetN(prhs[1])==num_patches,s_all_err);
  num_bases = mxGetM(prhs[1])/(patch_M*patch_N);
  s = mxGetPr(prhs[1]);
  s_sparse_rows = mxGetIr(prhs[1]);
  s_sparse_cols = mxGetJc(prhs[1]);
  s_num_elems = s_sparse_cols[num_patches];

  /* Precompute all the row and column numbers */
  s_row = mxCalloc(s_num_elems,sizeof(int));
  s_col = mxCalloc(s_num_elems,sizeof(int));
  s_basis = mxCalloc(s_num_elems,sizeof(int));
  s_patch = mxCalloc(s_num_elems,sizeof(int));
  s_patch_=0;
  for (i=0; i<s_num_elems; i++){
    while (i >= s_sparse_cols[s_patch_+1]){
      s_patch_++;
      mxAssert(s_patch_ <= num_patches,"Bad!");
    }

    s_basis[i] = s_sparse_rows[i]/(patch_M*patch_N);
    s_col[i] = (s_sparse_rows[i]%(patch_M*patch_N))/patch_M;
    s_row[i] = s_sparse_rows[i]%patch_M;
    s_patch[i] = s_patch_;

  }


  c = mxGetScalar(prhs[2]);

  mxAssert(mxGetM(prhs[3])==patch_M && mxGetN(prhs[3])==patch_N,"exps must be patch_M x patch_N.\n");
  mxAssert(mxIsComplex(prhs[3]),"exps must be complex.\n");
  exp_real = mxGetPr(prhs[3]);
  exp_imag = mxGetPi(prhs[3]);

  dims_[0] = num_bases*(num_bases+1)/2;
  dims_[1] = patch_M/2+1;
  dims_[2] = patch_N;
  plhs[0] = mxCreateNumericArray(3, dims_, mxDOUBLE_CLASS, mxCOMPLEX);
  CtC_all = plhs[0];
  CtC_allr = mxGetPr(CtC_all);
  CtC_alli = mxGetPi(CtC_all);

  dims_[0] = num_bases;
  dims_[1] = num_channels;
  dims_[2] = patch_M/2+1;
  dims_[3] = patch_N;
  plhs[1] = mxCreateNumericArray(4, dims_, mxDOUBLE_CLASS, mxCOMPLEX);
  Ctd_all = plhs[1];
  Ctd_allr = mxGetPr(Ctd_all);
  Ctd_alli = mxGetPi(Ctd_all);

  get_CtC();
  
  mxFree(s_row);
  mxFree(s_col);
  mxFree(s_basis);
  mxFree(s_patch);
}
