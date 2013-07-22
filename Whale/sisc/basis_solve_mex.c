

#include "math.h"
#include "mex.h"
#include "assert.h"
#include <stdio.h>
#include <string.h>
#include "sisc_lib.h"

int patch_M;
int patch_N;
int num_channels;
int num_bases;

const double* lambda;

double* A_freq_real;
double* A_freq_imag;

const mxArray* CtC_all;
const double* CtC_allr;
const double* CtC_alli;
const mxArray* Ctd_all;
const double* Ctd_allr;
const double* Ctd_alli;

void basis_solve(void)
{
  int row,col,chan,basis,elt,i,j;

  mxArray* CtC, *Ctd;
  
  double *zeros;
  int basis_elts, row_elts;
  double* CtCr, *CtCi;
  int symm;
  mxArray* ans;
  double *ans_real, *ans_imag;
  int ind, symm_row, symm_col;
  
  CtC = mxCreateDoubleMatrix(num_bases,num_bases,mxCOMPLEX);
  Ctd = mxCreateDoubleMatrix(num_bases,1,mxCOMPLEX);

  zeros = malloc(num_bases*num_bases*sizeof(double));
  for (elt=0; elt<num_bases*num_bases; elt++){
    zeros[elt]=0;
  }

  basis_elts = num_bases*(num_bases+1)/2;
  row_elts = patch_M/2+1;

  CtCr = mxGetPr(CtC);
  CtCi = mxGetPi(CtC);

  for (col=0; col<patch_N; col++){
    for (row=0; row<=patch_M/2; row++){
      mxArray* arguments[2];
      symm = (row > 0 && row < patch_M/2);

      /* Copy C'*C */
      for (j=0; j<num_bases; j++){
        for (i=0; i<=j; i++){
          CtCr[i*num_bases+j] = CtC_allr[row*basis_elts + col*row_elts*basis_elts + tri(i,j)];
          CtCr[j*num_bases+i] = CtC_allr[row*basis_elts + col*row_elts*basis_elts + tri(i,j)];
        }
      }
      if (mxIsComplex(CtC_all)){
        for (j=0; j<num_bases; j++){
          for (i=0; i<=j; i++){
            CtCi[i*num_bases+j] = -CtC_alli[row*basis_elts + col*row_elts*basis_elts + tri(i,j)];
            CtCi[j*num_bases+i] = CtC_alli[row*basis_elts + col*row_elts*basis_elts + tri(i,j)];
          }
        }
      } else {
        memcpy(CtCi, zeros, basis_elts*sizeof(double));
      }

      CtCr = mxGetPr(CtC);
      for (basis=0; basis<num_bases; basis++){
        CtCr[basis*num_bases+basis] += lambda[basis];
      }

      for (chan=0; chan<num_channels; chan++){
        /* d = reshape(X_freq_all(row,col,chan,:),num_patches,1); */

        /* Copy C'*d */
        memcpy(mxGetPr(Ctd), &Ctd_allr[chan*num_bases + row*num_bases*num_channels + col*row_elts*num_bases*num_channels],
               num_bases*sizeof(double));
        if (mxIsComplex(Ctd_all)){
          memcpy(mxGetPi(Ctd), &Ctd_alli[chan*num_bases + row*num_bases*num_channels + col*row_elts*num_bases*num_channels],
                 num_bases*sizeof(double));
        } else {
          memcpy(mxGetPi(Ctd), zeros, num_bases*sizeof(double));
        }

        /* A_freq(row,col,chan,:) = (C'*C+Lambda)\(C'*d); */
        arguments[0] = CtC;
        arguments[1] = Ctd;
        mexCallMATLAB(1,&ans,2,arguments,"mldivide");
        ans_real = mxGetPr(ans);
        if (mxIsComplex(ans)){
          ans_imag = mxGetPi(ans);
        } else {
          ans_imag = zeros;
        }

        for (basis=0; basis<num_bases; basis++){
          ind = basis*patch_M*patch_N*num_channels
                    +chan*patch_M*patch_N
                    +col*patch_M
                    +row;

          A_freq_real[ind] = ans_real[basis];
          A_freq_imag[ind] = ans_imag[basis];
          
          if (symm){
            symm_col = (col == 0) ? 0 : patch_N-col;
            symm_row = patch_M-row;
            ind = basis*patch_M*patch_N*num_channels
              +chan*patch_M*patch_N
              +symm_col*patch_M
              +symm_row;
            
            A_freq_real[ind] = ans_real[basis];
            A_freq_imag[ind] = -ans_imag[basis];
          }

        }

        mxDestroyArray(ans);
      }

    }
  }
  
  mxDestroyArray(CtC);
  mxDestroyArray(Ctd);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  const mwSize* dims;
  mwSize dims_[4];
  char *CtC_err, *Ctd_err;

  mxAssert(nlhs==1,"Expects one output argument.\n");
  mxAssert(nrhs==3,"Expects three input arguments.\n");

  mxAssert(mxIsDouble(prhs[0]), "CtC must be a double array.\n");
  CtC_err = "CtC must be num_bases*(num_bases+1)/2 x (patch_M/2 + 1) x patch_N.\n";
  CtC_all = prhs[0];
  dims = mxGetDimensions(CtC_all);
  num_bases = inv_tri(dims[0]);
  patch_M = 2*(dims[1]-1);
  if (mxGetNumberOfDimensions(CtC_all)==3){
    patch_N = dims[2];
  } else {
    patch_N = 1;
  }
  CtC_allr = mxGetPr(CtC_all);
  if (mxIsComplex(CtC_all)){
    CtC_alli = mxGetPi(CtC_all);
  }

  mxAssert(mxIsDouble(prhs[1]), "Ctd must be a double array.\n");
  Ctd_err = "Ctd must be num_bases x num_channels x (patch_M/2 + 1) x patch_N.\n";
  Ctd_all = prhs[1];
  dims = mxGetDimensions(Ctd_all);
  mxAssert(dims[0] == num_bases, Ctd_err);
  num_channels = dims[1];
  mxAssert(dims[2] == patch_M/2+1, Ctd_err);
  if (mxGetNumberOfDimensions(Ctd_all)==4){
    mxAssert(dims[3] == patch_N, Ctd_err);
  } else {
    mxAssert(1 == patch_N, Ctd_err);
  }
  Ctd_allr = mxGetPr(Ctd_all);
  if (mxIsComplex(Ctd_all)){
    Ctd_alli = mxGetPi(Ctd_all);
  }

  mexPrintf("patch_M %d num_bases %d\n", patch_M, num_bases);

  mxAssert(mxIsDouble(prhs[2]), "lambda must be a double array.\n");
  mxAssert(mxGetM(prhs[2])==num_bases && mxGetN(prhs[2])==1,"lambda must be num_bases x 1.\n");
  lambda = mxGetPr(prhs[2]);

  dims_[0] = patch_M;
  dims_[1] = patch_N;
  dims_[2] = num_channels;
  dims_[3] = num_bases;
  plhs[0] = mxCreateNumericArray(4,dims_,mxDOUBLE_CLASS,mxCOMPLEX);
  A_freq_real = mxGetPr(plhs[0]);
  A_freq_imag = mxGetPi(plhs[0]);

  basis_solve();

}


