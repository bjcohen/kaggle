

#include "math.h"
#include "mex.h"
#include "assert.h"
#include <stdio.h>
#include <string.h>
#include "sisc_lib.h"


int nargout;

int patch_M;
int patch_N;
int num_channels;
int num_patches;
int num_bases;

const mxArray* CtC_all;
const double* CtC_allr;
const double* CtC_alli;

const mxArray* Ctd_all;
const double* Ctd_allr;
const double* Ctd_alli;

const double* lambda;

double* obj;
double* grad;
double* hessian;

double c;

void get_objective(void)
{
  int row,col,chan,basis,i,j;

  double stats1, stats2, stats3, stats4, stats5, stats6, stats7, stats8;

  int basis_elts, row_elts;
  mxArray *CtC, *Ctd;
  double *CtCr, *CtCi;
  double *zeros;
  int mult, symm;
  
  stats1 = stats2 = stats3 = stats4 = stats5 = stats6 = stats7 = stats8 = 0;
  
  basis_elts = num_bases*(num_bases+1)/2;
  row_elts = patch_M/2 + 1;

  CtC = mxCreateDoubleMatrix(num_bases,num_bases,mxCOMPLEX);
  Ctd = mxCreateDoubleMatrix(num_bases,1,mxCOMPLEX);
  CtCr = mxGetPr(CtC);
  CtCi = mxGetPi(CtC);
  
  zeros = malloc(num_bases * num_bases * sizeof(double));
  for (i=0; i<num_bases*num_bases; i++){
    zeros[i] = 0;
  }

  for (col=0; col<patch_N; col++){
    for (row=0; row<=patch_M/2; row++){
      mxArray* arguments[2];
      mxArray* Minv;
      mult = (row == 0 || row == patch_M/2) ? 1 : 2;
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

      /* C'*C + Lambda */
      CtCr = mxGetPr(CtC);
      for (basis=0; basis<num_bases; basis++){
        CtCr[basis*num_bases+basis] += lambda[basis];
      }

      /* Minv = inv(C'*C+diag(lambda)) */
      mexCallMATLAB(1,&Minv,1,&CtC,"inv");

      for (chan=0; chan<num_channels; chan++){
        mxArray* MinvCtd;
        double *MinvCtdr, *MinvCtdi, *Ctdr, *Ctdi;
        int MinvCtdIsComplex;

        /* Copy C'*d */
        memcpy(mxGetPr(Ctd), &Ctd_allr[chan*num_bases + row*num_bases*num_channels + col*row_elts*num_bases*num_channels],
               num_bases*sizeof(double));
        if (mxIsComplex(Ctd_all)){
          memcpy(mxGetPi(Ctd), &Ctd_alli[chan*num_bases + row*num_bases*num_channels + col*row_elts*num_bases*num_channels],
                 num_bases*sizeof(double));
        } else {
          memcpy(mxGetPi(Ctd), zeros, num_bases*sizeof(double));
        }
        Ctdr = mxGetPr(Ctd);
        Ctdi = mxGetPi(Ctd);
        
        /* Minv*C'*d */
        arguments[0] = Minv;
        arguments[1] = Ctd;
        mexCallMATLAB(1,&MinvCtd,2,arguments,"mtimes");

        MinvCtdr = mxGetPr(MinvCtd);
        MinvCtdIsComplex = mxIsComplex(MinvCtd);
        if (MinvCtdIsComplex){
          MinvCtdi = mxGetPi(MinvCtd);
        }

        /* obj += d'*C*Minv*C'*d */
        for (i=0; i<num_bases; i++){
          *obj += mult*MinvCtdr[i] * Ctdr[i];
          if (MinvCtdIsComplex){
            *obj += mult*MinvCtdi[i] * Ctdi[i];
          }
        }

        if (nargout >= 2){
          for (i=0; i<num_bases; i++){
            grad[i] -= mult*MinvCtdr[i]*MinvCtdr[i];
            if (MinvCtdIsComplex)
              grad[i] -= mult*MinvCtdi[i]*MinvCtdi[i];
          }
        }

        if (nargout >= 3){
          
          for (i=0; i<num_bases; i++){
            for (j=0; j<=i; j++){
              double temp1r = 0;
              double temp1i = 0;
              double temph = 0;
              temp1r += MinvCtdr[i]*MinvCtdr[j];
              if (MinvCtdIsComplex){
                temp1i -= MinvCtdr[i]*MinvCtdi[j];
                temp1i += MinvCtdi[i]*MinvCtdr[j];
                temp1r += MinvCtdi[i]*MinvCtdi[j];
              }

              
              temph += 2*temp1r*mxGetPr(Minv)[i*num_bases+j];
              if (mxIsComplex(Minv) && MinvCtdIsComplex){
                temph -= 2*temp1i*mxGetPi(Minv)[i*num_bases+j];
              }
              hessian[i*num_bases+j] += mult*temph;
              if (i != j){
                hessian[j*num_bases+i] += mult*temph;
              }

            }
          }
          
        }
        
        
        mxDestroyArray(MinvCtd);
      }

      mxDestroyArray(Minv);
    }

  }
    
  for (i=0; i<num_bases; i++){
    *obj += lambda[i]*c;
    if (nargout >= 2)
      grad[i] += c;
  }

}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  const mwSize* dims;
  char* CtC_err, *Ctd_err;

  nargout = nlhs;
  mxAssert(nrhs==4,"Expects four input arguments.\n");
  
  mxAssert(mxIsDouble(prhs[0]), "CtC_all must be a double array.\n");
  CtC_err = "CtC must be num_bases x num_bases x patch_M x patch_N.\n";
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
  Ctd_err = "Ctd must be num_bases x num_channels x patch_M x patch_N.\n";
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

  mxAssert(mxIsDouble(prhs[2]), "lambda must be a double array.\n");
  mxAssert(mxGetM(prhs[2])==num_bases && mxGetN(prhs[2])==1,"lambda must be num_bases x 1.\n");
  lambda = mxGetPr(prhs[2]);

  c = mxGetScalar(prhs[3]);

  plhs[0] = mxCreateDoubleScalar(0);
  obj = mxGetPr(plhs[0]);

  if (nargout >= 2){
    plhs[1] = mxCreateDoubleMatrix(num_bases,1,mxREAL);
    grad = mxGetPr(plhs[1]);
  }

  if (nargout >= 3){
    plhs[2] = mxCreateDoubleMatrix(num_bases,num_bases,mxREAL);
    hessian = mxGetPr(plhs[2]);
  }

  get_objective();
  
}
