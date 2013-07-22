#include "mex.h"
#include "sisc_lib.h"


mxArray** t0; /* MATLAB structure containing the time at which CLOCK was called for this clock. */ 
double* prev_time; /* Saved total of time taken by all
    invocations of this clock. Does not count time since last invocation of CLOCK. */ 
bool* running; /* Is the clock currently running? */ 
int num_clocks;

double* zeros;
int zeros_len;

bool trace_first = true;
bool doTrace[NUM_TRACE];

void tisc_init(void){
  int i;
  t0 = NULL;
  prev_time = NULL;
  running = NULL;
  num_clocks = 0;
  zeros = NULL;
  zeros_len = 0;
  for (i = 0; i < NUM_TRACE; i++){
    doTrace[i] = false;
  }
}

void tisc_destroy(void){
  if (t0 != NULL)
    mxFree(t0);
  if (prev_time != NULL)
    mxFree(prev_time);
  if (running != NULL)
    mxFree(running);
  if (zeros != NULL)
    mxFree(zeros);
}

int max(int a, int b){
  return((a>b) ? a : b);
}

int min(int a, int b){
  return((a<b) ? a : b);
}

double maxdbl(double a, double b){
  return ((a>b) ? a : b);
}

double mindbl(double a, double b){
  return ((a<b) ? a : b);
}

double absdbl(double d){
  return d>0 ? d : -d;
}

/* Print an MxN array of doubles. */
void print_double_array(const double* values, int M, int N)
{
  int i,j;
  for (i=0; i<M; i++){
    mexPrintf("    ");
    for (j=0; j<N; j++){
      mexPrintf("%5.5f ", values[j*M+i]);
    }
    mexPrintf("\n");
  }
  mexPrintf("\n");
}

/* Print a MATLAB array. */
void print_mex_array(const mxArray* array){
  int M, N;
  M = mxGetM(array);
  N = mxGetN(array);
  mexPrintf("  (real)\n");
  print_double_array(mxGetPr(array),M,N);

  if (mxIsComplex(array)){
    mexPrintf("  (complex)\n");
    print_double_array(mxGetPi(array),M,N);
  }
  mexPrintf("\n");
}

/* Multiply two complex numbers. */
void times(double* a_real, double* a_imag, double b_real, double b_imag, double c_real, double c_imag)
{
  *a_real = b_real*c_real - b_imag*c_imag;
  *a_imag = b_real*c_imag + b_imag*c_real;
}

/* Compute the jth triangular number and add i. */
int tri(int i, int j){
  return i + j*(j+1)/2;
}

/* Inverse of the triangular number function. Fails if input is not a 
   triangular number. */
int inv_tri(int i){
  int g = 0;
  int ind;
  for (ind = 0; ind < 1000; ind++){
    g += ind;
    if (g == i){
      return ind;
    } else if (g > i) {
      mxAssert(0, "Error in inv_tri");
    }
  }
  mxAssert(0, "Error in inv_tri");
  return -1;
}

/* Compute the FFT of C. */
mxArray* fft2(const mxArray* C)
{
  mxArray* result;
  int error;
  
  tracepoint(TRACE_LIB, "fft2 1\n");

  error = mexCallMATLAB(1,&result,1,(mxArray**)&C,"fft2");
  mxAssert(!error,"Something wrong in fft2.\n");
  tracepoint(TRACE_LIB, "fft2 2\n");
  return result;  
}

/* Compute the IFFT of C. */
mxArray* ifft2(const mxArray* C)
{
  mxArray* result;
  int error;
  
  tracepoint(TRACE_LIB, "ifft2 1\n");
  
  error = mexCallMATLAB(1,&result,1,(mxArray**)&C,"ifft2");
  mxAssert(!error,"Something wrong in ifft2.\n");
  tracepoint(TRACE_LIB, "ifft2 2\n");
  return result;
}

/* Return an array of some number of zeros. Assumes contents will
   never be modified. */
double* alloc_zeros(unsigned len){
  int i;
  if (zeros == NULL || len > zeros_len){
    if (zeros == NULL)
      zeros = mxCalloc(len, sizeof(double));
    else
      zeros = mxRealloc(zeros, len*sizeof(double));
    
    for (i = zeros_len; i < len; i++){
      zeros[i] = 0; 
    }
    
    zeros_len = len;
  }
  
  return zeros;
}

/* Get pointers to the real and imaginary parts of a MATLAB
   array, replacing the imaginary part with zeros if the array
   is real-valued. */
void get_real_imag(double** real, double** imag, const mxArray* arr){
  *real = mxGetPr(arr);
  if (mxIsComplex(arr))
    *imag = mxGetPi(arr);
  else
    *imag = alloc_zeros(mxGetNumberOfElements(arr));
}

/* Compute the reconstruction given Fourier domain bases A_freq
   and spatial domain coefficients S */
mxArray* get_reconstruction(const mxArray* A_freq, const mxArray* S){
  int frame_M, frame_N, ndim, num_channels, num_bases;
  double* S_freq_real, *S_freq_imag;
  double* A_freq_real, *A_freq_imag;
  mxArray* S_freq;
  const mwSize* dims;
  mxArray* rec_freq;
  double* rec_freq_real, *rec_freq_imag;
  mwSize dims_[3];
  int chan, basis, elt;
  int rec_ind, S_ind, A_ind;
  mxArray* rec;
  
  tracepoint(TRACE_LIB, "get_reconstruction 1\n");
  
  /* Get the dimensions */
  dims = mxGetDimensions(S);
  frame_M = dims[0];
  frame_N = dims[1];
  ndim = mxGetNumberOfDimensions(A_freq);
  mxAssert(ndim == 4, "get_reconstruction: number of dimensions of A must equal 4.");
  dims = mxGetDimensions(A_freq);
  mxAssert(dims[0] == frame_M, "A_freq must be frame_M x frame_N x num_channels x num_bases");
  mxAssert(dims[1] == frame_N, "A_freq must be frame_M x frame_N x num_channels x num_bases");
  num_channels = dims[2];
  num_bases = dims[3];
  
  /* Compute FFT of S */
  S_freq = fft2(S);
  get_real_imag(&S_freq_real, &S_freq_imag, S_freq);
  get_real_imag(&A_freq_real, &A_freq_imag, A_freq);
  
  tracepoint(TRACE_LIB, "get_reconstruction 2\n");
  
  dims_[0] = frame_M;
  dims_[1] = frame_N;
  dims_[2] = num_channels;
  rec_freq = mxCreateNumericArray(3, dims_, mxDOUBLE_CLASS, mxCOMPLEX);
  get_real_imag(&rec_freq_real, &rec_freq_imag, rec_freq);
  
  tracepoint(TRACE_LIB, "get_reconstruction 3\n");
  
  /* Compute the reconstruction in the Fourier domain */

  for (chan = 0; chan < num_channels; chan++){
    for (basis = 0; basis < num_bases; basis++){
      for (elt = 0; elt < frame_M*frame_N; elt++){
        rec_ind = chan*frame_M*frame_N + elt;
        S_ind = basis*frame_M*frame_N + elt;
        A_ind = basis*frame_M*frame_N*num_channels + 
          chan*frame_M*frame_N + elt;
          
        rec_freq_real[rec_ind] += (S_freq_real[S_ind]*A_freq_real[A_ind]
            - S_freq_imag[S_ind]*A_freq_imag[A_ind]);
        rec_freq_imag[rec_ind] += (S_freq_real[S_ind]*A_freq_imag[A_ind]
            + S_freq_imag[S_ind]*A_freq_real[A_ind]);
      }
    }
  }
  
  tracepoint(TRACE_LIB, "get_reconstruction 4\n");
  
  /* Convert back into spatial domain */
  rec = ifft2(rec_freq);
  mxDestroyArray(rec_freq);
  mxDestroyArray(S_freq);
  tracepoint(TRACE_LIB, "get_reconstruction 5\n");
  return rec;
}

void get_objective_function(double* fres, double* fspars, double* fobj,
const mxArray* err, const mxArray* S, double beta){
  int err_len, S_len, elt;
  const double* err_real, *S_real;
  *fres = *fspars = *fobj = 0;
  
  err_len = mxGetNumberOfElements(err);
  S_len = mxGetNumberOfElements(S);
  err_real = mxGetPr(err);
  S_real = mxGetPr(S);
  
  for (elt = 0; elt < err_len; elt++)
        *fres += err_real[elt] * err_real[elt];
      
  for (elt = 0; elt < S_len; elt++)
        *fspars += beta*absdbl(S_real[elt]);
      
  *fobj = *fres + *fspars;
}

/* Compute the correlation matrix for bases A_freq (in Fourier
   space) and the error term err. */
mxArray* get_correlation(const mxArray* A_freq, const mxArray* err){
  const mwSize* dims;
  int frame_M, frame_N, ndim, num_channels, num_bases;
  double *A_freq_real, *A_freq_imag;
  double* corr_freq_real, *corr_freq_imag;
  mxArray* err_freq, *corr_freq, *corr;
  double *err_freq_real, *err_freq_imag;
  int basis, chan, elt;
  mwSize dims_[3];

  tracepoint(TRACE_LIB, "get_correlation 1\n");
  
  /* Get the dimensions */
  dims = mxGetDimensions(err);
  frame_M = dims[0];
  frame_N = dims[1];
  ndim = mxGetNumberOfDimensions(A_freq);
  mxAssert(ndim == 4, "get_reconstruction: number of dimensions of A must equal 4.");
  dims = mxGetDimensions(A_freq);
  mxAssert(dims[0] == frame_M, "A_freq must be frame_M x frame_N x num_channels x num_bases");
  mxAssert(dims[1] == frame_N, "A_freq must be frame_M x frame_N x num_channels x num_bases");
  num_channels = dims[2];
  num_bases = dims[3];
  
  tracepoint(TRACE_LIB, "get_correlation 2\n");
  
  get_real_imag(&A_freq_real, &A_freq_imag, A_freq);
  
  dims_[0] = frame_M;
  dims_[1] = frame_N;
  dims_[2] = num_bases;
  corr_freq = mxCreateNumericArray(3,dims_,mxDOUBLE_CLASS,mxCOMPLEX);
  get_real_imag(&corr_freq_real, &corr_freq_imag, corr_freq);
  
  tracepoint(TRACE_LIB, "get_correlation 3\n");

  /* Compute FFT of error */
  err_freq = fft2(err);
  get_real_imag(&err_freq_real, &err_freq_imag, err_freq);
  
  tracepoint(TRACE_LIB, "get_correlation 4\n");
  
  /* Compute the correlation in Fourier space. */
  for (basis=0; basis<num_bases; basis++){
    for (chan=0; chan<num_channels; chan++){
      for (elt=0; elt<frame_M*frame_N; elt++){
        int corr_ind, err_ind, A_ind;
        corr_ind = basis*frame_M*frame_N + elt;
        err_ind = chan*frame_M*frame_N + elt;
        A_ind = basis*frame_M*frame_N*num_channels + chan*frame_M*frame_N + elt;

        corr_freq_real[corr_ind] += (err_freq_real[err_ind]*A_freq_real[A_ind] + err_freq_imag[err_ind]*A_freq_imag[A_ind]);
        corr_freq_imag[corr_ind] += (-err_freq_real[err_ind]*A_freq_imag[A_ind] + err_freq_imag[err_ind]*A_freq_real[A_ind]);
      }
    }
  }
  
  tracepoint(TRACE_LIB, "get_correlation 5\n");
  
  /* Compute IFFT of frequency domain correlations */
  corr = ifft2(corr_freq);
  mxDestroyArray(corr_freq);
  mxDestroyArray(err_freq);
  tracepoint(TRACE_LIB, "get_correlation 6\n");
  return corr;
  
}


void allocate_clocks(int num){
  int i;
  if (num > num_clocks){
    if (t0 == NULL){
      mxAssert(prev_time == NULL, "allocate_clocks: prev_time != NULL");
      mxAssert(running == NULL, "allocate_clocks: running != NULL");
      t0 = mxCalloc(num, sizeof(mxArray*));
      prev_time = mxCalloc(num, sizeof(double));
      running = mxCalloc(num, sizeof(bool));
    } else {
      mxAssert(prev_time != NULL, "allocate_clocks: prev_time == NULL");
      mxAssert(running != NULL, "allocate_clocks: running == NULL");
      t0 = mxRealloc(t0, num*sizeof(mxArray*));
      prev_time = mxRealloc(prev_time, num*sizeof(double));
      running = mxRealloc(running, num*sizeof(bool));
    }
    
    for (i = num_clocks; i < num; i++){
      t0[i] = NULL;
      prev_time[i] = 0.0;
      running[i] = false;
    }
    
    num_clocks = num;
  }
}

void start_clock(int id){
  /*mexPrintf("start_clock(%d)\n", id);*/
  allocate_clocks(id+1);
  mexCallMATLAB(1,&t0[id],0,NULL,"clock");
  running[id] = true;
}

void stop_clock(int id){
  /*mexPrintf("stop_clock(%d)\n", id);*/
  mxAssert(num_clocks >= id+1, "stop_clock: id not allocated.\n");
  mxAssert(running[id], "stop_clock: clock must be running.\n");
  mxAssert(t0[id] != NULL, "stop_clock: t0 is NULL.\n");
  
  prev_time[id] = clock_time(id);
  running[id] = false;
  mxDestroyArray(t0[id]);
  t0[id] = NULL;
}

double clock_time(int id){
  mxArray *t1, *args[2], *result;
  double diff;
  if (running[id]){
    mexCallMATLAB(1, &t1, 0, NULL, "clock");
    args[0] = t1;
    args[1] = t0[id];
    mexCallMATLAB(1, &result, 2, args, "etime");
    diff = mxGetScalar(result);
    mxDestroyArray(result);
    mxDestroyArray(t1);
    return prev_time[id] + diff;
  } else {
    return prev_time[id];
  }
}

void reset_clock(int id){
  allocate_clocks(id+1);
  /*mexPrintf("reset_clock(%d)\n", id);*/
  mxAssert(!running[id], "reset_clock: clock must be stopped.\n");
  prev_time[id] = 0;
}

void print_time(int id, const char* name){
  mexPrintf("%25s: %4.4f seconds\n", name, clock_time(id));
}

void trace(int id){
  doTrace[id] = true;
}

void untrace(int id){
  doTrace[id] = false;
}

void tracepoint(int id, const char* str){
  if (doTrace[id])
    mexPrintf(str);
}


