
#ifndef __tisc_lib_h
#define __tisc_lib_h

#include "mex.h"

void tisc_init(void);
void tisc_destroy(void);

int min(int a, int b);
int max(int a, int b);
double mindbl(double, double);
double maxdbl(double, double);
double absdbl(double);

void print_double_array(const double*, int, int);
void print_mex_array(const mxArray*);

void times(double*, double*, double, double, double, double);

int tri(int, int);
int inv_tri(int);

mxArray* fft2(const mxArray*);
mxArray* ifft2(const mxArray*);

void get_real_imag(double**, double**, const mxArray*);

mxArray* get_reconstruction(const mxArray*, const mxArray*);
mxArray* get_correlation(const mxArray*, const mxArray*);
void get_objective_function(double*, double*, double*, const mxArray*, const mxArray*, double);

void start_clock(int);
void stop_clock(int);
double clock_time(int);
void print_time(int, const char*);

#define NUM_TRACE 7
#define TRACE_FS 0
#define TRACE_LIB 1
#define TRACE_GD 2
#define TRACE_FILT 3
#define TRACE_BASIS_GD 4
#define TRACE_BASIS_GD_INNER 5
#define TRACE_BASIS 6

void trace(int);
void untrace(int);
void tracepoint(int, const char*);


#endif


