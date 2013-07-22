MEX = mex #C:\MATLAB6p5sp\bin\win32\mex.bat 
INC = sisc_lib.c 
FLAGS = -largeArrayDims -g

all: basis_compute_CtC.c basis_dual_objective.c basis_solve_mex.c get_responses_mex.c sisc_lib.c
	$(MEX)  $(FLAGS) basis_compute_CtC.c $(INC) 
	$(MEX)  $(FLAGS) basis_dual_objective.c $(INC)
	$(MEX)  $(FLAGS) basis_solve_mex.c $(INC) 
	$(MEX)  $(FLAGS) get_responses_mex.c $(INC) 


