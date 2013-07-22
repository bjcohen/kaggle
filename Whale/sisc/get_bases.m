function [A, lambda, A_freq] = get_bases(X_all, s_all, basis_M, basis_N, verbosity, lambda);

% GET_BASES. Solves for the bases given a set of patches and activations.
%
% [A, lambda] = get_bases3(X_all, s_all, basis_M, basis_N, verbosity, lambda)
%   Solves for bases given a set of images X_ALL (which should be M x N x num_images),
%   a set of sparse activations S_ALL (see 'help run_batch' for more info),
%   basis dimensions BASIS_M and BASIS_N, and verbosity level VERBOSITY.
%   LAMBDA is the initial guess for the dual problem.
%   Returns bases A and Lagrange multipliers LAMBDA.

[patch_M, patch_N, num_channels, num_patches] = size(X_all);
num_elems = size(find(s_all~=0),1);
num_bases = size(s_all,1) / (patch_M*patch_N);

c = patch_M*patch_N*basis_M*basis_N*num_channels;
X_freq_all = fft2(X_all);
[C,R] = meshgrid(0:patch_N-1,0:patch_M-1);
exps = exp(-2*pi*i*R/patch_M).*exp(-2*pi*i*C/patch_N);

if verbosity >= 1,
  fprintf('patch_M %d patch_N %d num_channels %d num_patches %d num_elems %d num_bases %d\n',...
          patch_M, patch_N, num_channels, num_patches, num_elems, num_bases);
end
t0 = clock;
[CtC_all, Ctd_all] = basis_compute_CtC(X_freq_all, s_all, c, exps);
if verbosity >= 1, fprintf('Time for computing CtC: %f\n', etime(clock,t0)); end;


% Solve dual problem
if verbosity >= 1,
  options = optimset('GradObj','on','Hessian','on','Display','iter');
  fprintf('Solving the dual problem.\n');
else
  options = optimset('GradObj','on','Hessian','on','Display','off');
end
t0 = clock;
[lambda,fobj] = fmincon(@(x) basis_dual_objective(CtC_all,Ctd_all,x,c),lambda,[],[],[],[],zeros(num_bases,1),[],[],options);
if verbosity >= 1, fprintf('Time for solving dual: %f\n', etime(clock,t0)); end;

% Solve for the bases given the Lagrange multipliers
t0 = clock;
A_freq = basis_solve_mex(CtC_all, Ctd_all, lambda);
if verbosity >= 1, fprintf('Time for finding bases: %f\n', etime(clock,t0)); end;

A = real(ifft2(A_freq));
A = A(1:basis_M,1:basis_N,:,:);

for m=1:num_bases,
  %temp = mean(mean(mean(A(:,:,:,m).^2)));
  %temp2 = mean(mean(mean(A_freq(:,:,:,m).*conj(A_freq(:,:,:,m)))));
  %fprintf('norm: %1.2f, norm_freq: %1.2f, lambda: %1.5f\n', temp, temp2, lambda(m));
  A(:,:,:,m) = A(:,:,:,m) / sqrt(mean(mean(mean(A(:,:,:,m).^2))));
end




