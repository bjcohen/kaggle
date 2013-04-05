function [A,stats,lambda,s_all]=run_batch(X_all,A,pars,coef_pars,batch_num,learn,lambda,s_all)

% RUN_BATCH. Computes activations for all of the images in a batch, and possibly
% solves for the bases given those activations.
%
% [A, stats, lambda, S_all] = run_batch(X_all, A, pars, coef_pars, batch_num,
%     learn, lambda, S_all)
%
%   (The last two arguments, LAMBDA and S_ALL, are optional.)
%
%   First solves for activations for all images in in X_ALL using basis set A and parameters
%   given by PARS and COEF_PARS. X_ALL should be an M x N x num_images array.
%   BATCH_NUM is the batch ID, only used for output. If LEARN is set to true,
%   it also solves for the bases given the activations. LAMBDA is the initial value
%   for the dual variables for the bases; if you don't have a previous dual solution,
%   it is initialized to ones(num_bases, 1). 
%   You can optionally specify the initial values for the coefficients through S_ALL;
%   by default they are all assigned to zero. S_ALL must be a sparse matrix, and
%   since MATLAB requires all sparse matrices to be 2-dimensional, you need to
%   reshape the initial values for activations so that each column contains the activations
%   for one patch:
%
%      S_all(:,i) = sparse(reshape(S, M*N*num_bases, 1));
%
%   The function returns four values: A is the new basis set (if changed), STATS
%   contains some information about the objective function and running time,
%   LAMBDA is the optimal value of the dual variables for basis learning, and
%   S_ALL is a sparse array containing the activations. You can retrieve the activations
%   for a single patch using
%
%      S = reshape(full(S_all(:,i)), M, N, num_bases);

[patch_M, patch_N, num_channels, num_patches] = size(X_all);
[basis_M, basis_N, ig, num_bases] = size(A);
beta = pars.beta
A_freq = fft2(A,patch_M,patch_N);

if ~exist('s_all') || (size(s_all,1)*size(s_all,2)==0),
  s_all = sparse(patch_M*patch_N*num_bases,num_patches);
end

if ~exist('lambda'),
  lambda = ones(num_bases, 1);
end

stats.bases_time = 0;

% Cache Hessian
AtA = get_AtA(A);

A_new = A;

for p=1:num_patches,
  if pars.verbosity >= 2, 
    if batch_num == 0,
      fprintf('Test batch, Patch %d\n', p);
    else,
      fprintf('Batch %d, Patch %d\n', batch_num, p); 
    end
  end;
    
  X = X_all(:,:,:,p);

  % Retrieve initial values (see documentation above)
  temp = full(s_all(:,p));
  s = reshape(temp,patch_M,patch_N,num_bases); 
  
  t0 = clock;
  
  % Solve for the activations
  [s,coef_stats] = get_responses(X,A,pars.beta,coef_pars,p,s,AtA);
  fobj = coef_stats.f_obj;
  
  t1 = clock;
  stats.coef_time(p) = etime(t1,t0);  
  stats.fobj_pre(p) = fobj;
  
  if pars.verbosity >=2, fprintf('Time elapsed for this patch: %f seconds\n', stats.coef_time(p)); end;
  
  s_all(:,p) = reshape(s, patch_M*patch_N*pars.num_bases,1);
  
  if pars.verbosity == 1,
    if mod(p, 10) == 0, fprintf('[%d]', p); end;
  end
  
end

if pars.verbosity == 1, fprintf('\n'); end;
total_time = sum(stats.coef_time);
if pars.verbosity >= 1, fprintf('Finished solving for coefficients. Total time: %d minutes, %1.1f seconds\n', floor(total_time/60), mod(total_time, 60)); end;

clear AtA;

stats.bases_time = 0;
stats.coef_time_total = sum(stats.coef_time);
stats.fobj_pre_total = sum(stats.fobj_pre);
  

% Solve for the bases exactly.
if learn,
  t0 = clock;
  [A,lambda] = get_bases(X_all,s_all,basis_M,basis_N,pars.verbosity,lambda);
  t1 = clock;
  stats.bases_time = etime(t1,t0);
  
  % Center the bases
  for b=1:num_bases,
    [C,R] = meshgrid(1:basis_N,1:basis_M);
    center_r = round(sum(sum(sum(A(:,:,:,b).^2,3).*R)) / sum(sum(sum(A(:,:,:,b).^2))));
    dr = ceil(basis_M/2)-center_r;
    center_c = round(sum(sum(sum(A(:,:,:,b).^2,3).*C)) / sum(sum(sum(A(:,:,:,b).^2))));
    dc = ceil(basis_N/2)-center_c;
    
    A(:,:,:,b) = shift(A(:,:,:,b),[dr dc]);  
  end
end
