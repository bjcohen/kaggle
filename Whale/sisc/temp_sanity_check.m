function temp_sanity_check(tests);

num_fail = 0;

fprintf('Testing that LARS converges to correct value... ');
t0 = clock;
try,
  patch_M = 72;
  patch_N = 104;
  
  num_bases = 16;
  
  load data/IMAGES.mat
  X = IMAGES(1:patch_M,1:patch_N,1);
  load /afs/cs/group/brain/scratch/roger/data/bases/whitened_16.mat;
  
  pars.beta = 200;
  [s,coef_stats] = temp_get_responses_lars(X,A,pars);
  fobj_lars = coef_stats.fobj;
  if ~coef_stats.pass, error('Incorrect result.'); end;
  if coef_stats.num_iters < 15, error('Not enough iterations.\n'); end;
  fprintf('pass (%f seconds)\n', etime(clock,t0));
catch,
  fprintf('FAIL (%f seconds)\n', etime(clock,t0));
  print_error(lasterror);
  num_fail = num_fail + 1;
end

fprintf('Testing that other algorithms are less than 1%% suboptimal:\n');
fprintf('    feature_sign_mex... ');
t0 = clock;
try,
  A_freq = fft2(A,patch_M,patch_N);
  
  AtA = get_AtA(A);
  S = zeros(patch_M, patch_N, num_bases);
  
  coef_pars = default_coef_pars(struct,'verbosity',0);
  coef_pars.coeff_iter = 20;
  s = get_responses(X,A,200,coef_pars, 0, S, AtA);
  
  % Compute objective function
  reconst_freq = zeros(patch_M, patch_N, 1);
  for m=1:16,
    reconst_freq = reconst_freq + fft2(s(:,:,m)).*A_freq(:,:,1,m);
  end
  err = real(X-ifft2(reconst_freq));
  fres = sum(sum(sum(err.^2)));
  fspars = 200*sum(sum(sum(abs(s))));
  fobj_fs = fres + fspars;
  
  subopt = (fobj_fs-fobj_lars)/fobj_lars;
  if subopt > 0.01, error ('%f%% suboptimal', 100*subopt); end;
  fprintf('pass (%f seconds, %f%% suboptimal)\n', etime(clock,t0), 100*subopt);
catch,
  fprintf('FAIL (%f seconds)\n', etime(clock,t0));
  print_error(lasterror);
  num_fail = num_fail + 1;
end


if num_fail == 0,
  fprintf('All tests pass :-)\n\n');
else
  fprintf('%d tests fail.\n\n', num_fail);
end




function print_error(error);

fprintf('        %s\n', error.message);
for ind=1:length(error.stack),
  fprintf('          %s, line %d\n', error.stack(ind).name, error.stack(ind).line);
end











