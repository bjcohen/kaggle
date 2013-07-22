function [s,coef_stats] = temp_get_responses_lars(X,A,pars,patch_num);
 
if ~exist('pars'), pars.blah = ''; end;

if ~isfield(pars, 'beta'), pars.beta = 200; end;
if ~isfield(pars, 'verbosity'), pars.verbosity = 0; end;
if ~isfield(pars, 'coeff_iter'), pars.coeff_iter = 10000; end;
if ~isfield(pars, 'patch_M'), pars.patch_M = size(X,1); end;
if ~isfield(pars, 'patch_N'), pars.patch_N = size(X,2); end;
if ~isfield(pars, 'buffer_M'), pars.buffer_M = 0; end;
if ~isfield(pars, 'buffer_N'), pars.buffer_N = 0; end;
if ~isfield(pars, 'basis_M'), pars.basis_M = size(A,1); end;
if ~isfield(pars, 'basis_N'), pars.basis_N = size(A,2); end;
if ~isfield(pars, 'display_every'), pars.display_every = 0; end;

if ~exist('patch_num'), patch_num = 0; end;

max_iter = pars.coeff_iter;
num_bases = size(A,4);
beta = pars.beta;
patch_M = pars.patch_M;
patch_N = pars.patch_N;
buffer_M = pars.buffer_M;
buffer_N = pars.buffer_N;
frame_M = patch_M + 2*buffer_M;
frame_N = patch_N + 2*buffer_N;
basis_M = pars.basis_M;
basis_N = pars.basis_N;
num_channels = size(A,3);

coef_mask = zeros(patch_M, patch_N, num_bases);
coef_mask(1:patch_M-basis_M+1, 1:patch_N-basis_N+1, :) = 1;

s = zeros(frame_M,frame_N,num_bases);
s_freq = zeros(frame_M,frame_N,num_bases);
corr_freq = zeros(frame_M,frame_N,num_bases);
corr = zeros(frame_M,frame_N,num_bases);
lars_G = [];

A_freq = fft2(A,frame_M,frame_N);

err_mask = zeros(frame_M,frame_N);
err_mask(buffer_M+(1:patch_M),buffer_N+(1:patch_N)) = 1;

reconst = zeros(frame_M,frame_N);
err = X.*err_mask;
err_freq = fft2(err);

for m=1:num_bases,
  corr_freq = sum(err_freq.*conj(A_freq(:,:,:,m)),3);
  corr(:,:,m) = corr(:,:,m) + real(ifft2(corr_freq));
end

clear coef_stats;
coef_stats.time(1) = 0;
coef_stats.total_time(1) = 0;

if pars.verbosity >= 1, disp('Starting optimization...'); end;
for iter=1:max_iter,
  if pars.verbosity == 1 && mod(iter,100)==0, fprintf('.'); end;
  
  % Start timer
  t0 = clock;
  
    % Display the image and the reconstruction
  if pars.display_every > 0 && mod(iter,pars.display_every) == 0,
    save_mode = (pars.save_coeffs_every > 0);
    display_reconstruction(X,real(s),real(reconst),5,pars,save_mode,patch_num,iter);
  end
  
  % Compute the objective function
  f_res = norm(err, 'fro')^2;
  f_spars = beta*sum(sum(sum(abs(s))));
  f_obj = f_res + f_spars;
  
  coef_stats.f_res(iter)=f_res;
  coef_stats.f_spars(iter)=f_spars;
  coef_stats.f_obj(iter)=f_obj;
  
  if pars.verbosity >= 2,
    fprintf('patch %d iter %d\n', patch_num, iter);
    fprintf('  res %1.20d spars %1.20d obj %1.20d\n', f_res, f_spars, f_obj);
  end
  
  % If first iteration, initialize the active set
  if iter==1,
    % Find coefficient with maximum correlation
    [val,ind] = max(reshape(abs(corr).*coef_mask,frame_M*frame_N*num_bases,1));
    
    active_set = ind;
    signs = sign(corr(ind));
    as_old = [];
    effective_beta = 2*val;
  end
  
  num_active = size(active_set,1);
  
  % Add a row and column to G if necessary
  if iter==1 || strcmp(action,'add'),
    if pars.verbosity >= 4, disp('Expanding G...\n'); end;
    
    active_set_old = active_set(1:num_active-1);
    
    % get_quadratic returns the correlation matrix in the coefficients,
    % while lars_G is the correlation matrix in the coefficients times their sign.
    lars_G = lars_G .* (signs(1:num_active-1)*signs(1:num_active-1)');
    lars_G = temp_get_quadratic2(A,active_set,patch_M,patch_N,buffer_M,buffer_N,pars.verbosity,active_set_old,lars_G);
    lars_G = lars_G .* (signs*signs');
  end

  if pars.verbosity >= 4, fprintf('Inverting G...\n'); end;
  lars_G_inv = inv(lars_G);
  lars_A = 1/sqrt(sum(sum(lars_G_inv)));
  lars_w = lars_A * lars_G_inv * ones(num_active,1);

  % Form u, the equiangular unit vector
  if pars.verbosity >= 4, disp('Forming u...\n'); end;
  lars_u = zeros(frame_M,frame_N);
  for i=1:num_active,
    [ind1, ind2, ind3] = ind2sub([frame_M frame_N num_bases], active_set(i));
  
    lars_u(mymod(ind1+(0:basis_M-1),frame_M),mymod(ind2+(0:basis_N-1),frame_N)) ...
        = lars_u(mymod(ind1+(0:basis_M-1),frame_M),mymod(ind2+(0:basis_N-1),frame_N)) ...
        + signs(i)*lars_w(i)*A(:,:,:,ind3);
  end
  lars_u = lars_u.*err_mask;
 
  % Compute a, the vector of inner products with u
  if pars.verbosity >= 4, disp('Forming a...'); end;
  lars_u_freq = fft2(lars_u);
  for m=1:num_bases,
    lars_a_freq = lars_u_freq.*conj(A_freq(:,:,m));
    lars_a(:,:,m) = real(ifft2(lars_a_freq));
  end

  % Compute gamma, the step size, and update active set
  if pars.verbosity >= 4, disp('Finding gamma...'); end;
  lars_C = abs(corr(active_set(1)));
  % Find the first new coefficient to join the active set
  temp1 = (lars_C - corr)./(lars_A - lars_a + 10^-18); % Avoid divide-by-zero messsages
  temp1(find(temp1<=0)) = Inf;
  temp2 = (lars_C + corr)./(lars_A + lars_a + 10^-18);
  temp2(find(temp2<=0)) = Inf;
  temp = min(temp1,temp2);
  temp(active_set) = Inf;
  temp(find(coef_mask==0)) = Inf;
  %[temp1a,temp1b]=min(temp,[],3);
  %[temp2a,temp2b]=min(temp1a,[],2);
  %[temp3a,temp3b]=min(temp2a);
  %ind1 = temp3b;
  %ind2 = temp2b(ind1);
  %ind3 = temp1b(ind1,ind2);
  %val1 = temp3a;
  
  [val1,cross_ind] = min(reshape(temp,frame_M*frame_N*num_bases,1));

  % Find the first active set coefficient to cross zero
  val2 = Inf;
  for i=1:num_active,
    ind = active_set(i);
    if lars_w(i) < 0,
      if -abs(s(ind))/lars_w(i) < val2,
        val2 = -abs(s(ind))/lars_w(i);
        val2_ind = i;
      end
    end
  end
  
  % Find the point at which the maximum correlation equals beta
  val3 = (lars_C-beta/2)/lars_A;

  if val1 < val2 && val1 < val3,  % New coefficient joins active set
    lars_gamma = val1;
    if corr(cross_ind) - lars_gamma*lars_a(cross_ind) > 0,
      sgn = 1;
    else
      sgn = -1;
    end
    if pars.verbosity >= 5, 
      [ind1,ind2,ind3] = ind2sub([frame_M frame_N num_bases],cross_ind);
      fprintf('** Adding %d %d %d %d **\n', ind1, ind2, ind3, sgn); 
    end
    active_set = [active_set; cross_ind];
    signs = [signs; sgn];
    action = 'add';
    
  elseif val2 < val3,             % Coefficient leaves active set because it crosses zero
    if pars.verbosity >= 5, 
      [ind1,ind2,ind3] = ind2sub([frame_M frame_N num_bases],active_set(val2_ind));
      fprintf('** Removing %d %d %d **\n', ind1, ind2, ind3); 
    end
    s(active_set(val2_ind))=0;
    lars_gamma = val2;
    active_set = active_set([1:val2_ind-1 val2_ind+1:num_active]);
    signs = signs([1:val2_ind-1 val2_ind+1:num_active]);
    lars_w = lars_w([1:val2_ind-1 val2_ind+1:num_active]);
    lars_G = lars_G([1:val2_ind-1 val2_ind+1:num_active],[1:val2_ind-1 val2_ind+1:num_active]);
    num_active = num_active - 1;
    action = 'remove';
  else                            % We're done, since the optimal point lies on this path
    lars_gamma = val3;
    action = 'done';
  end
  
    
  % Update s
  if pars.verbosity >= 4, disp('Updating s'); end;
  for i=1:num_active,
    s(active_set(i)) = s(active_set(i)) + signs(i)*lars_w(i)*lars_gamma;
  end

  reconst = reconst + lars_gamma*lars_u;
  err = X - reconst;
  corr = corr - lars_gamma*lars_a;
  
  if size(find(reconst.*(1-err_mask))>10^-10,1) > 0, error('Reconstruction nonzero outside patch'); end;

  lars_C = abs(corr(active_set(1)));
  effective_beta = 2*lars_C;
  
  if pars.verbosity >= 2, fprintf('  Size of active set: %d\n', num_active); end;
  if pars.verbosity >= 2, fprintf('  Effective beta: %d\n', 2*lars_C); end;
  if pars.verbosity >= 3, fprintf('  Step size: %d\n', lars_gamma); end;
  
  % Record the time for this iteration
  t1 = clock;
  tm = etime(t1,t0);
  if pars.verbosity >= 2,
    fprintf('Elapsed time for iteration %d: %d seconds\n', iter, tm);
  end
  coef_stats.time(iter+1)=tm;
  coef_stats.total_time(iter+1)=sum(coef_stats.time(1:iter+1));
  
  if pars.verbosity >= 5, sanity_check(active_set,s,A_freq,X,corr,effective_beta,pars,err_mask,coef_mask); end;
  
  if strcmp(action,'done'), break; end;
end

% Compute the objective function
f_res = norm(err, 'fro')^2;
f_spars = beta*sum(sum(sum(abs(s))));
f_obj = f_res + f_spars;

coef_stats.f_res(iter+1)=f_res;
coef_stats.f_spars(iter+1)=f_spars;
coef_stats.f_obj(iter+1)=f_obj;
coef_stats.fobj = f_obj;

coef_stats.num_iters = iter;

coef_stats.pass = sanity_check(active_set,s,A_freq,X,corr,effective_beta,pars,err_mask,coef_mask);

if pars.verbosity >= 1,
  if pars.verbosity == 1, fprintf('\n'); end;
  if strcmp(action,'done'),
    fprintf('Finished optimization in %d iterations.\n', iter); 
  else
    fprintf('Did not finish optimization in %d iterations.\n', iter);
  end
  fprintf('Final f_obj value: %d\n\n', f_obj); 
end


% Check that:
%  correlations all equal effective_beta
%  correlations are correct
%  correlations agree with the corresponding sign of s
%  gradient would be zero if beta == effective_beta
function result=sanity_check(as,s,A_freq,X,corr,beta,pars,err_mask,coef_mask);

result = true;
num_bases = size(A_freq,4);

if pars.verbosity >= 5, fprintf('  Sanity check!\n'); end;
error1 = max(abs(abs(2*corr(as))-beta));
if pars.verbosity >= 5, fprintf('    Correlations equal effective_beta within %d.\n', error1); end;
if error1 > 10^-8, result = false; end;

err_freq = fft2(X);
s_freq = fft2(s);
for m=1:num_bases,
  err_freq = err_freq - s_freq(:,:,m).*A_freq(:,:,m);
end
err = ifft2(err_freq).*err_mask;
err_freq = fft2(err);
for m=1:num_bases,
  corr_ex(:,:,m) = ifft2(err_freq.*(conj(A_freq(:,:,m))));
end
error2 = max(max(max(abs(corr-corr_ex))));
if pars.verbosity >= 5, fprintf('    Correlations accurate within %d.\n', error2); end;
if error2 > 10^-8, result = false; end;

if size(find((sign(real(s))==-sign(real(corr))).*(real(corr)>10^-10)),1) == 0,
  if pars.verbosity >= 5, fprintf('    Signs of correlations match signs of s.\n'); end;
else
  if pars.verbosity >= 5, fprintf('    Uh-oh. Some signs are wrong.\n'); end;
  result = false;
end

grad = -2*corr;
grad = (s>0).*(grad+beta)+(s<0).*(grad-beta)+(s==0).* ...
       (grad<-beta).*(grad+beta)+(s==0).*(grad>beta).*(grad-beta);
error3 = max(max(max(abs(grad).*coef_mask)));
if pars.verbosity >= 5, fprintf('    Gradient zero within %d\n', error3); end;
if error3 > 10^-8, result = false; end;



function res=mymod(A,b)

res = mod(A-1,b)+1;