function [fs_A,fs_b,fs_c] = get_quadratic2(A, active_set, patch_M, patch_N, buffer_M, buffer_N, verbosity, as_old, fs_A_old, err)
 
% Return the objective function as a quadratic in terms of the
% coefficients in the active set, e.g. 
%
%            f_obj(x) = x'*fs_A*x + fs_b'*x + c
%
% Assumes all coefficients of S in the active set are zero.
%
% Slower, but hopefully more readable, than get_quadratic.

if ~exist('as_old'), as_old = []; end;
if ~exist('fs_A_old'), fs_A_old = []; end;
if ~exist('verbosity'), verbosity = 0; end;

num_active = size(active_set,1);
basis_M = size(A,1);
basis_N = size(A,2);
num_channels = size(A,3);
num_bases = size(A,4);
frame_M = patch_M + 2*buffer_M;
frame_N = patch_N + 2*buffer_N;

err_mask = zeros(frame_M,frame_N);
err_mask(buffer_M+(1:patch_M),buffer_N+(1:patch_N)) = 1;

% Retrieve cached entries
cache_ind = zeros(num_active,1);
for i=1:num_active,
  tmp = find(as_old==active_set(i));
  if size(tmp,1) > 0 && size(tmp,2) > 0,
    cache_ind(i) = tmp(1,1);
  else
    cache_ind(i) = -1;
  end
end
old_inds = find(cache_ind~=-1);
new_inds = find(cache_ind==-1);

% Compute the Hessian
if verbosity >= 4, fprintf('forming fs_A\n'); end;
fs_A = zeros(num_active);
fs_A(old_inds,old_inds) = fs_A_old(cache_ind(old_inds),cache_ind(old_inds));
for i=new_inds',
  [ind11 ind12 ind13] = ind2sub([frame_M frame_N num_bases],active_set(i));
  for j=1:num_active,
    [ind21 ind22 ind23] = ind2sub([frame_M frame_N num_bases],active_set(j));

    temp1 = zeros(frame_M, frame_N);
    temp1(mymod(ind11+(0:basis_M-1),frame_M),mymod(ind12+(0:basis_N-1),frame_N))=A(:,:,:,ind13);
    
    temp2 = zeros(frame_M, frame_N);
    temp2(mymod(ind21+(0:basis_M-1),frame_M),mymod(ind22+(0:basis_N-1),frame_N))=A(:,:,:,ind23);
    
    val = sum(sum(err_mask.*temp1.*temp2));
    
    fs_A(i,j) = val;
    fs_A(j,i) = val;
  end
end
fs_A = real(fs_A);
if verbosity >= 4, fprintf('  %d hits, %d misses\n', size(old_inds,1)^2, size(new_inds,1)*num_active); end;

if nargout >= 2,
  if verbosity >= 4, fprintf('forming fs_b\n'); end;
  fs_b = zeros(num_active,1);
  for i=1:num_active,
    [ind1 ind2 ind3] = ind2sub([frame_M frame_N num_bases],active_set(i));
    
    fs_b(i) = -2*sum(sum(sum(err(mymod(ind1+(0:basis_M-1),frame_M),...
                                 mymod(ind2+(0:basis_N-1),frame_N),:)...
                  .* A(:,:,:,ind3))));
  end
  
  fs_c = sum(sum(sum(err.^2)));

  fs_b = real(fs_b);
  fs_c = real(fs_c);
end


function res=mymod(A,b)

res = mod(A-1,b)+1;