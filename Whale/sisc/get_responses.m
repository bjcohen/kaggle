function [S,coef_stats]=get_responses(X,A,beta,coef_pars,patch_num,S,AtA)

% GET_RESPONSES Wrapper function for computing activations using all of the algorithms. Currently
%   supported algorithms: gradient_descent, feature_sign, feature_sign_mex, lars.
% 
%   [S,COEF_STATS]=get_responses(X,A,BETA,ALGORITHM,PARS,PATCH_NUM,S_INIT, AtA) minimizes 
%   the sparse coding objective function for image X and bases A with respect 
%   to activations S using the algorithm ALGORITHM. The rest of the parameters are optional.
%   PARS specifies all of the other parameters as explained in default_params.m. PATCH_NUM
%   gives the ID of the current patch, for purposes of printing. S_INIT is the initial value
%   of S (default is all zeros). AtA is the cached second derivative matrix which is obtained
%   by calling get_AtA.

[patch_M, patch_N, num_channels, num_patches] = size(X);
[basis_M, basis_N, ig, num_bases] = size(A);

A_freq = fft2(A,patch_M,patch_N);

if coef_pars.tile,
  % Iteratively solve for the activations within a sliding window. We treat each window
  % as its own image, so we are not solving for coefficients whose bases lie partly
  % outside of the window. Implicitly, we clamp all of the other coefficients to their
  % previous value, but this is actually implemented by clamping them to zero and subtracting
  % them out from the image that's being solved for.
  
  window_M = min(patch_M,3*basis_M);
  window_N = min(patch_N,3*basis_N);
  hop_M = ceil(window_M/2);
  hop_N = ceil(window_N/2);
  
  A_freq2 = fft2(A,window_M,window_N);
  
  for r=1:coef_pars.tile_iters,
    for v=1:1+ceil((patch_M-window_M)/hop_M),
      vstart = min((v-1)*hop_M,patch_M-window_M);
      for h=1:1+ceil((patch_N-window_N)/hop_N),
        hstart = min((h-1)*hop_N,patch_N-window_N);
        
        % Find the reconstruction of the image obtained using all of the coefficients
        % EXCEPT the ones that will be solved for this iteration.
        S_temp = S;
        S_temp(vstart+(1:window_M-basis_M+1),hstart+(1:window_N-basis_N+1),:) = 0;
        rec = reconstruction(S_temp,A);
        err1 = X - rec;
        err = err1(vstart+(1:window_M),hstart+(1:window_N),:);
        
        % Initialize the coefficients we're solving for to their previous value.
        S_start = zeros(window_M,window_N,num_bases);
        S_start(1:window_M-basis_M+1,1:window_N-basis_N+1,:) = ...
            S(vstart+(1:window_M-basis_M+1),hstart+(1:window_N-basis_N+1),:);
        
        [S_new,coef_stats1] = get_responses_helper(err,A,beta,coef_pars,patch_num,S_start, AtA, A_freq2);
        
        % Replace the coefficients with their new values.
        S(vstart+(1:window_M-basis_M+1), hstart+(1:window_N-basis_N+1),:) = ...
            S_new(1:window_M-basis_M+1, 1:window_N-basis_N+1,:);
        
        rec = zeros(patch_M,patch_N,num_channels);
        for b = 1:num_bases,
          for k = 1:num_channels,
            temp = conv2(S(:,:,b),A(:,:,k,b));
            rec(:,:,k) = rec(:,:,k) + temp(1:patch_M,1:patch_N);
          end
        end
        
        if coef_pars.verbosity >= 2, 
          fres = sum(sum((X-rec).^2));
          fspars = beta*sum(sum(sum(abs(S))));
          fobj = fres + fspars;
          fprintf('Tile complete. round %d vstart %d hstart %d fres %1.10f fspars %1.10f fobj %1.10f\n', r, vstart, hstart, fres,fspars, fobj); 
        end;
        
      end
    end
  end
  
  fres = sum(sum((X-rec).^2));
  fspars = beta*sum(sum(sum(abs(S))));
  coef_stats.f_obj = fres + fspars;
else,
  [S,coef_stats] = get_responses_helper(X, A, beta, coef_pars, patch_num, S, AtA, A_freq);
end




function [S,coef_stats] = get_responses_helper(X, A, beta, coef_pars, patch_num, S, AtA, A_freq);

[patch_M, patch_N, num_channels, num_patches] = size(X);
[basis_M, basis_N, ig, num_bases] = size(A);

[S, times_all, fobj_all] = get_responses_mex(X, A, A_freq, coef_pars.coeff_iter, ...
                                                beta, coef_pars.exact, coef_pars.num_coords, ...
                                                coef_pars.verbosity, S, AtA);

last_iter = length(find(times_all~=0));
last_iter = max(last_iter, 1);
coef_stats.f_obj = fobj_all(last_iter);


