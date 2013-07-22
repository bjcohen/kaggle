function fs_x_all = solve_fs(fs_A_all, fs_b_all, fs_x_all, beta_all);

num_active = size(fs_A_all,1);
%fs_sign = (fs_x>0)+(fs_x==0).*(fs_b<0) - ((fs_x<0)+(fs_x==0).*(fs_b>0));
num_active_all = num_active;

passes = 0;

max_tries = 5;

% Note: beta is a vector the same size as fs_x

%fprintf('fobj: %0.15f\n', fs_x_all'*fs_A_all*fs_x_all + fs_b_all'*fs_x_all + beta_all'*abs(fs_x_all));

%num_active

for ind=1:max_tries,
  
  nonzero_terms = (1:num_active_all)';
  
  %fs_x_all
  corr_all = 2*fs_A_all*fs_x_all + fs_b_all;
  %good = find((fs_x_all ~= 0) + (abs(corr_all)>beta_all+10^-6));
  good = find((fs_x_all ~= 0) + (abs(corr_all)>beta_all+10^-6));
  %size(fs_x_all)
  fs_x = fs_x_all(good);
  %size(fs_A_all)
  fs_A = fs_A_all(good,good);
  %size(fs_b_all)
  fs_b = fs_b_all(good);
  %size(beta_all)
  beta = beta_all(good);
  %size(corr_all)
  corr = corr_all(good);
  %size(nonzero_terms)
  nonzero_terms = nonzero_terms(good);
  
  num_active = size(good,1);

  %fs_x = fs_x + 0.0000001*rand(size(fs_x));
 
  fs_sign = ((fs_x>0) + (fs_x==0).*(corr<0)) ...
            - ((fs_x<0) + (fs_x==0).*(corr>0));
  
  nzg = size(find(abs(corr)>beta+10^-6),1);
  %fprintf('Number with nonzero gradient: %d\n',nzg);
  if nzg == 0, break; end;
  
  % Make sure none of the zero coefficients want to flip sign.
  while true,
    
    %size(fs_b)
    %size(beta)
    %size(fs_sign)
    
    fs_x_new = -0.5*(fs_A\(fs_b+beta.*fs_sign));
    
    good_ = ((fs_x==0).*(sign(fs_x_new)==fs_sign)+(fs_x~=0));
    
    good = find(good_);
    
    if size(find(1-good_),1)==0,
      break;
    else
      nonzero_terms = nonzero_terms(good);
      fs_A = fs_A(good,good);
      fs_b = fs_b(good);
      fs_x = fs_x(good);
      fs_sign = fs_sign(good);
      beta = beta(good);
      num_active = size(nonzero_terms,2);
      
      fs_sign = sign(fs_x) + fs_sign.*(fs_x==0);
      
      %fprintf('B\n');
      %fprintf('fobj: %0.15f\n', fs_x'*fs_A*fs_x + fs_b'*fs_x + beta'*abs(fs_x));
    end
  end
  
  while true,
    if num_active == 0,
      break;
    end
    
    passes = passes + 1;
    
    fs_x_new = -0.5*(fs_A\(fs_b+beta.*fs_sign));
    
    if find(abs(fs_x_new) == Inf) > 0,
      fs_x_new = -0.5*((fs_A+10^-5*eye(num_active))\(fs_b+beta.*fs_sign));
      warning('Singular second derivative matrix. Using nonsingular approximation.');
    end
    
    % Find all zero crossings, 0th entry represents exact solution
    num_crossings=1;
    crossings = [0 1];
    for i=1:num_active,
      if sign(fs_x_new(i)) == -fs_sign(i),
        cross = fs_x(i) / (fs_x(i)-fs_x_new(i));
        crossings(num_crossings+1,:) = [i cross];
        num_crossings = num_crossings+1;
      end
    end
    
    best_val=Inf;
    best_ind=0;
    for i=1:size(crossings,1),
      fs_x_guess = fs_x + crossings(i,2)*(fs_x_new-fs_x);
      fs_x_guess = fs_x_guess.*(sign(fs_x_guess)==fs_sign);
      guess_obj = fs_x_guess'*fs_A*fs_x_guess + fs_b'*fs_x_guess + sum(beta.*abs(fs_x_guess));
      if guess_obj < best_val,
        best_val = guess_obj;
        best_ind = i;
      end
    
      %fprintf('i %d length %d obj %d\n', i, crossings(i,2), guess_obj);
    end
    
    fc_ind = crossings(best_ind);
    fs_x_new = fs_x + crossings(best_ind,2)*(fs_x_new-fs_x);
    fs_x_new = fs_x_new.*(sign(fs_x_new)==fs_sign);
    
    
    % TODO: figure out why I got this error
    %if crossings(best_ind,2) == 0, error; end;
    
    if crossings(best_ind,2) == 0, 
      fs_x_all = zeros(size(fs_x));
      break;
    end;
    
    %fprintf('*** BEST: i %d length %d\n', best_ind, crossings(i,2));
    
    if fc_ind == 0 && size(find(sign(fs_x_new)~=fs_sign),1)==0,
      fs_x = fs_x_new;
      break;
    else
      good = find(fs_x_new);
      
      nonzero_terms = nonzero_terms(good);
      fs_A = fs_A(good,good);
      fs_b = fs_b(good);
      fs_x = fs_x_new(good);
      fs_sign = fs_sign(good);
      num_active = size(good,1);
      beta = beta(good);
      
      %fprintf('C\n');
      %fprintf('fobj: %0.15f\n', fs_x'*fs_A*fs_x + fs_b'*fs_x + beta'*abs(fs_x));
    end
    
  end

  fs_x_all = zeros(size(fs_x_all));
  fs_x_all(nonzero_terms') = fs_x;
  
  %%fprintf('D\n');
  %fprintf('fobj: %0.15f\n', fs_x'*fs_A*fs_x + fs_b'*fs_x + beta'*abs(fs_x));
  
end
