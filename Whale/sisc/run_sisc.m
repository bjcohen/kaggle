function run_sisc();

% An example script to demonstrate how to use the SISC code.
% Learns a set of 16 shift-invariant bases from Olshausen's
% natural image set.

% Get default parameters
pars = default_pars(struct);
coef_pars = default_coef_pars(struct);

patch_M = pars.patch_M;
patch_N = pars.patch_N;

% Generate white noise bases, normalize
rand('seed', 100);
A = randn(pars.basis_M,pars.basis_N,1,pars.num_bases);  
for m=1:pars.num_bases,
  A(:,:,:,m)=A(:,:,:,m)-mean(mean(mean(A(:,:,:,m))));
  A(:,:,:,m)=A(:,:,:,m)/sqrt(mean(mean(mean(A(:,:,:,m).^2))));
end

if pars.verbosity >= 1, disp('Loading images...'); end;
load data/IMAGES.mat

% Crop images to eliminate edge effects
IMAGES = IMAGES(pars.crop_amt+1:size(IMAGES,1)-pars.crop_amt, pars.crop_amt+1:size(IMAGES,2)-pars.crop_amt, :);

% Make images zero mean, unit variance
for ind=1:size(IMAGES,3),
  IMAGES(:,:,ind) = IMAGES(:,:,ind)-mean(mean(IMAGES(:,:,ind)));
  IMAGES(:,:,ind) = IMAGES(:,:,ind)/sqrt(mean(mean(IMAGES(:,:,ind).^2)));
end

% Generate training and test sets
[M, N, num_images] = size(IMAGES);
num_horz = floor((M-pars.overlap_M)/(pars.patch_M-pars.overlap_M));
num_vert = floor((N-pars.overlap_N)/(pars.patch_N-pars.overlap_N));
total_patches = num_images*num_horz*num_vert;

patch_set = randperm(total_patches);
train_patches = patch_set(pars.num_test+1:total_patches);
num_train = total_patches-pars.num_test;
test_patches = patch_set(1:pars.num_test);

X_all = zeros(patch_M, patch_N, total_patches);
for k=1:total_patches,
  m = patch_set(k);
  p=mod(m-1,num_images)+1;
  v=mod(floor((m-1)/num_images),num_vert)+1;
  h=floor((m-1)/(num_images*num_vert))+1;
  temp = IMAGES((pars.patch_M-pars.overlap_M)*(h-1)+(1:pars.patch_M),(pars.patch_N-pars.overlap_N)*(v-1)+(1:pars.patch_N),p);  
  
  % Smooth edges to reduce edge effects
  [C,R] = meshgrid(1:pars.patch_N, 1:pars.patch_M);
  temp = temp ./ (1+exp(-4*(R/pars.basis_M)+2));
  temp = temp ./ (1+exp(-4*((pars.patch_M+1-R)/pars.basis_M)+2));
  temp = temp ./ (1+exp(-4*(C/pars.basis_N)+2));
  temp = temp ./ (1+exp(-4*((pars.patch_N+1-C)/pars.basis_N)+2));
  temp = temp-mean(mean(temp));
  
  X_all(:,:,k) = temp;
end

if pars.verbosity >= 1, fprintf('Using %d patches\n', total_patches); end;

X_all = reshape(X_all, patch_M, patch_N, 1, total_patches);
X_all_train = X_all(:,:,:,pars.num_test+1:total_patches);
X_all_test = X_all(:,:,:,1:pars.num_test);

num_batches = floor(num_train/pars.batch_size);
if num_batches == 0, error('Not enough patches for that batch size.'); end;

if pars.verbosity >= 1, fprintf('Using %d patches and %d batches, %d patches per batch.\n', total_patches, num_batches, pars.batch_size); end;

% Main loop
s_all = sparse(patch_M*patch_N*pars.num_bases,total_patches);
total_time = 0;
lambda = ones(pars.num_bases,1);
for tr=1:pars.num_trials,
  batch = mod(tr-1,num_batches);
  patches = pars.batch_size*batch+(1:pars.batch_size);
  
  % Solve for coefficients, bases in one batch
  [A,batch_stats,lambda,s_all(:,patches)] = run_batch(X_all_train(:,:,:,patches),A,pars,coef_pars,tr,true,lambda,s_all(:,patches));
  
  % Save timing information, compute objective function on test set
  total_time = total_time + batch_stats.coef_time_total + batch_stats.bases_time;
  train_time(tr) = total_time/60;
  
  dummy_pars = coef_pars;
  dummy_pars.exact = true;
  dummy_pars.coeff_iter = 10000;
  dummy_pars.num_coords = 100;
  [A,batch_stats] = run_batch(X_all_test,A,pars,dummy_pars,0,false);
  fobj_all(tr) = batch_stats.fobj_pre_total;
  figure(2); plot(fobj_all); title('Objective function by batch.');
  figure(3); plot(train_time,fobj_all); title('Objective function by time.');
  drawnow;
  
  % Display and save
  if pars.display_bases_every ~= 0 && mod(tr,pars.display_bases_every)==0,
    display_bases(A,1);
    drawnow;
  end
  
  if pars.save_bases_every ~= 0 && mod(tr,pars.save_bases_every)==0,
    bases_outfile = [pars.basedir '/' pars.savename '.mat'];
    save(bases_outfile, 'A');
    save([pars.basedir '/' pars.savename '_stats.mat'],'pars','train_time', 'fobj_all');
  end    
end


