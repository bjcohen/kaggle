function temp_run_audio(pars, coef_pars, tracks);


switch pars.source,
  case 'music_old',
    for ind = 1:20,
      fname = ['../music/data/' int2str(ind) '.au'];
      Y_time{ind} = auread(fname);
      Y_time{ind} = Y_time{ind}(1:pars.downsample:size(Y_time{ind},1));
      Y_time{ind} = Y_time{ind} / std(Y_time{ind});
    end
    num_tracks = 20;


  case 'timit',
    %matfile = '/afs/cs/u/rajatr/scratch/timit/matdata/tr_1_data.mat';
    %load(matfile);
    %num_tracks = size(tr_data,2);
    %for ind=1:num_tracks,
    %  Y{ind} = cell2mat(tr_data(1,ind));
    %  Y{ind} = Y{ind}(1:pars.downsample:size(Y{ind},1));
    %  Y{ind} = Y{ind} / std(Y{ind});
    %end

    %filenames = all_files_timit('/afs/cs/group/brain/scratch/roger/data/timit2', pars.regions);
    filenames = all_files_timit(pars.regions);
    for ind=1:min(length(filenames),pars.max_tracks),
      %Y{ind} = wavread(files{ind});
      file = dir(filenames{ind});
      fid = fopen(filenames{ind},'r');
      Y_time{ind} = double(fread(fid, file.bytes/2, 'int16'));
      fclose(fid);
      Y_time{ind} = Y_time{ind}(1:pars.downsample:size(Y_time{ind},1));
      Y_time{ind} = Y_time{ind} / std(Y_time{ind});
      ind = ind + 1;
    end

  case 'music',
    ind = 1;
    for gen = pars.genres,
      for sng = pars.songs,
        fname = ['/afs/cs/u/rajatr/scratch/music/train/' int2str(gen) '___' int2str(sng) '.wav'];
        fname
        Y_time{ind} = wavread(fname);
        Y_time{ind} = Y_time{ind}(1:pars.downsample:size(Y_time{ind},1));
        Y_time{ind} = Y_time{ind} / std(Y_time{ind});
        ind = ind + 1;
      end
    end

  otherwise,
    error;
end

if exist('tracks'),
  Y_time = Y_time(tracks);
end
num_tracks = size(Y_time,1);

if strcmp(pars.test_type,'track'),
  num_test_tracks = pars.num_test;
else
  num_test_tracks = 0;
end


switch pars.mode,
  case 'time',
    Y = Y_time;
    clear Y_time;

    switch pars.basis_source,
      case 'random',
        real_freqs = ceil(200*pars.basis_len/1000);
        A_freq = zeros(pars.basis_len, 1, 1, pars.num_bases);
        A_freq(1:real_freqs,:,:,:) = randn(real_freqs,1,1,pars.num_bases);
        A = real(ifft(A_freq));
        for b=1:pars.num_bases,
          A(:,:,:,b) = A(:,:,:,b) / sqrt(mean(mean(A(:,:,:,b).^2)));
        end
      case 'file',
        load(pars.basis_file,'A');
      case 'gabors',
        % TODO: read sample rate from data
        fs = 22000;
      otherwise,
        error('Invalid setting for pars.basis_source.\n');;
    end

  case 'spec',
    addpath ../../sparsenet/code;
    for ind=1:length(Y_time),
      fs = 16000;
      wintime = pars.wintime;
      hoptime = wintime/2;
      nfft = ceil(fs*wintime);
      WINDOW = hamming(nfft);
      noverlap = nfft-ceil(fs*hoptime);
      [P,F,T] = myspectrogram(Y_time{ind},WINDOW,noverlap, nfft,fs,pars.numfreqs);
      P = abs(P)';
      P = P - mean(mean(P));
      Y{ind} = P/sqrt(mean(mean(P.^2)));
      Y{ind} = reshape(Y{ind}, size(Y{ind},1), 1, size(Y{ind},2));
      fprintf('.');
      if mod(ind,50)==0,
        fprintf('\n');
      end
    end

    switch pars.basis_source,
      case 'random',
        A = randn(pars.basis_len, 1, pars.numfreqs, pars.num_bases);
        for b=1:pars.num_bases,
          A(:,:,:,b) = A(:,:,:,b) / sqrt(mean(mean(A(:,:,:,b).^2)));
        end
      case 'file',
        load(pars.basis_file, 'A');
    end
    
    clear Y_time;
        
        
  case 'spec2',
    for ind = 1:length(Y_time),
      fs = 16000;
      wintime = pars.wintime;
      hoptime = wintime/2;
      nfft = ceil(fs*wintime);
      WINDOW = hamming(nfft);
      noverlap = nfft-ceil(fs*hoptime);
      [P,F,T] = spectrogram(Y_time{ind},WINDOW,noverlap, nfft,fs);
      %F
      P = P .* repmat((1:size(P, 1))', 1, size(P, 2));
      %size(P)
      %pars.numfreqs
      P = abs(P(1:pars.numfreqs, :))';
      P = P - mean(mean(P));
      Y{ind} = P/sqrt(mean(mean(P.^2)));
      Y{ind} = reshape(Y{ind}, size(Y{ind},1), 1, size(Y{ind},2));
      fprintf('.');
      if mod(ind,50)==0,
        fprintf('\n');
      end
      %figure(1); imagesc(P);
      %error
    end
    
    switch pars.basis_source,
      case 'random',
        A = randn(pars.basis_len, 1, pars.numfreqs, pars.num_bases);
        for b=1:pars.num_bases,
          A(:,:,:,b) = A(:,:,:,b) / sqrt(mean(mean(A(:,:,:,b).^2)));
        end
      case 'file',
        load(pars.basis_file, 'A');
    end
    
    clear Y_time;

end


if strcmp(pars.mode,'time'),
  X_train = zeros(pars.patch_len, 1, 1, pars.batch_size, pars.num_batches);
else
  X_train = zeros(pars.patch_len, 1, pars.numfreqs, pars.batch_size, pars.num_batches);
end

switch pars.test_type,
  case 'track',
    tracks_test = num_tracks-pars.num_test+1:num_tracks;
    tracks_train = 1:num_tracks-pars.num_test;
    for ind=1:pars.num_test,
      X_test(:,:,:,ind,1) = Y{tracks_test(ind)};
    end

    [tracks_all, starts_all] = get_patches(Y{tracks_train}, pars);
    num_patches = size(tracks_all, 1);

    perm = randperm(num_patches);
    tracks_all = tracks_all(perm);
    starts_all = starts_all(perm);

    ind = 1;
    for ind1=1:pars.num_batches,
      %tracks(:, ind) = tracks_all((ind-1)*pars.batch_size + (1:pars.batch_size));
      %starts(:, ind) = starts_all((ind-1)*pars.batch_size + (1:pars.batch_size));
      for ind2=1:pars.batch_size,
        X_train(:,:,:,ind2,ind1) = Y{tracks_all(ind)}(starts_all(ind)+(1:pars.patch_len),:,:);
        ind = ind + 1;
      end
    end

  case 'mixed',
    [tracks_all, starts_all] = get_patches(Y, pars);
    num_patches = size(tracks_all, 2);

    rand('seed', 100);
    perm = randperm(num_patches);
    tracks_all = tracks_all(perm);
    starts_all = starts_all(perm);

    ind = 1;
    for ind1=1:pars.num_batches,
      %tracks(:, ind) = tracks_all((ind-1)*pars.batch_size + (1:pars.batch_size));
      %starts(:, ind) = starts_all((ind-1)*pars.batch_size + (1:pars.batch_size));
      for ind2=1:pars.batch_size,
        %fprintf('one');
        %ind
        %size(tracks_all)
        %size(Y)
        %fprintf('two');
        %size(Y{tracks_all(ind)})
        %size(X_train)
        %starts_all(ind)
        %fprintf('three');
        %pars.patch_len
        X_train(:,:,:,ind2,ind1) = Y{tracks_all(ind)}(starts_all(ind)+(1:pars.patch_len),:,:);
        ind = ind + 1;
      end
    end
    for ind2=1:pars.num_test,
      X_test(:,:,:,ind2,1) = Y{tracks_all(ind)}(starts_all(ind)+(1:pars.patch_len),:,:);
      ind = ind + 1;
    end

end

if pars.smooth_edges,
  for b=1:pars.num_batches,
    for p = 1:pars.batch_size,
      X_train(:,:,:,p,b) = smooth(X_train(:,:,:,p,b), pars.basis_len, 2);
    end
  end
  for p=1:pars.num_test,
    X_test(:,:,:,p,1) = smooth(X_test(:,:,:,p,1), pars.basis_len, 2);
  end
end

fprintf('Number of patches: %d\n', num_patches);

clear Y;

if strcmp(pars.mode,'time'),
  display_bases_audio(A,1);
else
  display_bases(reshape(A, size(A,1), size(A,3), 1, size(A,4)),1);
end
drawnow;

for ind = 1:pars.num_batches,
  s_all{ind} = [];
  lambda{ind} = ones(pars.num_bases,1);
end

fobj_test = zeros(pars.num_trials,1);

total_time = 0;
for tr = 1:pars.num_trials,
  batch_id = mod(tr-1,pars.num_batches)+1;

  [A, coef_stats, lambda{1}, s_all{batch_id}] = run_batch(X_train(:,:,:,:,batch_id),A,pars,coef_pars,tr,true,lambda{1});
  total_time = total_time + coef_stats.coef_time_total + coef_stats.bases_time;
  train_time(tr) = total_time/60;
  
  if strcmp(pars.mode,'time'),
    display_bases_audio(A(:,:,:,1:16),1);
  else
    %display_bases(reshape(A,size(A,1), size(A,3), 1, size(A,4)),1);
    display_bases_spec(A, 1);
  end


  if pars.num_test > 0,
    dummy_pars = coef_pars;
    dummy_pars.tile = strcmp(pars.test_type,'track');
    dummy_pars.coeff_iter = 20;
    %dummy_pars.verbosity = 2;
    dummy_pars.tile_iters = 2;
    dummy_pars.exact = true;
    if pars.cache_AtA,
      AtA = get_AtA(A);
    else
      AtA = [];
    end

    size(A)
    size(X_test)

    for ind=1:pars.num_test,
      S = zeros(size(X_test, 1), size(X_test, 2), pars.num_bases);
      [S,coef_stats] = get_responses(X_test(:,:,:,ind,1),A,pars.beta,dummy_pars,0,S,AtA);
      fobj_test(tr) = fobj_test(tr) + coef_stats.f_obj;
    end

  end

  % temporary
  %save('/afs/cs/group/brain/scratch/roger/temp2/temp.mat', 'A', 'S_all', 'X_train');
  % end temporary

  if pars.save_bases_every > 0,
    save(['/afs/cs/group/brain/scratch/roger/temp/' pars.savename '.mat'],'A','fobj_test', 'train_time', 'pars');
    if strcmp(pars.mode,'time'),
      display_bases_audio(A,1);
    else
      %display_bases(reshape(A,size(A,1),size(A,3),1,size(A,4)),1);
      display_bases_spec(A, 1);
    end
    saveas(gcf,['/afs/cs/group/brain/scratch/roger/temp/' pars.savename '_' sprintf('%03d',tr) '.png']);
  end

  figure(3); plot(fobj_test(1:tr));
  drawnow;

end


function [tracks_all, starts_all] = get_patches(Y, pars);

num_tracks = length(Y);
tracks_all = [];
starts_all = [];

for t = 1:num_tracks,
  len = size(Y{t},1);
  starts = 0:(pars.patch_len - pars.patch_overlap):(len-pars.patch_len);
  starts_all = [starts_all starts];
  tracks_all = [tracks_all t*ones(size(starts))];
end


%function X = smooth(X, basis_len);
%
%patch_len = size(X,1);
%num_freqs = size(X,2);
%[C,R] = meshgrid(1:num_freqs, 1:patch_len);
%X = X ./ (1+exp(-4*(R/basis_len)+2));
%X = X ./ (1+exp(-4*((patch_len+1-R)/basis_len)+2));

