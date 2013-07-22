function AtA = get_AtA(A);

basis_M = size(A,1);
basis_N = size(A,2);
num_channels = size(A,3);
num_bases = size(A,4);

AtA = zeros(2*basis_M-1, 2*basis_N-1, num_bases*(num_bases+1)/2);

for j = 1:num_bases,
  for i = 1:j,
    for k = 1:num_channels,
      %AtA(:,:,i + j*(j-1)/2) = sum(conv2(A(:,:,:,i), A(basis_M:-1:1,basis_N:-1:1,:,j)),3);
      AtA(:,:,i + j*(j-1)/2) = AtA(:,:,i + j*(j-1)/2) + conv2(A(:,:,k,i), A(basis_M:-1:1,basis_N:-1:1,k,j));
    end
  end
end

