function h=display_bases(A,figstart,color_code)
%
%  display_network -- displays the state of the network (weights and 
%                     output variances)
%
%  Usage:
%
%    h=display_network(A,S_var,h);
%
%    A = basis function matrix
%    S_var = vector of coefficient variances
%    h = display handle (optional)

if nargin < 2
   figstart = 1; 
end

if ~exist('color_code'), color_code = false; end;

figure(figstart)

colormap(gray)

[M N num_channels num_bases]=size(A);

if color_code,
  buf=5;
else
  buf=1;
end
  
  
%if floor(sqrt(num_bases))^2 ~= num_bases
%  m=ceil(sqrt(num_bases/2));
%  n=ceil(num_bases/m);
%else
%  m=sqrt(num_bases);
%  n=m;
%end

best = Inf;
for m_=1:num_bases,
  n_ = ceil(num_bases/m_);
  score = M*m_+N*n_;
  if score < best,
    best = score;
    m=m_;
    n=n_;
  end
end

array=-ones(buf+m*(M+buf),buf+n*(N+buf),3);  

k=1;

for i=1:m
  for j=1:n
    if k>num_bases continue; end
    clim=max(abs(A(:,k)));
    
    if color_code,
      temp = zeros(M+4,N+4,num_bases);
      temp(:,:,k) = ones(M+4,N+4);
      array(buf+(i-1)*(M+buf)+[-1:M+2],buf+(j-1)*(N+buf)+[-1:N+2],:) = color_coded_image(temp,-1,1);
    end
    
    array(buf+(i-1)*(M+buf)+[1:M],buf+(j-1)*(N+buf)+[1:N],:) = color_coded_image(A(:,:,:,k),-1,1);
    
    k=k+1;
  end
end

array = (array+1)/2;

h=image(array);
axis image off


drawnow

figure(figstart)
