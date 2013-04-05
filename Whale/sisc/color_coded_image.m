function ccimg=color_coded_image(X,outmin,outmax,inmin,inmax)

frame_dim1 = size(X,1);
frame_dim2 = size(X,2);
num_colors = size(X,3);

if ~exist('inmin'), inmin = -max(max(max(abs(X)))); end;
if ~exist('inmax'), inmax = max(max(max(abs(X))))+10^-10; end;

if num_colors==1,
  for k=1:3,
    ccimg(:,:,k) = X;
  end
elseif num_colors==2,
  ccimg = zeros(frame_dim1,frame_dim2,3);
  ccimg(:,:,[1 3]) = X;
  ccimg(:,:,2) = 0;
elseif num_colors==3,
  ccimg = X;
else
  colors = zeros(num_colors,3);
  for i=1:num_colors,
    colors(i,:) = reshape(hsv2rgb(i/num_colors,1,1),1,3);
  end
  
  [temp1 temp2] = max(abs(X),[],3);
  ccimg = zeros(frame_dim1,frame_dim2,3);
  for i=1:frame_dim1,
    for j=1:frame_dim2,
      for k=1:3,
        if X(i,j,temp2(i,j)) > 0,
          ccimg(i,j,k) = X(i,j,temp2(i,j))*colors(temp2(i,j),k);
        else
          ccimg(i,j,k) = X(i,j,temp2(i,j))*(1-colors(temp2(i,j),k));
        end
      end
    end
  end
end

ccimg = (ccimg-inmin)/(inmax-inmin);
ccimg = outmin + ccimg*(outmax-outmin);
ccimg = max(ccimg,outmin);
ccimg = min(ccimg,outmax);

if num_colors == 2,
  ccimg(:,:,2) = outmin;
end