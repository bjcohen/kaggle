function B = shift(A,delta);

dr = delta(1,1);
dc = delta(1,2);

dr = mod(dr,size(A,1));
dc = mod(dc,size(A,2));

B = zeros(size(A));

B(1:dr,1:dc,:) = A(size(A,1)-dr+(1:dr),size(A,2)-dc+(1:dc),:);
B(dr+1:size(A,1),1:dc,:) = A(1:size(A,1)-dr,size(A,2)-dc+(1:dc),:);
B(1:dr,dc+1:size(A,2),:) = A(size(A,1)-dr+(1:dr),1:size(A,2)-dc,:);
B(dr+1:size(A,1),dc+1:size(A,2),:) = A(1:size(A,1)-dr,1:size(A,2)-dc,:);

