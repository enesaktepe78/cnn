load images.mat
load test_images.mat
% take the first image
%döngü eklendi
firstImage = inpM(:,:,1);
filter = [1,0;
          0,1];
      
weights=ones(3,144)/100;
      
for x=1:12 
conv2ed = conv2(firstImage,filter,'valid');
%cleared = clearPadding(conv2ed);

dlX = dlarray(conv2ed, 'SSCB');

[pooling_Matrix, indx, dataSize] = maxpool(dlX,4,'Stride',4);

y = extractdata(pooling_Matrix);
flattening = reshape(y,[],1);
[sizeOf, temp] = size(flattening); 
%weights = rand(3, sizeOf);

Vs = weights * flattening;
softmaxVs = softmax(Vs);

%Backpropagation
S=sum(softmaxVs);
grad(1)=0.05;
grad(2)=-0.05;
grad(3)=-0.04;

%finding local gradient flattening
for i=1:sizeOf
    F_gradient(i)=weights(1,i)*grad(1)+weights(2,i)*grad(2)+weights(3,i)*grad(3);
end

%updating weights
weights=grad'.*flattening'*(1/100)+weights; 

%writing flatteing gradients to indices
B=reshape(F_gradient,12,12);
C=dlarray(B, 'SSCB');
dlY = maxunpool(C,indx,dataSize);
%rotating  Cs
M90=rot90(dlY);
M180=rot90(M90);

%finding delta filter
delta_filter=conv2(firstImage,extractdata(M180),'valid');

%updating filter
filter=filter+(1/100)*delta_filter;

end








