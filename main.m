load images.mat
load test_images.mat
% take the first image
firstImage = inpM(:,:,1);
filter = [1,0;
          0,1];
      
conv2ed = conv2(firstImage,filter,'valid');
%cleared = clearPadding(conv2ed);

dlX = dlarray(conv2ed, 'SSCB');

[pooling_Matrix, indx, dataSize] = maxpool(dlX,4,'Stride',4);

y = extractdata(pooling_Matrix);
flattening = reshape(y,[],1);
[sizeOf, temp] = size(flattening); 
%weights = rand(3, sizeOf);
weights=ones(3,sizeOf)/10;

Vs = weights * flattening;
softmaxVs = softmax(Vs);

%Backpropagation
S=sum(softmaxVs);
grad(1)=-1*(exp(Vs(1))*exp(Vs(2)))/(S^2);
grad(2)=-1*(exp(Vs(2))*(S-exp(Vs(2))))/S^2;
grad(3)=-1*(exp(Vs(2))*exp(Vs(3)))/S^2;

%finding local gradient flattening
for i=1:sizeOf
    F_gradient(i)=weights(1,i)*grad(1)+weights(2,i)*grad(2)+weights(3,i)*grad(3);
end

%updating weights
w_up=grad'.*flattening'*(1/10)+weights; 

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
update_filter=filter+(1/10)*delta_filter;

%deneme amaçlý
asdf=conv2(firstImage,update_filter,'valid');








