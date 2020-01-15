load images.mat
load DesiredValuesVactor.mat
load test_images.mat

%initializing  filter
filter = [0.01,0;
          0,0.01];
%initializing weights
weights=rand(3,144);
%initialize learning constant
errN = 0.001;
%ep is epoch 
for ep = 1:300

    for x=1:60
    %convolution of image
    conv2ed = conv2(inpM(:,:,x),filter,'valid');

    dlX = dlarray(conv2ed, 'SSCB');
    %maxpooling of image  
    [pooling_Matrix, indx, dataSize] = maxpool(dlX,4,'Stride',4);

    y = extractdata(pooling_Matrix);
    %flattening part
    flattening = reshape(y,[],1);
    [sizeOf, temp] = size(flattening); 
    
    

    %finding V values of image  Vs contains 3V in this case 
    Vs = weights * flattening;
    % y values  
    softmaxVs = softmax(Vs);
    %Backpropagation part !!!   
    S = exp(Vs(1))+exp(Vs(2))+exp(Vs(3));

    %Local gradients
    Souts = CalcSouts(desV(x, 1), S, Vs);

    %Backpropagation update !!! 

    %Finding local gradients of  flattening
    for i=1:sizeOf
        Flatten_gradient(i)=weights(1,i)*Souts(1,1)+weights(2,i)*Souts(2,1)+weights(3,i)*Souts(3,1);
    end

    %updating weights
    weights=Souts.*flattening'*errN+weights; 

    %writing flatteing gradients to indices
    B=reshape(Flatten_gradient,12,12);
    
    C=dlarray(B, 'SSCB');
    dlY = maxunpool(C,indx,dataSize);
    %rotating  Cs
    M90=rot90(dlY);
    M180=rot90(M90);

    %finding delta filter
    delta_filter=conv2(inpM(:,:,x),extractdata(M180),'valid');

    %updating filter
    filter=filter+errN*delta_filter;
    end
end


figure;
subplot(3,3,1),imshow(test(:,:,1));
subplot(3,3,2),imshow(test(:,:,2));
subplot(3,3,3),imshow(test(:,:,3));
subplot(3,3,4),imshow(test(:,:,4));
subplot(3,3,5),imshow(test(:,:,5));
subplot(3,3,6),imshow(test(:,:,6));
subplot(3,3,7),imshow(test(:,:,7));
subplot(3,3,8),imshow(test(:,:,8));

%image is horizantal rectangle if value is 1, vertical rectangle if 2 and
%circle if 3
for testN = 1:8
    value = testM(test(:,:,testN),weights, filter);
    disp(strcat('image is  ','-->', num2str(value)));
end

