load images.mat
load DesiredValuesVactor.mat
load test_images.mat
format short g
digits(10)
% take the first image
%döngü eklendi
firstImage = inpM(:,:,1);
% filter = [0.01,0;
%           0,0.01];
filter = [ 1, 0 ; 0, 1 ];
weights=ones(3,144);
firstW = weights;
precision = 6;   
errN = 0.001;
for ep = 1:300

    for x=1:60

    conv2ed = conv2(inpM(:,:,x),filter,'valid');

    dlX = dlarray(conv2ed, 'SSCB');

    [pooling_Matrix, indx, dataSize] = maxpool(dlX,4,'Stride',4);

    y = extractdata(pooling_Matrix);
    flattening = reshape(y,[],1);
    [sizeOf, temp] = size(flattening); 
    %weights = rand(3, sizeOf);

    %V values of.. 
    Vs = weights * flattening;
    % y deðerleri 
    softmaxVs = softmax(Vs);
    %%%%%%%%%%%%%%%%%%%
%     for i = 1:3
%         tempVs = num2str(Vs(i,1));
%         a =  '0000000';
%         tempVs = strcat(tempVs(1:3), a);
%         Vs(i,1) = str2double(tempVs(1:5));
%         %Vs(i,1) = round(Vs(i,1),5);
%     end


    %%%%%%%%%%%%%%
    S = exp(Vs(1))+exp(Vs(2))+exp(Vs(3));

%     tempS = num2str(S);
%     tempS = strcat(tempS(1:3), '0000000');
%     S = str2double(tempS(1:5));
    Souts = CalcSouts(desV(x, 1), S, Vs);

    %Backpropagation update !!! 

    %finding local gradient flattening
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
    % M90=rot90(dlY);
    % M180=rot90(M90);

    %finding delta filter
    delta_filter=conv2(inpM(:,:,x),extractdata(dlY),'valid');

    %updating filter
    filter=filter+errN*delta_filter;

    end
end


final = testM(test(:,:,8),weights, filter);
