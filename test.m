    
    conv2ed = conv2(test(:,:,1),newF,'valid');

    dlX = dlarray(conv2ed, 'SSCB');

    [pooling_Matrix, indx, dataSize] = maxpool(dlX,4,'Stride',4);

    y = extractdata(pooling_Matrix);
    flattening = reshape(y,[],1);
    [sizeOf, temp] = size(flattening);  
    Vs = newW * flattening;

    softmaxVs = softmax(Vs);
   
    S = exp(Vs(1))+exp(Vs(2))+exp(Vs(3));

    SoutsN = CalcSouts(desV(x, 1), S, Vs);
    
