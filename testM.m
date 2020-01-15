function [value] = testM(testImage, weight, filter)

conv2ed = conv2(testImage,filter,'valid');

dlX = dlarray(conv2ed, 'SSCB');

[pooling_Matrix, indx, dataSize] = maxpool(dlX,4,'Stride',4);

y = extractdata(pooling_Matrix);
flattening = reshape(y,[],1);
Vs = weight * flattening;

softmaxVs = softmax(Vs);
value = softmaxVs;
end

