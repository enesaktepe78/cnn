function [value] = testM(testImage, weight, filter)
%elde ettiðimiz deðerlerin 0-1 aralýðýnda olmamasýndan dolayý
%hangisini alacaðýmýz konusunda emin olamadýk. 
%bu durumda en büyük deðere sahip olan outputun index deðerini 
%tahmin olarak döndürüyoruz.
conv2ed = conv2(testImage,filter,'valid');

dlX = dlarray(conv2ed, 'SSCB');

[pooling_Matrix, indx, dataSize] = maxpool(dlX,4,'Stride',4);

y = extractdata(pooling_Matrix);
flattening = reshape(y,[],1);
Vs = weight * flattening;

softmaxVs = softmax(Vs);

maximum = max(max(softmaxVs));
[x,y]=find(softmaxVs==maximum);
value = x;
end

