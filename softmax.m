function [softV] = softmax(Vs)
    softV(1,1) = exp(Vs(1))/ (exp(Vs(1))+exp(Vs(2))+exp(Vs(3)));
    softV(2,1) = exp(Vs(2))/ (exp(Vs(1))+exp(Vs(2))+exp(Vs(3)));
    softV(3,1) = exp(Vs(3))/ (exp(Vs(1))+exp(Vs(2))+exp(Vs(3)));
end


