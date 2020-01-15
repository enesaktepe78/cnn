function [Souts] = CalcSouts(desiredValue, S, Vs)
    %calculates gradients 
    switch desiredValue
        case 1
            Souts(1,1) = (exp(Vs(1,1)) * S - exp(Vs(1,1)) * exp(Vs(1,1))) / (S*S);
            Souts(2,1) = (-exp(Vs(1,1)) * exp(Vs(2,1))) / (S * S);
            Souts(3,1) = (-exp(Vs(1,1)) * exp(Vs(3,1))) / (S * S);
        case 2
            Souts(1,1) = (-exp(Vs(2,1)) * exp(Vs(1,1))) / ( S*S) ;
            Souts(2,1) = (exp(Vs(2,1)) * S - exp(Vs(2,1)) * exp(Vs(2,1))) / ( S*S) ;
            Souts(3,1) = (-exp(Vs(2,1)) * exp(Vs(3,1))) / (S * S);
        case 3
            Souts(1,1) = (-exp(Vs(1,1)) * exp(Vs(3,1))) / ( S*S) ;
            Souts(2,1) = (-exp(Vs(2,1)) * exp(Vs(3,1))) / ( S*S) ;
            Souts(3,1) = (exp(Vs(3,1)) * S - exp(Vs(3,1)) * exp(Vs(3,1))) / ( S*S) ;
    end
    
     yc = softmax(Vs);
     Souts = Souts * ( 1 / yc(desiredValue));
end

