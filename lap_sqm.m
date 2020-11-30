function [g1, ind] = lap_sqm(x1,y1,z1, x2,y2,z2, g2, cost_fun, sign, implementation)

    cf = cost_fun(1);
    if numel(cost_fun) > 1
        param = cost_fun(2); % 1/n or gamma or f0
    else
        param = 1;
    end
    if numel(cost_fun) > 2
        param2 = cost_fun(3); 
    else
        param2 = 1;
    end
    if nargin < 10
        implementation = 'avx512-single';
    end

    res_add = 0;
    res_multiple = 1;
    resexp = false;
    fastlog = true;
    switch cf
        case 16     % ((x1-x2)^2 + (y1-y2)^2)/(2 param)
            cost_fun_internal = 10;
            g2 = g2 * param - (x2.^2 + y2.^2) / 2;
            res_add = -(x1.^2 + y1.^2) / (2*param);
            res_multiple = 1 / param;
        case 6    % ((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)/(2 param)
            cost_fun_internal = 0;
            g2 = g2 * param - (x2.^2 + y2.^2 + z2.^2) / 2;
            res_add = -(x1.^2 + y1.^2 + z1.^2) / (2*param);
            res_multiple = 1 / param;
        case 10		% -(x1 x2 + y1 y2)
            cost_fun_internal = 10;
        case 0		% -(x1 x2 + y1 y2 + z1 z2)
            cost_fun_internal = 0;
        case 1		% -log(1-param(x1 x2 + y1 y2 + z1 z2))
            if fastlog % g1 = minmax  g2/(1-param ...)
                x1 = x1 .* param;
                y1 = y1 .* param;
                z1 = z1 .* param;
                cost_fun_internal = 101;
            else % -log(g1) = maxmin  -log(g2) - -log(1-param ...)
                resexp = true;
                x1 = x1 .* param;
                y1 = y1 .* param;
                z1 = z1 .* param;
                sign = -sign;
                g2 = -log(g2);
                cost_fun_internal = 1;
            end
        case 2	    % sqrt(param^2 + (x1-x2)^2 + (y1-y2)^2)
            x1 = x1 ./ param;
            y1 = y1 ./ param;
            z1 = z1 ./ param;
            x2 = x2 ./ param;
            y2 = y2 ./ param;
            z2 = z2 ./ param;
            g2 = g2 ./ param;
            res_multiple = param;
            cost_fun_internal = 2;
        case 3		% -sqrt(param^2 - param2*[(x1-x2)^2 - (y1-y2)^2])
            x1 = x1 .* (sqrt(param2) / param);
            y1 = y1 .* (sqrt(param2) / param);
            x2 = x2 .* (sqrt(param2) / param);
            y2 = y2 .* (sqrt(param2) / param);
            g2 = g2 ./ param;
            res_multiple = param;
            cost_fun_internal = 3;
        case 4		% sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
            cost_fun_internal = 4;
        case 5		% - param * sqrt((z1-z2)^2 - param2*(x1-x2)^2 - param2*(y1-y2)^2 )
            assert(param2>0);
            x1 = x1 .* param .* sqrt(param2);
            y1 = y1 .* param .* sqrt(param2);
            x2 = x2 .* param .* sqrt(param2);
            y2 = y2 .* param .* sqrt(param2);
            z1 = z1 .* param;
            z2 = z2 .* param;
            cost_fun_internal = 5;
        case 11		% ( x1 x2 + y1 y2)/(param-z2) 
            x2 = x2 ./ (param-z2);
            y2 = y2 ./ (param-z2);
            cost_fun_internal = 10;
        case 12		% ( x1 x2 + y1 y2)/(param-z1) 
            x1 = x1 ./ (param-z1);
            y1 = y1 ./ (param-z1);
            cost_fun_internal = 10;
    end

    switch implementation 
        case 'good-old' 
            [g1, ind] = lap_sqm_mex(x1(:), y1(:), z1(:), x2(:), y2(:), z2(:), g2(:), cost_fun_internal, sign);     
        case 'avx512';
            [g1, ind] = lap_sqm_mex_avx512(x1(:), y1(:), z1(:), x2(:), y2(:), z2(:), g2(:), cost_fun_internal, sign);
        case 'avx512-single';
            [g1, ind] = lap_sqm_mex_avx512_single(x1(:), y1(:), z1(:), x2(:), y2(:), z2(:), g2(:), cost_fun_internal, sign);
    end
    g1 = g1 .* res_multiple + res_add;
    if resexp
        g1 = exp(-g1);
    end
    g1 = reshape(g1, size(x1));
    ind = reshape(ind, size(x1));
end

%%

%mex -v -largeArrayDims lap_sqm.cpp COMPFLAGS="/O3 /Qopt-report:5 /Qopt-report-file=optrep.txt /Qip /Qrestrict /QxHost /Qopenmp /Qopenmp-simd $COMPFLAGS"

%%
%{
1.
mex -v -largeArrayDims lap_sqm.cpp COMPFLAGS="/O3 /Qprof-gen /Qprof-dir D:\_dyn_dump\ /Qip /Qrestrict /QxHost /Qopenmp /Qopenmp-simd $COMPFLAGS"

2.
run auctionAlgorithmMex_Opt4
clear all

3.
mex -v -largeArrayDims auctionAlgorithmMex_Opt5.cpp COMPFLAGS="/O3 /Qprof-use /Qprof-dir D:\_dyn_dump\ /Qip /Qrestrict /QxHost /Qopenmp /Qopenmp-simd $COMPFLAGS"

%}




