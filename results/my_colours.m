function [ c ] = my_colours( k )
%MY_COLOURS returns a colour based on the entered integer input.

% colours = get( gca, 'colororder' )
colours = [0    0.4470    0.7410;
        0.8500    0.3250    0.0980;
        0.9290    0.6940    0.1250;
        0.4940    0.1840    0.5560;
        0.4660    0.6740    0.1880;
        0.3010    0.7450    0.9330;
        0.6350    0.0780    0.1840];
    
n_col = size(colours, 1);

if( k > n_col )
    fprintf( '\n\n Colour pallete does not support this many colours \n\n' )
end

k = mod(k, n_col+1);
if( k == 0)
    k = 1;
end
    
c = colours(k,:);

end

