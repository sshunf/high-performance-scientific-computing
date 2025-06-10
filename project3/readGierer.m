function U = readGierer(fname)
% [U,V] = readGierer(fname)
%
% read and display output of the euler project
% Assumes all the data for steps k=0, 1, ..., 10 are present
% Autodetects the grid size from the file size
%
% Input: fname = generic name of data files with a '#' character that
%                will be replaced with U or V. For example, 
%                use readGierer('Gierer#.out') to read GiererU.out, 
%                and GiererV.out
% 
% Output: [U,V] = Last iteration of the data
%

% do U variable
fname1 = strrep(fname,'#','U');
info = dir(fname1);
N = round(sqrt(info.bytes/88));
fprintf('Based on file size, it looks like a %dx%d grid\n', N,N);
x = linspace(-100,100,N+1);
x = x(1:end-1);
[X,Y] = meshgrid(x,x);
fid = fopen(fname1, 'r');
for k=1:11
    U = fread(fid,[N,N],'double');
    subplot(2,11,k)
    contourf(X,Y,U,1:0.02:10.5,'LineStyle','none');
    xlabel(sprintf('U: k=%d',k-1));
    clim([1,11]);
    axis equal
    drawnow
end
fclose(fid);

% do V variable
fname1 = strrep(fname,'#','V');
fid = fopen(fname1, 'r');
for k=1:11
    V = fread(fid,[N,N],'double');
    subplot(2,11,k+11)
    contourf(X,Y,V,1:0.02:10.5,'LineStyle','none');
    xlabel(sprintf('V: k=%d',k-1));
    clim([1,11]);
    axis equal
    drawnow
end
fclose(fid);

end
