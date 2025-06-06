function U = readRPS(fname)
% [U,V,W] = readRPS(fname)
%
% read and display output of the euler project
% Assumes all the data for steps k=0, 1, ..., 10 are present
% Autodetects the grid size from the file size
%
% Input: fname = generic name of data files with a '#' character that
%                will be replaced with U, V, or W. For example, 
%                use readRPS('RPS#.out') to read RPSU.out, RPSV.out
%                and RPSW.out.
% 
% Output: [U,V,W] = Last iteration of the data
%

% do U variable
fname1 = strrep(fname,'#','U');
info = dir(fname1);
N = round(sqrt(info.bytes/88));
fprintf('Based on file size, it looks like a %dx%d grid\n', N,N);
x = linspace(-60,60,N);
[X,Y] = meshgrid(x,x);
fid = fopen(fname1, 'r');
for k=1:11
    U = fread(fid,[N,N],'double');
    subplot(4,11,k)
    contourf(X,Y,U,0:0.01:1,'LineStyle','none');
    xlabel(sprintf('U: k=%d',k-1));
    caxis([0,1]);
    axis equal
    drawnow
end
fclose(fid);

% do V variable
fname1 = strrep(fname,'#','V');
fid = fopen(fname1, 'r');
for k=1:11
    V = fread(fid,[N,N],'double');
    subplot(4,11,k+11)
    contourf(X,Y,V,0:0.01:1,'LineStyle','none');
    xlabel(sprintf('V: k=%d',k-1));
    caxis([0,1]);
    axis equal
    drawnow
end
fclose(fid);

% do W variable
fname1 = strrep(fname,'#','W');
fid = fopen(fname1, 'r');
for k=1:11
    W = fread(fid,[N,N],'double');
    subplot(4,11,k+22)
    contourf(X,Y,W,0:0.01:1,'LineStyle','none');
    xlabel(sprintf('W: k=%d',k-1));
    caxis([0,1]);
    axis equal
    drawnow
end
fclose(fid);

% do combined variable
fnameU = strrep(fname,'#','U');
fidU = fopen(fnameU, 'r');
fnameV = strrep(fname,'#','V');
fidV = fopen(fnameV, 'r');
fnameW = strrep(fname,'#','W');
fidW = fopen(fnameW, 'r');
for k=1:11
    U = fread(fidU,[N,N],'double');
    V = fread(fidV,[N,N],'double');
    W = fread(fidW,[N,N],'double');
    subplot(4,11,k+33)
    contourf(X,Y,U-max(V,W)+2,2:0.01:3,'LineStyle','none');
    caxis([0,3]);
    hold on
    contourf(X,Y,V-max(U,W)+1,1:0.01:2,'LineStyle','none');
    contourf(X,Y,W-max(U,V),0:0.01:1,'LineStyle','none');
    xlabel(sprintf('p: k=%d',k-1));
    axis equal
    hold off
    drawnow
end
fclose(fidU);
fclose(fidV);
fclose(fidW);

end
