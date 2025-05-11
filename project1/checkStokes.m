function [U,V,P] = checkStokes(fname)
% [U,V,P] = readStokes(fname)
%
% read and display output of the stokes project
% Autodetects the grid size from the file size
%
% Input: fname = generic name of data files with a '#' character that
%                will be replaced with U, V, or P. For example, 
%                use readStokes('Stokes#.out') to read StokesU.out,
%                StokesV.out, and StokesP.out
% 
% Output: [R,U,V,P] = Last iteration of the data
%

% do u variable
fname1 = strrep(fname,'#','U');
info = dir(fname1);
N = round((1+sqrt(1+info.bytes/2))/2);
fprintf('Based on file size, it looks like a %dx%d grid\n', N,N-1);
fid = fopen(fname1, 'r');
u = fread(fid,[N-1,N],'double');
fclose(fid);

% do v variable
fname1 = strrep(fname,'#','V');
info = dir(fname1);
N = round((1+sqrt(1+info.bytes/2))/2);
fprintf('Based on file size, it looks like a %dx%d grid\n', N-1,N);
fid = fopen(fname1, 'r');
v = fread(fid,[N,N-1],'double');
fclose(fid);

% do p variable
fname1 = strrep(fname,'#','P');
info = dir(fname1);
N = round(1+sqrt(info.bytes/8));
fprintf('Based on file size, it looks like a %dx%d grid\n', N-1,N-1);
fid = fopen(fname1, 'r');
p = fread(fid,[N-1,N-1],'double');
fclose(fid);

x = linspace(0,1,N);
y = x;
[X,Y] = meshgrid(x,y);
X = X'; Y = Y';
dx = x(2)-x(1);
dy = dx;
mu = 1; fx=0;
uresid = zeros(size(u));
vresid = zeros(size(v));
presid = zeros(size(p));

i = 1;

j = 1;
uresid(i,j) = dy/dx*(-u(i,j)+u(i+1,j))+dx/dy*(-3*u(i,j)+u(i,j+1)) ...
    -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;

for j=2:N-2
    uresid(i,j) = dy/dx*(-u(i,j)+u(i+1,j))+dx/dy*(u(i,j-1)-2*u(i,j)+u(i,j+1)) ...
        -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;
end

j = N-1;
uresid(i,j) = dy/dx*(-u(i,j)+u(i+1,j))+dx/dy*(-3*u(i,j)+u(i,j-1)) ...
    -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;

for i=2:N-2
    
    j = 1;
    uresid(i,j) = dy/dx*(u(i-1,j)-2*u(i,j)+u(i+1,j))+dx/dy*(-3*u(i,j)+u(i,j+1)) ...
        -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;

    for j=2:N-2
        uresid(i,j) = dy/dx*(u(i-1,j)-2*u(i,j)+u(i+1,j))+dx/dy*(u(i,j-1)-2*u(i,j)+u(i,j+1)) ...
            -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;
    end
    
    j = N-1;
    uresid(i,j) = dy/dx*(u(i-1,j)-2*u(i,j)+u(i+1,j))+dx/dy*(-3*u(i,j)+u(i,j-1)) ...
        -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;
end

i = N-1;

j = 1;
uresid(i,j) = dy/dx*(u(i-1,j)-2*u(i,j)+u(i+1,j))+dx/dy*(-3*u(i,j)+u(i,j+1)) ...
    -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;

for j=2:N-2
    uresid(i,j) = dy/dx*(u(i-1,j)-2*u(i,j)+u(i+1,j))+dx/dy*(u(i,j-1)-2*u(i,j)+u(i,j+1)) ...
        -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;
end

j = N-1;
uresid(i,j) = dy/dx*(u(i-1,j)-2*u(i,j)+u(i+1,j))+dx/dy*(u(i,j-1)-3*u(i,j)) ...
    -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;

i = N;

j = 1;
uresid(i,j) = dy/dx*(u(i-1,j)-u(i,j))+dx/dy*(-3*u(i,j)+u(i,j+1)) ...
    -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;

for j=2:N-2
    uresid(i,j) = dy/dx*(u(i-1,j)-u(i,j))+dx/dy*(u(i,j-1)-2*u(i,j)+u(i,j+1)) ...
        -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;
end

j = N-1;
uresid(i,j) = dy/dx*(u(i-1,j)-u(i,j))+dx/dy*(u(i,j-1)-3*u(i,j)) ...
    -dy/mu*(p(i+1,j)-p(i,j))+fx*dx*dy/mu;

% solve v equation

i = 1;

j = 1;
vresid(i,j) = v(i,j);
v(i,j) = 0;

for j=2:N-2
    vresid(i,j) = dy/dx*(-v(i,j)+v(i+1,j))+dx/dy*(v(i,j-1)-2*v(i,j)+v(i,j+1)) ...
        -dx/mu*(p(i+1,j)-p(i+1,j-1))+fy*dx*dy/mu;
end

j = N-1;
vresid(i,j) = dy/dx*(-v(i,j)+v(i+1,j))+dx/dy*(v(i,j-1)-2*v(i,j)+v(i,j+1)) ...
    -dx/mu*(p(i+1,j)-p(i+1,j-1))+fy*dx*dy/mu;

j = N;

for i=2:N-2
    
    j = 1;
    vresid(i,j) = v(i,j);
    
    for j=2:N-2
        vresid(i,j) = dy/dx*(v(i-1,j)-2*v(i,j)+v(i+1,j))+dx/dy*(v(i,j-1)-2*v(i,j)+v(i,j+1)) ...
            -dx/mu*(p(i+1,j)-p(i+1,j-1))+fy*dx*dy/mu;
    end
    
    j = N-1;
    vresid(i,j) = dy/dx*(v(i-1,j)-2*v(i,j)+v(i+1,j))+dx/dy*(v(i,j-1)-2*v(i,j)+v(i,j+1)) ...
        -dx/mu*(p(i+1,j)-p(i+1,j-1))+fy*dx*dy/mu;
        
end

i = N-1;

j = 1;
vresid(i,j) = 0;

for j=2:N-2
    vresid(i,j) = dy/dx*(v(i-1,j)-v(i,j))+dx/dy*(v(i,j-1)-2*v(i,j)+v(i,j+1)) ...
        -dx/mu*(p(i+1,j)-p(i+1,j-1))+fy*dx*dy/mu;
end

j = N-1;
vresid(i,j) = dy/dx*(v(i-1,j)-v(i,j))+dx/dy*(v(i,j-1)-2*v(i,j)+v(i,j+1)) ...
    -dx/mu*(p(i+1,j)-p(i+1,j-1))+fy*dx*dy/mu;

j = N;

% solve p equation

i = 1;

j = 1;
presid(i,j) = 2*P-p(i+1,j)-p(i,j);

for j=2:N-2
    presid(i,j) = 2*P-p(i+1,j)-p(i,j);
end

j = N-1;
presid(i,j) = 2*P-p(i+1,j)-p(i,j);

for i=2:N-2
    
    j = 1;
    presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));    
    for j=2:N-2
        presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));
    end
    
    j = N-1;
    presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));
    
end

i = N-1;

j = 1;
presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));

for j=2:N-2
    presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));
end

j = N-1;
presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));

i = N;

j = 1;
presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));

for j=2:N-2
    presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));
end

j = N-1;
presid(i,j) = -(u(i,j)-u(i-1,j))-dx/dy*(v(i-1,j+1)-v(i-1,j));

% do u variable
subplot(1,3,1)
surf(uresid,'LineStyle','none');
xlabel('u');
drawnow

% do v variable
subplot(1,3,2)
surf(vresid,'LineStyle','none');
xlabel('v');
drawnow

% do p variable
subplot(1,3,3)
surf(presid,'LineStyle','none');
xlabel('p');
drawnow

end
