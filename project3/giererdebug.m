function [u,v]=giererdebug(N,Du,Dv,a,b,c,ep,M)
% [u,v] = giererdebug(N,Du,Dv,a,b,c,ep,M)
% Pseudo-spectral solution of Gierer-Meinhardt equation
% N = number of grid points in both dimensions
% Du, Dv = diffusion rates for activator (Du) and inhibitor (Dv)
% a, b, c, ep = equation parameters
% M = number of time steps
%
% To use:
% 1. In your code add file output for checking each stage of
%    the Runge-Kutta algorithm
%
%   a. After the first stage, save your data this way:
%      
%      FILE* fid = fopen("stage1.out","w");
%      fwrite(u, sizeof(double), N*N, fid);
%      fwrite(v, sizeof(double), N*N, fid);
%      fclose(fid);
%
%      Or if using MPI and with padded arrays, then this way:
%      
%      FILE* fid = fopen("stage1.out","w");
%      fwrite(u, sizeof(double), N*(N+2), fid);
%      fwrite(v, sizeof(double), N*(N+2), fid);
%      fclose(fid);
%
%   b. After the second stage, save your data as above 
%      except using the file name "stage2.out". 
%
%   c. After the third stage, save your data as above
%      except using the file name "stage3.out".
% 
%   d. After the final stage, save your data as above
%      except using the file name "stage4.out" and then after
%      you close the file, add the line
%
%      exit(0);
% 
%      so that you get the first instance of each stage of the 
%      method.
%
% 2. Run your code to generate the output data
%
% 3. Run giererdebug with the same parameters. After each stage, 
%    the code will compare with your output. The plots should have 
%    max < 1e-11. The first stage that doesn't match is where a 
%    problem is. 
%
% 4. You can check your spectral derivatives by saving your spectral
%    coefficients (called ax, ay, and a2 respectively) right before 
%    you use the inverse transform for the first stage this way:
%
%    FILE* fid = fopen("da.out","w");
%    fwrite(a2, sizeof(fftw_complex), N*(N/2+1), fid);
%    fclose(fid);
%

Du = Du/(100/pi)^2;
Dv = Dv/(100/pi)^2;
dt = 100/M;
x=linspace(-100,100,N+1);
x=x(1:end-1);
[X,Y]=meshgrid(x,x);
% u=(a+c)/b*ones(N,N)-4.5*rand(N,N);
% v=(a+c)^2/b^2/c*ones(N,N);

fid = fopen('GiererU.out');
if fid==-1
    fprintf('Unable to find file GiererU.out\n');
    return;
end
info = dir('GiererU.out');
if info.bytes == N*N*8 || info.bytes == N*N*88
    u = fread(fid,[N,N],'double');
    padded = 0;
else
    A = fread(fid,[N+2,N],'double');
    u = A(1:N,N);
    padded = 1;
end
fclose(fid);
fid = fopen('GiererV.out');
if fid==-1
    fprintf('Unable to find file GiererV.out\n');
    return;
end
info = dir('GiererV.out');
if padded == 0
    v = fread(fid,[N,N],'double');
else
    A = fread(fid,[N+2,N],'double');
    v = v(1:N,:);
end
fclose(fid);

figure(1)
contourf(u,[-1:0.05:1]);

%for step=1:M

    % check spectral derivative
    fid = fopen('da.out');
    if fid~=-1
        dU = fftshift(fft2(u));
        d2Ux = -(ones(N,1)*[-N/2:N/2-1].^2).*dU;
        d2Uy = -dU.*(([-N/2:N/2-1]'*ones(1,N)).^2);
        d2U = fftshift(d2Ux+d2Uy);
        d2U = d2U(1:N/2+1,:)/N^2;
        dA = fread(fid,[N+2,N],'double');
        dA = dA(1:2:N+2,:)+i*dA(2:2:N+2,:);
        maxerr = min([max(abs(d2U-dA),[],'all'),max(abs(d2U-dA/N^2),[],'all')]);
        if maxerr > 1e-11
            figure(2)
            surf(abs(d2U(1:N/2+1,:)-dA),'LineStyle','none');
            title('spectral Laplacian error');
            fprintf('Check spectral Laplacian derivative\n');
            return 
        else
            fprintf('Spectral Laplacian derivative of U looks good\n');
        end
        dV = fftshift(fft2(v));
        d2Vx = -(ones(N,1)*[-N/2:N/2-1].^2).*dV;
        d2Vy = -dV.*(([-N/2:N/2-1]'*ones(1,N)).^2);
        d2V = fftshift(d2Vx+d2Vy);
        d2V = d2V(1:N/2+1,:)/N^2;
        dA = fread(fid,[N+2,N],'double');
        dA = dA(1:2:N+2,:)+i*dA(2:2:N+2,:);
        maxerr = min([max(abs(d2V(1:N/2+1,:)-dA),[],'all'),max(abs(d2V(1:N/2+1,:)-dA/N^2),[],'all')]);
        if maxerr > 1e-11
            figure(2)
            surf(abs(d2V-dA),'LineStyle','none');
            title('spectral Laplacian error');
            fprintf('Check spectral Laplacian derivative\n');
            return 
        else
            fprintf('Spectral Laplacian derivative of V looks good\n');
        end
    else
        fprintf('Skipping spectral derivatives check\n');
    end
    fclose(fid);

    % compute RK4
    u1 = u + dt/4*(Du*del2A(u)+a+u.^2./v./(1+ep*u.^2)-b*u);
    v1 = v + dt/4*(Dv*del2A(v)+u.^2-c*v);

    % check stage 1
    fid = fopen('stage1.out');
    if fid ~= -1
        info = dir('stage1.out');
        if info.bytes == N*N*16
            U = fread(fid,[N,N],'double');
            V = fread(fid,[N,N],'double');
            padded = 0;
        elseif info.bytes == N*(N+2)*16
            U = fread(fid,[N+2,N],'double');
            U = U(1:N,:);
            V = fread(fid,[N+2,N],'double');
            V = V(1:N,:);
            padded = 1;
        else
            fprintf('stage1.out file is the wrong size.\n');
            return
        end
        fclose(fid);
        maxerr = max([max(abs(U-u1),[],'all'),max(abs(V-v1),[],'all')]);
        if maxerr > 1e-11
            figure(2)
            subplot(1,2,1)
            surf(U-u1,'LineStyle','none');
            title('stage 1 error for U')
            subplot(1,2,2)
            surf(V-v1,'LineStyle','none');
            title('stage 1 error for V')
            fprintf('Check stage 1\n');
            return 
        else
            fprintf('Stage 1 looks good\n');
        end
    else
        fprintf('Skipping stage 1\n');
    end

    u2 = u + dt/3*(Du*del2A(u1)+a+u1.^2./v1./(1+ep*u1.^2)-b*u1);
    v2 = v + dt/3*(Dv*del2A(v1)+u1.^2-c*v1);

    % check stage 2
    fid = fopen('stage2.out');
    if fid ~= -1
        if padded == 0
            U = fread(fid,[N,N],'double');
            V = fread(fid,[N,N],'double');
        else
            U = fread(fid,[N+2,N],'double');
            U = U(1:N,:);
            V = fread(fid,[N+2,N],'double');
            V = V(1:N,:);
        end
        fclose(fid);
        maxerr = max([max(abs(U-u2),[],'all'),max(abs(V-v2),[],'all')]);
        if maxerr > 1e-11
            figure(2)
            subplot(1,2,1)
            surf(U-u2,'LineStyle','none');
            title('stage 2 error for U')
            subplot(1,2,2)
            surf(V-v2,'LineStyle','none');
            title('stage 2 error for V')
            fprintf('Check stage 1\n');
            return 
        else
            fprintf('Stage 2 looks good\n');
        end
    else
        fprintf('Skipping stage 2\n');
    end

    u1 = u + dt/2*(Du*del2A(u2)+a+u2.^2./v2./(1+ep*u2.^2)-b*u2);
    v1 = v + dt/2*(Dv*del2A(v2)+u2.^2-c*v2);

    % check stage 3
    fid = fopen('stage3.out');
    if fid ~= -1
        if padded == 0
            U = fread(fid,[N,N],'double');
            V = fread(fid,[N,N],'double');
        else
            U = fread(fid,[N+2,N],'double');
            U = U(1:N,:);
            V = fread(fid,[N+2,N],'double');
            V = V(1:N,:);
        end
        fclose(fid);
        maxerr = max([max(abs(U-u1),[],'all'),max(abs(V-v1),[],'all')]);
        if maxerr > 1e-11
            figure(2)
            subplot(1,2,1)
            surf(U-u1,'LineStyle','none');
            title('stage 3 error for U')
            subplot(1,2,2)
            surf(V-v1,'LineStyle','none');
            title('stage 3 error for V')
            fprintf('Check stage 1\n');
            return 
        else
            fprintf('Stage 3 looks good\n');
        end
    else
        fprintf('Skipping stage 3\n');
    end

    u = u + dt*(Du*del2A(u1)+a+u1.^2./v1./(1+ep*u1.^2)-b*u1);
    v = v + dt*(Dv*del2A(v1)+u1.^2-c*v1);

    % check stage 4
    fid = fopen('stage4.out');
    if fid ~= -1
        if padded == 0
            U = fread(fid,[N,N],'double');
            V = fread(fid,[N,N],'double');
        else
            U = fread(fid,[N+2,N],'double');
            U = U(1:N,:);
            V = fread(fid,[N+2,N],'double');
            V = V(1:N,:);
        end
        fclose(fid);
        maxerr = max([max(abs(U-u),[],'all'),max(abs(V-v),[],'all')]);
        if maxerr > 1e-11
            figure(2)
            subplot(1,2,1)
            surf(U-u,'LineStyle','none');
            title('stage 4 error for U')
            subplot(1,2,2)
            surf(V-v,'LineStyle','none');
            title('stage 4 error for V')
            fprintf('Check stage 4\n');
            return 
        else
            fprintf('Stage 4 looks good\n');
        end
    else
        fprintf('Skipping stage 4\n');
    end

%     if mod(step,100) == 0
    figure(1)
    subplot(2,1,1)
    contourf(u,[-1:0.05:1]);
    title('u');
    subplot(2,1,1)
    contourf(v,[-1:0.05:1]);
    title('v');
end

function d2A = del2A(A)
N = size(A,1)/2;
d2A = fftshift(fft2(A));
d2Ax = -(ones(2*N,1)*[-N:N-1].^2).*d2A;
d2Ay = -d2A.*(([-N:N-1]'*ones(1,2*N)).^2);
d2A = ifft2(fftshift(d2Ax+d2Ay));
end