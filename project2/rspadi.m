function [u,v,w,X,Y] = rpsadi(N,alpha,M)
% rps(N,alpha,M)
% simulate Rock-Paper-Scissors Model
% N = number of grid points in both dimensions
% alpha = depredation rate
% M = number of time steps

T=2000;
dt=T/M;
L=60;
x=linspace(-L,L,N);
beta = 1/(1+alpha);
dx = x(2)-x(1);
[X,Y]= meshgrid(x,x);
u = beta*rand(size(X));
v = beta*rand(size(X));
w = beta*rand(size(X));

D = -2./dx^2*diag(ones(N,1),0)+1./dx^2*diag(ones(N-1,1),1) ...
    +1./dx^2*diag(ones(N-1,1),-1);
D(1,2) = 2/dx^2;
D(N,N-1) = 2/dx^2;

rhou=zeros(size(u));
rhov=rhou;
rhow=rhov;

figure(1)
subplot(1,4,1);
contourf(X,Y,u,0:0.01:1,'LineStyle','none');
colormap('default');
caxis([0,1]);
axis equal
subplot(1,4,2);
contourf(X,Y,v,0:0.01:1,'LineStyle','none');
caxis([0,1]);
axis equal
title(sprintf('Time = %f',0))
subplot(1,4,3);
contourf(X,Y,w,0:0.01:1,'LineStyle','none');
caxis([0,1]);
axis equal
subplot(1,4,4);
contourf(X,Y,u-max(v,w)+2,2:0.01:3,'LineStyle','none');
caxis([0,3]);
hold on
contourf(X,Y,v-max(u,w)+1,1:0.01:2,'LineStyle','none');
contourf(X,Y,w-max(u,v),0:0.01:1,'LineStyle','none');
hold off
axis equal
drawnow

for step=1:M

    % explicit in x
    rhou = u.*(1-u-alpha*w);
    rhov = v.*(1-v-alpha*u);
    rhow = w.*(1-w-alpha*v);
    i = 1;
    for j=1:N
        u2(i,j) = u(i,j) + dt/dx^2*(u(i+1,j)-u(i,j)) + dt/2*rhou(i,j);
        v2(i,j) = v(i,j) + dt/dx^2*(v(i+1,j)-v(i,j)) + dt/2*rhov(i,j);
        w2(i,j) = w(i,j) + dt/dx^2*(w(i+1,j)-w(i,j)) + dt/2*rhow(i,j);
    end
    for i=2:N-1
        for j=1:N
            u2(i,j) = u(i,j) + dt/2/dx^2*(u(i+1,j)-2*u(i,j)+u(i-1,j)) + dt/2*rhou(i,j);
            v2(i,j) = v(i,j) + dt/2/dx^2*(v(i+1,j)-2*v(i,j)+v(i-1,j)) + dt/2*rhov(i,j);
            w2(i,j) = w(i,j) + dt/2/dx^2*(w(i+1,j)-2*w(i,j)+w(i-1,j)) + dt/2*rhow(i,j);
        end
    end
    i = N;
    for j=1:N
        u2(i,j) = u(i,j) + dt/dx^2*(-u(i,j)+u(i-1,j)) + dt/2*rhou(i,j);
        v2(i,j) = v(i,j) + dt/dx^2*(-v(i,j)+v(i-1,j)) + dt/2*rhov(i,j);
        w2(i,j) = w(i,j) + dt/dx^2*(-w(i,j)+w(i-1,j)) + dt/2*rhow(i,j);
    end

    % implicit in y
    u = ((eye(N,N)-dt/2*D)\u2')';
    v = ((eye(N,N)-dt/2*D)\v2')';
    w = ((eye(N,N)-dt/2*D)\w2')';
    
    % explicit in y
    rhou = u.*(1-u-alpha*w);
    rhov = v.*(1-v-alpha*u);
    rhow = w.*(1-w-alpha*v);
    for i=1:N
        j = 1;
        u2(i,j) = u(i,j) + dt/dx^2*(u(i,j+1)-u(i,j)) + dt/2*rhou(i,j);
        v2(i,j) = v(i,j) + dt/dx^2*(v(i,j+1)-v(i,j)) + dt/2*rhov(i,j);
        w2(i,j) = w(i,j) + dt/dx^2*(w(i,j+1)-w(i,j)) + dt/2*rhow(i,j);

        for j=2:N-1
            u2(i,j) = u(i,j) + dt/2/dx^2*(u(i,j+1)-2*u(i,j)+u(i,j-1)) + dt/2*rhou(i,j);
            v2(i,j) = v(i,j) + dt/2/dx^2*(v(i,j+1)-2*v(i,j)+v(i,j-1)) + dt/2*rhov(i,j);
            w2(i,j) = w(i,j) + dt/2/dx^2*(w(i,j+1)-2*w(i,j)+w(i,j-1)) + dt/2*rhow(i,j);
        end

        j = N;
        u2(i,j) = u(i,j) + dt/dx^2*(-u(i,j)+u(i,j-1)) + dt/2*rhou(i,j);
        v2(i,j) = v(i,j) + dt/dx^2*(-v(i,j)+v(i,j-1)) + dt/2*rhov(i,j);
        w2(i,j) = w(i,j) + dt/dx^2*(-w(i,j)+w(i,j-1)) + dt/2*rhow(i,j);
    end 

    % implicit in x
    u = (eye(N,N)-dt/2*D)\u2;
    v = (eye(N,N)-dt/2*D)\v2;
    w = (eye(N,N)-dt/2*D)\w2;

    if mod(step,10) == 0
        figure(1)
        subplot(1,4,1);
        contourf(X,Y,u,0:0.01:1,'LineStyle','none');
        colormap('default');
        caxis([0,1]);
        axis equal
        subplot(1,4,2);
        contourf(X,Y,v,0:0.01:1,'LineStyle','none');
        caxis([0,1]);
        axis equal
        title(sprintf('Time = %f',step*dt))
        subplot(1,4,3);
        contourf(X,Y,w,0:0.01:1,'LineStyle','none');
        caxis([0,1]);
        axis equal
        subplot(1,4,4);
        contourf(X,Y,u-max(v,w)+2,2:0.01:3,'LineStyle','none');
        caxis([0,3]);
        hold on
        contourf(X,Y,v-max(u,w)+1,1:0.01:2,'LineStyle','none');
        contourf(X,Y,w-max(u,v),0:0.01:1,'LineStyle','none');
        hold off
        axis equal
        drawnow
        count = 0;
    end

end

end