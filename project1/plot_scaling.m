data = load('scaling_data.txt');
N = data(:,1);
T = data(:,2);

figure;
loglog(N, T, 'o-','LineWidth',1.5);
grid on;
xlabel('Grid size N');
ylabel('Time to completion (s)');
title('Scaling study: log–log of time vs N');

% Fit a power law T = C * N^alpha
p = polyfit(log(N), log(T), 1);
alpha = p(1);
C = exp(p(2));
hold on;
loglog(N, C * N.^alpha, '--');
legend('measured','fit','Location','best');

fprintf('Fitted scaling exponent α = %.3f\n', alpha);
