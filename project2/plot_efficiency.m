% Load data from text file
data = readmatrix('times.txt');

% First 5 rows: strong scaling, last 5 rows: weak scaling
strong_data = data(1:5, :);
weak_data = data(6:10, :);

% Sort by number of processors for better visualization
strong_data = sortrows(strong_data, 1);
weak_data = sortrows(weak_data, 1);

% Extract variables
p_strong = strong_data(:, 1);
T_strong = strong_data(:, 2);
T1_strong = T_strong(p_strong == 1);
E_strong = T1_strong ./ (p_strong .* T_strong);

p_weak = weak_data(:, 1);
T_weak = weak_data(:, 2);
T1_weak = T_weak(p_weak == 1);
E_weak = T1_weak ./ T_weak;

% Plotting
figure;
subplot(1,2,1);
plot(p_strong, E_strong, 'bo-', 'LineWidth', 2);
title('Strong Scaling Efficiency');
xlabel('Number of Processors');
ylabel('Efficiency');
grid on;

subplot(1,2,2);
plot(p_weak, E_weak, 'ro-', 'LineWidth', 2);
title('Weak Scaling Efficiency');
xlabel('Number of Processors');
ylabel('Efficiency');
grid on;

sgtitle('Parallel Efficiency: Strong vs Weak Scaling');
