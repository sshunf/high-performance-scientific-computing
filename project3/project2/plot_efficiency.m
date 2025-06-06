procs_strong = [1, 2, 4, 8, 16];
times_strong = [48.808913, 18.367957, 13.868888, 9.044261, 6.424708];
T1_strong = times_strong(1);

strong_eff = (T1_strong ./ (procs_strong .* times_strong)) * 100;

procs_weak = [1, 4, 9, 9, 16, 25];
times_weak = [2.306465, 4.566021, 4.251426, 8.435974, 5.875494, 8.978110];

T1_weak = times_weak(1);

weak_eff = (T1_weak ./ times_weak) * 100;

figure;
hold on;

plot(procs_strong, strong_eff, 's--', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Strong Efficiency');
plot(procs_weak, weak_eff, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Weak Efficiency');

xlabel('Number of Processes');
ylabel('Efficiency (%)');
title('Strong vs Weak Efficiency');
legend('Location', 'northeast');
grid on;
ylim([0 110]);

for i = 1:length(procs_strong)
    text(procs_strong(i), strong_eff(i) + 3, sprintf('%.1f%%', strong_eff(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'blue');
end

for i = 1:length(procs_weak)
    text(procs_weak(i), weak_eff(i) + 3, sprintf('%.1f%%', weak_eff(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'black');
end

hold off;
