

function display_NPV(NPV_X_vec, NPV_Y_vec, Patients_X_vec, patients_VS_AI, th, Ty, Py)

NPV_X_vec = [0; NPV_X_vec; 100];
NPV_Y_vec = [NPV_Y_vec(1,1); NPV_Y_vec(:, 1); NPV_Y_vec(end, 1)];

figure;
hold on;
plot([th th], [0 1], 'color', [0.7 0.7 0.7], 'LineWidth',2)

yyaxis left
set(gca, 'YColor', 'k', 'fontsize', 20);
plot(NPV_X_vec, NPV_Y_vec, 'Color', 'r', 'LineWidth', 2.5);
yticks(0:0.1:1); % Define tick positions
h = ylabel('Negative Predictive Value', 'Color', 'k', 'fontsize', 20);
[x_line, y_line] = get_xy_line(gca, h, 'left');
annotation('line', [x_line, x_line], [y_line - 0.09, y_line]-0.02, 'LineWidth', 4, 'Color', 'r');

yyaxis right
set(gca, 'YColor', 'k', 'fontsize', 20);
plot(Patients_X_vec, patients_VS_AI, 'Color', 'b', 'LineWidth', 2.5);
yticks(0:0.1:1); % Define tick positions
yticklabels(0:10:100)
h = ylabel('% Impacted Patients', 'Color', 'k', 'fontsize', 20);
[x_line, y_line] = get_xy_line(gca, h, 'right');
annotation('line', [x_line, x_line], [y_line - 0.09, y_line]-0.02, 'LineWidth', 4, 'Color', 'b');



plot(th, Ty, 'o', 'MarkerSize', 14, 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');
str_result = sprintf('%.3f', Ty);
text(th-1, Ty-0.02, str_result, 'color', 'r', 'FontSize', 20, 'HorizontalAlignment', 'right','VerticalAlignment', 'top');


plot(th, Py, 'o', 'MarkerSize', 14, 'LineWidth', 2, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');
str_result = sprintf('%.1f%%', Py*100);
text(th-1, Py+0.02, str_result, 'color', 'b', 'FontSize', 20, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

xlabel('Binarization Threshold', 'fontsize', 20)
xticks(0:10:100)
axis([0 60 0 1])
grid on
hold off;

