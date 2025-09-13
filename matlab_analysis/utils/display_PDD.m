%{
    true_RS_scores - a vector of nx1 with true RS values
    predicted_RS_scores - a vector of nx1 with estimated RS values from
    clinical variables
    
    displays the probability density distribution for true and estimated RS
%}

function display_PDD(true_RS_scores, predicted_RS_scores)

n = length(true_RS_scores);

figure;
hold on;

[f_true, xi_true] = ksdensity(true_RS_scores, 'Bandwidth',3);
plot(xi_true, f_true, 'color', 'r', 'LineWidth', 2.5);

[f_pred, xi_pred] = ksdensity(predicted_RS_scores, 'Bandwidth',3);
plot(xi_pred, f_pred, 'color', 'b', 'LineWidth', 2.5);

RS_M = mean(true_RS_scores, 'omitmissing');
RS_S = std(true_RS_scores, 'omitmissing'); 
ERS_M = mean(predicted_RS_scores, 'omitmissing');
ERS_S = std(predicted_RS_scores, 'omitmissing'); 

t = abs(tinv(0.025, n-1));
RS_MS = RS_S./sqrt(n); % std of mean
RS_M_low = RS_M-t*RS_MS;
RS_M_high = RS_M+t*RS_MS;

fill([RS_M_low, RS_M_high, RS_M_high, RS_M_low], ...
     [0, 0, interp1(xi_true, f_true, RS_M_high), interp1(xi_true, f_true, RS_M_low)], 'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
plot([RS_M RS_M], [0 interp1(xi_true, f_true, RS_M)], '--', 'color', 'r', 'LineWidth', 2);

ERS_MS = ERS_S./sqrt(n); % std of mean
ERS_M_low = ERS_M-t*ERS_MS;
ERS_M_high = ERS_M+t*ERS_MS;
fill([ERS_M_low, ERS_M_high, ERS_M_high, ERS_M_low], ...
     [0, 0, interp1(xi_pred, f_pred, ERS_M_high), interp1(xi_pred, f_pred, ERS_M_low)],'b', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
plot([ERS_M ERS_M], [0 interp1(xi_pred, f_pred, ERS_M)], '--', 'color', 'b', 'LineWidth', 2);


hold off;
% Labels and Legend
xlabel('Score');
ylabel('Density');
legend('Genomic RS', 'Clinical RS');
grid on;
xlim([0 50])
set(gca, 'fontsize', 15)

