%{
    label_RS - a vector of nx1 with true RS values    
    AI_score_only_HE - a vector of nx1 with predicted RS values using only
    H&E images
    AI_score_combined - a vector of nx1 with predicted RS values using 
    H&E + clinical variables
    
    XCrit - measure for X axis (e.g. 'fpr')
    YCrit - measure for Y axis (e.g. 'tpr')
    
    Displays the performance curve and computes AUC
%}

function plot_performance_curve(label_RS, AI_score_only_HE, AI_score_combined, XCrit, YCrit)

n = length(label_RS);
[~,~,~,AUC_only_HE] = perfcurve(label_RS, AI_score_only_HE, 1, 'NBoot',1000, 'XCrit',XCrit, 'YCrit',YCrit);
fprintf('AUC only-HE=%.3f (95%% CI: %.3f–%.3f)\n', AUC_only_HE)

[~,~,~,AUC_combined] = perfcurve(label_RS, AI_score_combined, 1, 'NBoot',1000, 'XCrit',XCrit, 'YCrit',YCrit);
fprintf('AUC combined=%.3f (95%% CI: %.3f–%.3f)\n', AUC_combined)



[ROC_X_only_HE, ROC_Y_only_HE] = perfcurve(label_RS, AI_score_only_HE, 1, 'XCrit',XCrit, 'YCrit',YCrit);
[ROC_X_combined, ROC_Y_combined] = perfcurve(label_RS, AI_score_combined, 1, 'XCrit',XCrit, 'YCrit',YCrit);

%%
figure;
hold on;



if strcmp(XCrit, 'fpr') && strcmp(YCrit, 'tpr')
    p1 = plot(ROC_X_only_HE, ROC_Y_only_HE, 'Color','b', 'LineWidth',2.5, 'DisplayName',sprintf('Only H&E: AUC=%.3f', AUC_only_HE(1)));
    p2 = plot(ROC_X_combined, ROC_Y_combined, 'Color','r', 'LineWidth',2.5, 'DisplayName',sprintf('Combined: AUC=%.3f', AUC_combined(1)));
    plot([0 1], [0 1], "color", [0.7 0.7 0.7], 'LineWidth',2)
    legend([p1, p2], 'location', 'southeast', 'fontsize', 15);
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
elseif strcmp(XCrit, 'reca') && strcmp(YCrit, 'prec')
    p2 = plot(ROC_X_combined, ROC_Y_combined, 'Color','r', 'LineWidth',2.5, 'DisplayName',sprintf('AUC=%.3f', AUC_combined(1)));
    baseline = sum(label_RS)/n;
    p_base = plot([0 1], [baseline baseline], "color", [0.8 0.8 0.8], 'LineWidth',4,'DisplayName',sprintf('Baseline=%.3f', baseline));
    legend([p2 p_base], 'location', 'northeast', 'fontsize', 15);
    xlabel('Recall')
    ylabel('Precision')
elseif strcmp(XCrit, 'spec') && strcmp(YCrit, 'npv')
    p2 = plot(ROC_X_combined, ROC_Y_combined, 'Color','r', 'LineWidth',2.5, 'DisplayName',sprintf('AUC=%.3f', AUC_combined(1)));
    baseline = sum(~label_RS)/n;
    p_base = plot([0 1], [baseline baseline], "color", [0.8 0.8 0.8], 'LineWidth',4,'DisplayName',sprintf('Baseline=%.3f', baseline));
    legend([p2 p_base], 'location', 'southeast', 'fontsize', 15);
    xlabel('Specificity')
    ylabel('Negative Predictive Value')
end




set(gca, 'fontsize', 20)
axis equal
hold off;
axis([0 1 0 1])
xticks([0 0.5 1])
yticks([0 0.5 1])
