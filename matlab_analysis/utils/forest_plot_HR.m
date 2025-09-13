%{
    AI_score - a vector of nx1 with predicted RS values    
    RS - a vector of nx1 with true RS values
    timeList - a vector of nx1 with patient survival times 
    censored - a corresponding vector of nx1. 0 means event
    subgroups - a struct with indicators for the desired subgroups
    f_valid - a vector of nx1 with true for valid patients

    computes HR for AI and RS within each subgroup.
    Displays the forest plot
%}


function forest_plot_HR(AI_score, RS, timeList, censorList,subgroups, f_valid)

nSub = numel(subgroups);
for i = 1:nSub
    if ~subgroups(i).isHeading
        nCount = sum(subgroups(i).filter & f_valid);
        subgroups(i).label = sprintf('%s (%d)', subgroups(i).label, nCount);
    end
end

%% Loop Over Subgroups and Fit Cox Models
HR_AI = nan(nSub,1);  
HR_AI_L = nan(nSub,1);  
HR_AI_U = nan(nSub,1);
HR_RS = nan(nSub,1);  
HR_RS_L = nan(nSub,1);  
HR_RS_U = nan(nSub,1);


for i = 1:nSub
    if subgroups(i).isHeading
        continue;
    end
    
    f_sub = subgroups(i).filter & f_valid;
    
    %------------------------------------------
    % Cox for AI_score
    %------------------------------------------
    X_AI = AI_score(f_sub);
    T_AI = timeList(f_sub);
    C_AI = censorList(f_sub);

    
    [bAI,~,~,statsAI] = coxphfit(X_AI, T_AI, 'Censoring', C_AI);
    ciAI = 1.96 * statsAI.se;
    HR_AI(i)   = exp(bAI);
    HR_AI_L(i) = exp(bAI - ciAI);
    HR_AI_U(i) = exp(bAI + ciAI);

    %------------------------------------------
    % Cox for RS
    %------------------------------------------
    X_RS = RS(f_sub);
    T_RS = timeList(f_sub);
    C_RS = censorList(f_sub);
    
    [bRS,~,~,statsRS] = coxphfit(X_RS, T_RS, 'Censoring', C_RS);
    ciRS = 1.96 * statsRS.se;
    HR_RS(i)   = exp(bRS);
    HR_RS_L(i) = exp(bRS - ciRS);
    HR_RS_U(i) = exp(bRS + ciRS);
end

%% Display the Forest Plot
fig = figure('Color','w');
hold on;

for i = 1:nSub
    yPos = nSub - i + 1;  % top row is i=1

    if subgroups(i).isHeading
        continue
    else
        % Plot AI (purple circle)
        if ~isnan(HR_AI(i))
            xVal  = HR_AI(i);
            xLow  = HR_AI_L(i);
            xHigh = HR_AI_U(i);

            % shift +0.15 in y
            errorbar(xVal, yPos+0.15, ...
                xVal - xLow, xVal - xHigh, ...  % left/right errors
                'horizontal', ...
                'LineWidth',1.8, ...
                'Color','b', ...
                'Marker','o', ...
                'MarkerFaceColor','b', ...   % filled circle
                'MarkerSize',5, ...
                'CapSize',0 ...             % remove end-caps
                );
        end

        % Plot RS (red square)
        if ~isnan(HR_RS(i))
            xVal  = HR_RS(i);
            xLow  = HR_RS_L(i);
            xHigh = HR_RS_U(i);

            % shift -0.15 in y
            errorbar(xVal, yPos-0.15, ...
                xVal - xLow, xVal - xHigh, ...
                'horizontal', ...
                'LineWidth',1.8, ...
                'Color','r', ...
                'Marker','s', ...
                'MarkerFaceColor','r', ...  % filled square
                'MarkerSize',5, ...
                'CapSize',0 ...
                );
        end
    end
end

plot([1 1],[0 nSub+1], '-', 'Color',[0.7 0.7 0.7], 'LineWidth',2);

% Set X-limits to min/max of the computed CIs (if any exist)
allLower = [HR_AI_L; HR_RS_L];
allUpper = [HR_AI_U; HR_RS_U];
validIdx = ~isnan(allLower) & ~isnan(allUpper);
if any(validIdx)
    buffer = 0.1;  % expand edges slightly
    xMin = min(allLower(validIdx)) - buffer;
    xMax = max(allUpper(validIdx)) + buffer;
    if xMin < 0, xMin = 0.01; end  % avoid negative/zero bound if HR is near 0
    xlim([xMin, xMax]);
end
xlim([0.9, 1.25]);
set(gca,'XGrid','on','YGrid','off');

box off;
yticks(1:nSub);
labelStrings = cell(nSub,1);
for i=1:nSub
    labelStrings{i} = subgroups(nSub - i + 1).label;
end
yticklabels(labelStrings);

set(gca,'TickLabelInterpreter','tex');

xlabel('Hazard Ratio (95% CI)','Interpreter','tex');
set(gca, 'FontSize', 13);
hold off;

ylim([0 22])
fig.Position = [1220 369 450 700];

