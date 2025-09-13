
%{ 
according to the supplementary appendix in MINDACT trial study, they used
a modified version of Adjuvant online. 
For more details see:
"70-Gene Signature as an Aid to Treatment Decisions in Early-Stage Breast
Cancer"
Table S 13

%}

function ClinicalRisk = getClinicalRisk(Grade, TumorSize, label_node)
n = length(Grade);

ClinicalRisk = nan(n, 1);
ClinicalRisk(~isnan(Grade) & ~isnan(TumorSize) & ~isnan(label_node)) = 1;

ClinicalRisk((Grade == 1) & (TumorSize <= 3) & (label_node == 0)) = 0;
ClinicalRisk((Grade == 1) & (TumorSize <= 2) & (label_node == 1)) = 0;
ClinicalRisk((Grade == 2) & (TumorSize <= 2) & (label_node == 0)) = 0;
ClinicalRisk((Grade == 3) & (TumorSize <= 1) & (label_node == 0)) = 0;

%% special cases with nans:
ClinicalRisk((Grade == 1) & (TumorSize <= 2) & isnan(label_node)) = 0;
ClinicalRisk(isnan(Grade) & (TumorSize <= 1) & (label_node == 0)) = 0;
