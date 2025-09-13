function BACC = compute_BACC(label_pred, label_true)

a = sum(label_pred == 1 & label_true == 1)/sum(label_true == 1);
b = sum(label_pred == 0 & label_true == 0)/sum(label_true == 0);
BACC = (a+b)/2;
