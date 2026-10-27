# =============================
# LOAD PACKAGES
# =============================
install.packages("randomForest")
install.packages("lightgbm")
install.packages("pROC")

library(randomForest)
library(lightgbm)
library(pROC)

# =============================
# LOAD DATA
# =============================
data = read.csv("motor_vehicle_insurance.csv")

# =============================
# STEP 1: TARGET VARIABLE
# =============================
data$Buy = ifelse(data$Lapse == 0, 1, 0)
data$Buy = as.factor(data$Buy)

# Get proportions
class_counts = table(data$Buy)
class_prop = prop.table(class_counts)

# Barplot
bp = barplot(class_counts,
             col = c("red", "blue"),
             names.arg = c("Lapse (0)", "Renew (1)"),
             main = "Class Imbalance in Motor Insurance",
             ylab = "Count")

# Add percentage labels
text(x = bp,
     y = class_counts,
     label = paste0(round(class_prop * 100, 1), "%"),
     pos = 3)

# =============================
# STEP 2: FEATURE ENGINEERING
# =============================
# Convert dates
data$Date_birth = as.Date(data$Date_birth, format="%d/%m/%Y")
data$Date_driving_licence = as.Date(data$Date_driving_licence, format="%d/%m/%Y")
data$Date_start_contract = as.Date(data$Date_start_contract, format="%d/%m/%Y")

# Core features
data$Age = as.numeric(format(data$Date_start_contract, "%Y")) - 
  as.numeric(format(data$Date_birth, "%Y"))

data$Experience = as.numeric(format(data$Date_start_contract, "%Y")) - 
  as.numeric(format(data$Date_driving_licence, "%Y"))

# Keep ONLY 2 strong engineered features
data$Claim_Frequency = data$N_claims_history / (data$Seniority + 1)

data$Premium_per_Value = data$Premium / (data$Value_vehicle + 1)

# =============================
# STEP 3: SELECT VARIABLES (SAME FOR ALL MODELS)
# =============================
model_data = data[, c("Buy",
                      "Premium",
                      "Age", "Experience",
                      "Seniority", "N_claims_history",
                      "Claim_Frequency",
                      "Power", "Value_vehicle",
                      "Premium_per_Value",
                      "Payment", "Area", "Type_risk",
                      "Second_driver", "Policies_in_force")]

# Convert categorical
model_data$Area = as.factor(model_data$Area)
model_data$Type_risk = as.factor(model_data$Type_risk)
model_data$Payment = as.factor(model_data$Payment)
model_data$Second_driver = as.factor(model_data$Second_driver)

model_data = na.omit(model_data)

# =============================
# STEP 4: TRAIN-TEST SPLIT
# =============================
set.seed(2012)
sample_data = sample(nrow(model_data), 0.7*nrow(model_data))

train_data = model_data[sample_data, ]
test_data  = model_data[-sample_data, ]

# =============================
# MODEL 1: LOGISTIC REGRESSION
# =============================
logit_model = glm(Buy ~ ., data = train_data, family = "binomial")

p_logit_prob = predict(logit_model, test_data, type = "response")
p_logit = ifelse(p_logit_prob > 0.5, 1, 0)
p_logit = as.factor(p_logit)

# =============================
# MODEL 2: RANDOM FOREST
# =============================
rf_model = randomForest(Buy ~ ., data = train_data, ntree = 500)

p_rf = predict(rf_model, test_data)
p_rf_prob = predict(rf_model, test_data, type = "prob")[,2]

# =============================
# MODEL 3: LIGHTGBM
# =============================
train_matrix = as.matrix(train_data[, -1])
test_matrix  = as.matrix(test_data[, -1])

train_label = ifelse(train_data$Buy == "1", 1, 0)
test_label  = ifelse(test_data$Buy == "1", 1, 0)

dtrain = lgb.Dataset(data = train_matrix, label = train_label)

params = list(
  objective = "binary",
  metric = "auc",
  learning_rate = 0.03,
  num_leaves = 31,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  scale_pos_weight = sum(train_label == 0) / sum(train_label == 1)
)

lgb_model = lgb.train(
  params = params,
  data = dtrain,
  nrounds = 500
)

p_lgb_prob = predict(lgb_model, test_matrix)
p_lgb = ifelse(p_lgb_prob > 0.5, 1, 0)
p_lgb = as.factor(p_lgb)

# =============================
# LIGHTGBM FEATURE IMPORTANCE
# =============================

importance = lgb.importance(lgb_model)

print(importance)

# Plot top 10 most important features
lgb.plot.importance(
  importance,
  top_n = 10,
  measure = "Gain"
)

# =============================
# EVALUATION FUNCTION
# =============================
evaluate_model = function(actual, predicted, prob = NULL){
  
  cm = table(actual, predicted)
  
  TN = cm[1,1]
  FP = cm[1,2]
  FN = cm[2,1]
  TP = cm[2,2]
  
  accuracy    = (TP + TN) / sum(cm)
  precision   = TP / (TP + FP)
  recall      = TP / (TP + FN)
  specificity = TN / (TN + FP)
  F1          = 2 * (precision * recall) / (precision + recall)
  
  if(!is.null(prob)){
    auc_value = auc(roc(as.numeric(as.character(actual)), prob))
  } else {
    auc_value = NA
  }
  
  return(data.frame(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    F1 = F1,
    AUC = auc_value
  ))
}

# =============================
# RESULTS
# =============================
result_logit = evaluate_model(test_data$Buy, p_logit, p_logit_prob)
result_rf    = evaluate_model(test_data$Buy, p_rf, p_rf_prob)
result_lgb   = evaluate_model(test_data$Buy, p_lgb, p_lgb_prob)

results = rbind(
  cbind(Model = "Logistic", result_logit),
  cbind(Model = "Random Forest", result_rf),
  cbind(Model = "LightGBM", result_lgb)
)

results

# =============================
# ROC CURVE
# =============================
actual = as.numeric(as.character(test_data$Buy))

roc_logit = roc(actual, p_logit_prob)
roc_rf    = roc(actual, p_rf_prob)
roc_lgb   = roc(actual, p_lgb_prob)

plot(roc_logit, col = "blue", lwd = 2,
     main = "ROC Curve Comparison")
plot(roc_rf, col = "red", lwd = 2, add = TRUE)
plot(roc_lgb, col = "green", lwd = 2, add = TRUE)

legend("bottomright",
       legend = c("Logistic", "Random Forest", "LightGBM"),
       col = c("blue", "red", "green"),
       lwd = 2)

