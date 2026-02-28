# 🌿 Module 02: Regularised Regression and Classification

**R + Statistical Learning Project | 620 Observations | Plant Growth ➝ Network Security ➝ Method Comparison**

---

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Tools and Technologies](#-tools-and-technologies)
- [Dataset 1: Plant Growth](#-dataset-1-plant-growth)
- [Dataset 2: Network Security](#-dataset-2-network-security)
- [Regularised Regression Framework](#-regularised-regression-framework)
- [Classification Methods](#-classification-methods)

- [Model Comparison](#-model-comparison)
- [Insights & Findings](#-insights--findings)
- [Business & Scientific Recommendations](#-business--scientific-recommendations)
- [Future Work](#-future-work)

---

## 🎯 Project Overview

This module combines **120 plant growth observations** with **15 environmental predictors** and **500 network security sessions** to demonstrate advanced supervised learning techniques. Students progress from regularised regression for handling multicollinearity to classification methods for binary outcome prediction, understanding how the same core principles apply across problem types.

### The Learning Journey
```
Plant Growth Data → Regularised Regression → Variable Selection → Biological Insights
       ↓                      ↓                       ↓                    ↓
  15 Predictors           Ridge/LASSO           Key Factors:        Light Duration
  Multicollinearity       Cross-Validation      Light, Nutrients     NPK Values

Network Security Data → Classification Methods → Performance Metrics → Security Insights
       ↓                         ↓                       ↓                    ↓
  500 Sessions             Logistic Reg           Precision/Recall     Failed Attempts
  Binary Outcome           Naive Bayes            Confusion Matrix     Packet Size
```

### Unified Concepts
| **Concept** | **Regression Context** | **Classification Context** |
|-------------|------------------------|----------------------------|
| Overfitting | Ridge/LASSO penalties | Cross-validation tuning |
| Cross-Validation | Select optimal λ | Compare model families |
| Feature Selection | LASSO zero coefficients | Logistic regression p-values |
| Performance Metrics | RMSE, R² | Accuracy, Precision, Recall, F1 |

---

## 🔧 Tools and Technologies

| **Category** | **Tools** | **Purpose** |
|--------------|-----------|-------------|
| **Regularised Regression** | glmnet | Ridge (α=0) and LASSO (α=1) implementation |
| **Classification** | stats (glm), naivebayes | Logistic regression and Naive Bayes |
| **Cross-Validation** | rsample | Creating validation folds |
| **Model Evaluation** | yardstick | Precision, recall, F1 metrics |
| **Visualization** | ggplot2, gridExtra | Coefficient paths, ROC curves |
| **Reporting** | knitr, kableExtra | Professional tables |
| **Data Manipulation** | tidyverse | Data preparation |

---

## 📊 Dataset 1: Plant Growth

The dataset contains **120 plant specimens** with **15 environmental variables** measuring factors that influence biomass production in controlled greenhouse experiments.

### Dataset Structure
```
Rows: 120  Columns: 15
```

| **Variable** | **Type** | **Description** | **Role** |
|--------------|----------|-----------------|----------|
| `biomass_g` | double | Plant biomass in grams | **Target** |
| `nitrogen_ppm` | double | Nitrogen concentration | Predictor |
| `phosphorus_ppm` | double | Phosphorus concentration | Predictor |
| `potassium_ppm` | double | Potassium concentration | Predictor |
| `ph_level` | double | Soil pH | Predictor |
| `light_intensity` | double | Light intensity (lux) | Predictor |
| `light_duration` | double | Daily light exposure (hours) | Predictor |
| `temperature_avg` | double | Average temperature (°C) | Predictor |
| `temperature_range` | double | Daily temperature variation | Predictor |
| `humidity_avg` | double | Average humidity (%) | Predictor |
| `humidity_range` | double | Daily humidity variation | Predictor |
| `water_ml` | double | Daily water amount (ml) | Predictor |
| `co2_ppm` | double | CO₂ concentration | Predictor |
| `air_circulation` | double | Air flow rate | Predictor |
| `growth_days` | double | Days until harvest | Predictor |

### Multicollinearity Analysis

```r
# Calculate correlation matrix
predictors <- plant_data %>% select(-biomass_g)
cor_matrix <- cor(predictors)

# Display correlation for key variables
cor_matrix[1:6, 1:6]
```

| | nitrogen_ppm | phosphorus_ppm | potassium_ppm | ph_level | light_intensity | light_duration |
|---|--------------|----------------|---------------|----------|-----------------|----------------|
| nitrogen_ppm | 1.000 | **0.760** | **0.740** | -0.301 | -0.065 | 0.079 |
| phosphorus_ppm | **0.760** | 1.000 | **0.794** | -0.377 | -0.014 | 0.040 |
| potassium_ppm | **0.740** | **0.794** | 1.000 | -0.260 | -0.095 | -0.103 |
| ph_level | -0.301 | -0.377 | -0.260 | 1.000 | -0.070 | -0.108 |
| light_intensity | -0.065 | -0.014 | -0.095 | -0.070 | 1.000 | 0.603 |
| light_duration | 0.079 | 0.040 | -0.103 | -0.108 | 0.603 | 1.000 |

**Screenshot Suggestion 1**: *Add correlation heatmap showing multicollinearity*
```
📸 [Screenshot: correlation_heatmap.png]
Caption: Figure 1: Correlation matrix heatmap revealing strong multicollinearity among nutrient variables (|r| > 0.7)
```

### High Correlations Detected

```r
# Identify correlations > |0.7|
high_corr_pairs <- which(abs(cor_matrix) > 0.7 & abs(cor_matrix) < 1, arr.ind = TRUE)

for(i in 1:nrow(high_corr_pairs)) {
  var1 <- rownames(cor_matrix)[high_corr_pairs[i, 1]]
  var2 <- colnames(cor_matrix)[high_corr_pairs[i, 2]]
  corr_val <- cor_matrix[high_corr_pairs[i, 1], high_corr_pairs[i, 2]]
  cat(sprintf("%s - %s: %.3f\n", var1, var2, corr_val))
}
```

| Variable Pair | Correlation |
|---------------|-------------|
| phosphorus_ppm - nitrogen_ppm | 0.760 |
| potassium_ppm - nitrogen_ppm | 0.740 |
| potassium_ppm - phosphorus_ppm | 0.794 |
| humidity_range - humidity_avg | 0.747 |

**Key Finding**: Strong correlations among nutrients and humidity variables confirm multicollinearity—the perfect scenario for regularised regression.

---

## 📈 Dataset 2: Network Security

The dataset contains **500 network sessions** with **6 features** for classifying normal traffic versus security threats.

### Dataset Structure
```
Rows: 500  Columns: 7
```

| **Variable** | **Type** | **Description** | **Role** |
|--------------|----------|-----------------|----------|
| `session_id` | integer | Unique session identifier | ID |
| `packet_size` | double | Average packet size (bytes) | Predictor |
| `duration` | double | Session duration (seconds) | Predictor |
| `failed_attempts` | integer | Number of failed login attempts | Predictor |
| `port_activity` | integer | Number of ports accessed | Predictor |
| `bandwidth_usage` | double | Bandwidth consumed (MB) | Predictor |
| `class` | factor | Normal / Threat | **Target** |

### Class Distribution

```r
table(network_data$class)
```

| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 373 | 74.6% |
| Threat | 127 | 25.4% |

**Screenshot Suggestion 2**: *Add class imbalance bar chart*
```
📸 [Screenshot: class_distribution.png]
Caption: Figure 2: Class distribution showing imbalance (74.6% Normal, 25.4% Threat)
```

### Summary Statistics by Class

```r
network_summary <- network_data %>%
  group_by(class) %>%
  summarise(
    n = n(),
    avg_packet_size = mean(packet_size),
    avg_duration = mean(duration),
    avg_failed_attempts = mean(failed_attempts),
    avg_port_activity = mean(port_activity),
    avg_bandwidth_usage = mean(bandwidth_usage)
  )
```

| Statistic | Normal | Threat | Difference |
|-----------|--------|--------|------------|
| n | 373 | 127 | - |
| avg_packet_size | 632.75 | 545.66 | ↓ 87.09 |
| avg_duration | 45.17 | 41.34 | ↓ 3.83 |
| avg_failed_attempts | 1.65 | 2.98 | ↑ 1.33 |
| avg_port_activity | 4.99 | 5.63 | ↑ 0.64 |
| avg_bandwidth_usage | 53.96 | 58.79 | ↑ 4.83 |

**Screenshot Suggestion 3**: *Add boxplots comparing feature distributions by class*
```
📸 [Screenshot: class_feature_boxplots.png]
Caption: Figure 3: Feature distributions comparing Normal vs Threat sessions
```

---

## 🔧 Regularised Regression Framework

### 1. Data Preparation for glmnet

```r
library(glmnet)
library(rsample)

# Create 80-20 train-test split
set.seed(2024)
data_split <- initial_split(plant_data, prop = 0.8, strata = biomass_g)
train_data <- training(data_split)
test_data <- testing(data_split)

cat(sprintf("Training set: %d observations\n", nrow(train_data)))
cat(sprintf("Test set: %d observations\n", nrow(test_data)))
```
```
Training set: 96 observations
Test set: 24 observations
```

```r
# Prepare matrices for glmnet (requires matrix format, no intercept column)
x_train <- model.matrix(biomass_g ~ . - 1, data = train_data)
x_test <- model.matrix(biomass_g ~ . - 1, data = test_data)
y_train <- train_data$biomass_g
y_test <- test_data$biomass_g

cat(sprintf("Training predictors: %d x %d\n", nrow(x_train), ncol(x_train)))
```
```
Training predictors: 96 x 14
```

### 2. Ridge Regression (L2 Penalty, α = 0)

```r
# Fit ridge regression
ridge_model <- glmnet(x_train, y_train, alpha = 0)

# Cross-validation to select optimal lambda
set.seed(2024)
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)

# Plot cross-validation results
plot(ridge_cv)
```

**Screenshot Suggestion 4**: *Add ridge cross-validation plot*
```
📸 [Screenshot: ridge_cv_plot.png]
Caption: Figure 4: Ridge regression cross-validation curve showing MSE vs log(λ)
```

```r
# Extract coefficients at optimal lambda (lambda.1se)
ridge_coef <- coef(ridge_cv, s = "lambda.1se")
ridge_coef_df <- data.frame(
  Variable = rownames(ridge_coef)[-1],
  Coefficient = as.numeric(ridge_coef[-1])
) %>% arrange(desc(abs(Coefficient)))
```

| Variable | Ridge Coefficient |
|----------|-------------------|
| light_duration | 2.549 |
| ph_level | 1.300 |
| phosphorus_ppm | 1.076 |
| temperature_avg | 0.911 |
| nitrogen_ppm | 0.600 |
| temperature_range | -0.447 |
| potassium_ppm | 0.307 |
| humidity_range | -0.293 |

```r
# Make predictions and evaluate
ridge_pred <- predict(ridge_cv, newx = x_test, s = "lambda.1se")
ridge_rmse <- sqrt(mean((y_test - ridge_pred)^2))
ridge_r2 <- cor(y_test, ridge_pred)^2

cat(sprintf("Ridge Test RMSE: %.3f\n", ridge_rmse))
cat(sprintf("Ridge Test R²: %.3f\n", ridge_r2))
```
```
Ridge Test RMSE: 11.510
Ridge Test R²: 0.905
```

### 3. LASSO Regression (L1 Penalty, α = 1)

```r
# Fit LASSO regression
lasso_model <- glmnet(x_train, y_train, alpha = 1)

# Plot regularisation path
plot(lasso_model, xvar = "lambda")
```

**Screenshot Suggestion 5**: *Add LASSO coefficient path plot*
```
📸 [Screenshot: lasso_path_plot.png]
Caption: Figure 5: LASSO coefficient paths showing variable selection as λ increases
```

```r
# Cross-validation for LASSO
set.seed(2024)
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 10)

cat(sprintf("Lambda min: %.4f (CV error: %.3f)\n", lasso_cv$lambda.min, min(lasso_cv$cvm)))
cat(sprintf("Lambda 1se: %.4f (CV error: %.3f)\n", lasso_cv$lambda.1se, 
            lasso_cv$cvm[lasso_cv$lambda == lasso_cv$lambda.1se]))
```
```
Lambda min: 0.0736 (CV error: 129.541)
Lambda 1se: 1.4440 (CV error: 149.613)
```

```r
# Extract coefficients at lambda.1se
lasso_coef <- coef(lasso_cv, s = "lambda.1se")
lasso_coef_df <- data.frame(
  Variable = rownames(lasso_coef)[-1],
  Coefficient = as.numeric(lasso_coef[-1])
) %>%
  mutate(Selected = abs(Coefficient) > 0) %>%
  arrange(desc(abs(Coefficient)))
```

| Variable | LASSO Coefficient | Selected |
|----------|-------------------|----------|
| light_duration | 2.618 | ✓ TRUE |
| nitrogen_ppm | 0.703 | ✓ TRUE |
| temperature_avg | 0.689 | ✓ TRUE |
| phosphorus_ppm | 0.603 | ✓ TRUE |
| potassium_ppm | 0.376 | ✓ TRUE |
| growth_days | 0.074 | ✓ TRUE |
| light_intensity | 0.031 | ✓ TRUE |
| air_circulation | 0.013 | ✓ TRUE |
| ph_level | 0.000 | ✗ FALSE |
| temperature_range | 0.000 | ✗ FALSE |
| humidity_avg | 0.000 | ✗ FALSE |
| humidity_range | 0.000 | ✗ FALSE |
| water_ml | 0.000 | ✗ FALSE |
| co2_ppm | 0.000 | ✗ FALSE |

**Key Finding**: LASSO selected **8 of 14 predictors**—automatic variable selection in action!

```r
# Evaluate LASSO
lasso_pred <- predict(lasso_cv, newx = x_test, s = "lambda.1se")
lasso_rmse <- sqrt(mean((y_test - lasso_pred)^2))
lasso_r2 <- cor(y_test, lasso_pred)^2

cat(sprintf("LASSO Test RMSE: %.3f\n", lasso_rmse))
cat(sprintf("LASSO Test R²: %.3f\n", lasso_r2))
```
```
LASSO Test RMSE: 7.889
LASSO Test R²: 0.953
```

### 4. Ordinary Least Squares Comparison

```r
# Fit OLS for baseline comparison
ols_model <- lm(biomass_g ~ ., data = train_data)
ols_pred <- predict(ols_model, newdata = test_data)
ols_rmse <- sqrt(mean((y_test - ols_pred)^2))
ols_r2 <- cor(y_test, ols_pred)^2

cat(sprintf("OLS Test RMSE: %.3f\n", ols_rmse))
cat(sprintf("OLS Test R²: %.3f\n", ols_r2))
```
```
OLS Test RMSE: 9.163
OLS Test R²: 0.935
```

### 5. Regularised Regression Summary

```r
performance_table <- data.frame(
  Method = c("OLS", "Ridge", "LASSO"),
  RMSE = c(ols_rmse, ridge_rmse, lasso_rmse),
  R_squared = c(ols_r2, ridge_r2, lasso_r2),
  Variables_Selected = c(14, 14, 8)
)
```

| Method | RMSE | R² | Variables Selected |
|--------|------|-----|-------------------|
| OLS | 9.163 | 0.935 | 14 |
| Ridge | 11.510 | 0.905 | 14 |
| **LASSO** | **7.889** | **0.953** | **8** |

**Screenshot Suggestion 6**: *Add performance comparison bar chart*
```
📸 [Screenshot: regularised_performance.png]
Caption: Figure 6: RMSE comparison across OLS, Ridge, and LASSO regression
```

---

## 🔄 Classification Methods

### 1. Data Preparation

```r
# Ensure class is factor with correct levels
network_data <- read.csv("network_data.csv") %>%
  mutate(class = factor(class, levels = c("Normal", "Threat")))

# Create stratified 10-fold cross-validation folds
set.seed(2024)
cv_folds <- vfold_cv(network_data, v = 10, strata = class)

# Display fold structure
cv_folds
```

```
# A 10-fold cross-validation using stratification
# A tibble: 10 × 2
   splits             id    
   <list>             <chr> 
 1 <split [449/51]>   Fold01
 2 <split [449/51]>   Fold02
 3 <split [449/51]>   Fold03
 4 <split [450/50]>   Fold04
 5 <split [450/50]>   Fold05
 6 <split [450/50]>   Fold06
 7 <split [450/50]>   Fold07
 8 <split [451/49]>   Fold08
 9 <split [451/49]>   Fold09
10 <split [451/49]>   Fold10
```

**Screenshot Suggestion 7**: *Add diagram explaining stratified cross-validation*
```
📸 [Screenshot: stratified_cv_diagram.png]
Caption: Figure 7: Stratified 10-fold cross-validation preserving class proportions
```

### 2. Logistic Regression Implementation

```r
# Initialize results storage
logistic_results <- data.frame(
  fold = integer(),
  accuracy = numeric(),
  precision = numeric(),
  recall = numeric(),
  f1_score = numeric()
)

# Cross-validation loop
for(i in 1:nrow(cv_folds)) {
  # Extract training and testing data
  train_data <- analysis(cv_folds$splits[[i]])
  test_data <- assessment(cv_folds$splits[[i]])
  
  # Fit logistic regression
  model <- glm(
    class ~ packet_size + duration + failed_attempts + 
      port_activity + bandwidth_usage,
    data = train_data,
    family = binomial
  )
  
  # Make predictions (probability threshold 0.5)
  pred_probs <- predict(model, newdata = test_data, type = "response")
  pred_class <- factor(
    ifelse(pred_probs > 0.5, "Threat", "Normal"),
    levels = c("Normal", "Threat")
  )
  
  # Calculate metrics using yardstick
  fold_results <- test_data %>%
    mutate(pred_class = pred_class)
  
  accuracy_val <- accuracy(fold_results, truth = class, estimate = pred_class) %>% pull(.estimate)
  precision_val <- precision(fold_results, truth = class, estimate = pred_class, event_level = "second") %>% pull(.estimate)
  recall_val <- recall(fold_results, truth = class, estimate = pred_class, event_level = "second") %>% pull(.estimate)
  f1_val <- f_meas(fold_results, truth = class, estimate = pred_class, event_level = "second") %>% pull(.estimate)
  
  # Store results
  logistic_results[i, ] <- c(i, accuracy_val, precision_val, recall_val, f1_val)
}
```

### Logistic Regression Cross-Validation Results

| fold | accuracy | precision | recall | f1_score |
|------|----------|-----------|--------|----------|
| 1 | 0.843 | 0.778 | 0.538 | 0.636 |
| 2 | 0.824 | 0.750 | 0.462 | 0.571 |
| 3 | 0.824 | 0.833 | 0.385 | 0.526 |
| 4 | 0.840 | 0.857 | 0.462 | 0.600 |
| 5 | 0.840 | 0.727 | 0.615 | 0.667 |
| 6 | 0.840 | 1.000 | 0.385 | 0.556 |
| 7 | 0.780 | 0.625 | 0.385 | 0.476 |
| 8 | 0.796 | 0.667 | 0.333 | 0.444 |
| 9 | 0.714 | 0.400 | 0.333 | 0.364 |
| 10 | 0.776 | 0.571 | 0.333 | 0.421 |

```r
# Calculate mean performance
logistic_summary <- logistic_results %>%
  summarise(
    mean_accuracy = mean(accuracy),
    mean_precision = mean(precision),
    mean_recall = mean(recall),
    mean_f1 = mean(f1_score)
  )
```

| Metric | Mean Value |
|--------|------------|
| Accuracy | 0.808 |
| Precision | 0.721 |
| Recall | 0.423 |
| F1-Score | **0.526** |

**Screenshot Suggestion 8**: *Add boxplots of logistic regression metrics across folds*
```
📸 [Screenshot: logistic_cv_boxplots.png]
Caption: Figure 8: Distribution of logistic regression performance metrics across 10 CV folds
```

### 3. Coefficient Interpretation

```r
# Fit final logistic model on complete dataset
final_logistic <- glm(
  class ~ packet_size + duration + failed_attempts + 
    port_activity + bandwidth_usage,
  data = network_data,
  family = binomial
)

# Extract odds ratios and confidence intervals
coef_summary <- tidy(final_logistic, conf.int = TRUE, exponentiate = TRUE)
```

| term | estimate (odds ratio) | std.error | statistic | p.value | conf.low | conf.high |
|------|----------------------|-----------|-----------|---------|----------|-----------|
| (Intercept) | 0.094 | 0.681 | -3.468 | 0.0005 | 0.024 | 0.350 |
| packet_size | **0.997** | 0.001 | -4.552 | **<0.001** | 0.996 | 0.998 |
| duration | 0.989 | 0.006 | -1.873 | 0.061 | 0.977 | 1.000 |
| failed_attempts | **2.267** | 0.097 | 8.449 | **<0.001** | 1.888 | 2.762 |
| port_activity | **1.176** | 0.053 | 3.086 | **0.002** | 1.062 | 1.305 |
| bandwidth_usage | **1.015** | 0.006 | 2.372 | **0.018** | 1.003 | 1.027 |

### Interpretation of Key Predictors

| Variable | Odds Ratio | Interpretation | Significance |
|----------|------------|----------------|--------------|
| **failed_attempts** | 2.267 | Each additional failed attempt **more than doubles** threat odds | p < 0.001 |
| **packet_size** | 0.997 | Each byte increase reduces threat odds by 0.3% | p < 0.001 |
| **port_activity** | 1.176 | Each additional port increases threat odds by 17.6% | p = 0.002 |
| **bandwidth_usage** | 1.015 | Each MB increase raises threat odds by 1.5% | p = 0.018 |
| **duration** | 0.989 | Not statistically significant | p = 0.061 |

**Screenshot Suggestion 9**: *Add forest plot of odds ratios with confidence intervals*
```
📸 [Screenshot: odds_ratio_forest.png]
Caption: Figure 9: Forest plot showing odds ratios and 95% confidence intervals for threat predictors
```

### 4. Gaussian Naive Bayes Implementation

```r
library(naivebayes)

# Initialize results storage
nb_results <- data.frame(
  fold = integer(),
  accuracy = numeric(),
  precision = numeric(),
  recall = numeric(),
  f1_score = numeric()
)

# Cross-validation loop
for(i in 1:nrow(cv_folds)) {
  train_data <- analysis(cv_folds$splits[[i]])
  test_data <- assessment(cv_folds$splits[[i]])
  
  # Fit Naive Bayes (Gaussian assumption)
  model <- naive_bayes(
    class ~ packet_size + duration + failed_attempts + 
      port_activity + bandwidth_usage,
    data = train_data,
    usekernel = FALSE,  # Gaussian distribution
    usepoisson = FALSE
  )
  
  # Make predictions
  pred_class <- predict(model, newdata = test_data)
  
  # Calculate metrics
  fold_results <- test_data %>%
    mutate(pred_class = pred_class)
  
  accuracy_val <- accuracy(fold_results, truth = class, estimate = pred_class) %>% pull(.estimate)
  precision_val <- precision(fold_results, truth = class, estimate = pred_class, event_level = "second") %>% pull(.estimate)
  recall_val <- recall(fold_results, truth = class, estimate = pred_class, event_level = "second") %>% pull(.estimate)
  f1_val <- f_meas(fold_results, truth = class, estimate = pred_class, event_level = "second") %>% pull(.estimate)
  
  nb_results[i, ] <- c(i, accuracy_val, precision_val, recall_val, f1_val)
}
```

### Naive Bayes Cross-Validation Results

| fold | accuracy | precision | recall | f1_score |
|------|----------|-----------|--------|----------|
| 1 | 0.824 | 0.750 | 0.462 | 0.571 |
| 2 | 0.824 | 0.833 | 0.385 | 0.526 |
| 3 | 0.824 | 0.833 | 0.385 | 0.525 |
| 4 | 0.840 | 0.857 | 0.462 | 0.600 |
| 5 | 0.820 | 0.700 | 0.538 | 0.609 |
| 6 | 0.820 | 1.000 | 0.308 | 0.471 |
| 7 | 0.780 | 0.667 | 0.308 | 0.421 |
| 8 | 0.755 | 0.500 | 0.250 | 0.333 |
| 9 | 0.673 | 0.300 | 0.250 | 0.273 |
| 10 | 0.796 | 0.667 | 0.333 | 0.444 |

```r
# Calculate mean performance
nb_summary <- nb_results %>%
  summarise(
    mean_accuracy = mean(accuracy),
    mean_precision = mean(precision),
    mean_recall = mean(recall),
    mean_f1 = mean(f1_score)
  )
```

| Metric | Mean Value |
|--------|------------|
| Accuracy | 0.796 |
| Precision | 0.711 |
| Recall | 0.368 |
| F1-Score | **0.477** |

---

## 📊 Model Comparison

### Method Comparison Table

```r
comparison <- data.frame(
  Method = c("Logistic Regression", "Gaussian Naive Bayes"),
  Accuracy = c(logistic_summary$mean_accuracy, nb_summary$mean_accuracy),
  Precision = c(logistic_summary$mean_precision, nb_summary$mean_precision),
  Recall = c(logistic_summary$mean_recall, nb_summary$mean_recall),
  F1_Score = c(logistic_summary$mean_f1, nb_summary$mean_f1)
)
```

| Method | Accuracy | Precision | Recall | **F1-Score** |
|--------|----------|-----------|--------|--------------|
| **Logistic Regression** | **0.808** | **0.721** | **0.423** | **0.526** |
| Gaussian Naive Bayes | 0.796 | 0.711 | 0.368 | 0.477 |

**Screenshot Suggestion 10**: *Add side-by-side bar chart comparing all metrics*
```
📸 [Screenshot: method_comparison.png]
Caption: Figure 10: Performance comparison between Logistic Regression and Naive Bayes
```

### Final Model Selection

```r
best_method <- ifelse(logistic_summary$mean_f1 > nb_summary$mean_f1, 
                      "Logistic Regression", "Gaussian Naive Bayes")
cat("Best performing method based on F1-Score:", best_method)
```
```
Best performing method based on F1-Score: Logistic Regression
```

### Final Confusion Matrix

```r
# Fit best method on full dataset
final_model <- glm(
  class ~ packet_size + duration + failed_attempts + 
    port_activity + bandwidth_usage,
  data = network_data,
  family = binomial
)

final_pred_probs <- predict(final_model, newdata = network_data, type = "response")
final_pred <- factor(
  ifelse(final_pred_probs > 0.5, "Threat", "Normal"),
  levels = c("Normal", "Threat")
)

# Create confusion matrix
conf_matrix <- table(Truth = network_data$class, Prediction = final_pred)
```

```
          Prediction
Truth      Normal Threat
  Normal      350     23
  Threat       73     54
```

| | Predicted Normal | Predicted Threat |
|---|------------------|------------------|
| **Actual Normal** | 350 (True Negatives) | 23 (False Positives) |
| **Actual Threat** | 73 (False Negatives) | **54 (True Positives)** |

### Performance Metrics Interpretation

```r
# Calculate final metrics
final_metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score"),
  Value = c(
    (350 + 54) / 500,                    # Accuracy
    54 / (54 + 23),                       # Precision
    54 / (54 + 73),                       # Recall
    2 * (54 / (54 + 23)) * (54 / (54 + 73)) / 
      ((54 / (54 + 23)) + (54 / (54 + 73)))  # F1
  )
)
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 0.808 | 80.8% of all sessions correctly classified |
| Precision | 0.701 | When model predicts "Threat", it's correct 70.1% of time |
| Recall | 0.425 | Model catches only 42.5% of actual threats |
| F1-Score | 0.529 | Harmonic mean of precision and recall |

**Screenshot Suggestion 11**: *Add confusion matrix heatmap*
```
📸 [Screenshot: confusion_matrix_heatmap.png]
Caption: Figure 11: Confusion matrix heatmap for final logistic regression model
```

---

## 💡 Insights & Findings

### Plant Growth Insights

1. **LASSO outperforms both OLS and Ridge**
   - Test RMSE: **7.889** (vs Ridge 11.51, OLS 9.163)
   - Test R²: **0.953** (explains 95.3% of variance)
   - Selected **8 of 14 predictors**—automatic feature selection

2. **Top 5 Factors Influencing Biomass**

| Rank | Variable | LASSO Coef | Ridge Coef | Biological Role |
|------|----------|------------|------------|-----------------|
| 1 | light_duration | 2.618 | 2.549 | Photosynthesis duration |
| 2 | nitrogen_ppm | 0.703 | 0.600 | Protein synthesis |
| 3 | temperature_avg | 0.689 | 0.911 | Metabolic rate |
| 4 | phosphorus_ppm | 0.603 | 1.076 | Energy transfer (ATP) |
| 5 | potassium_ppm | 0.376 | 0.307 | Enzyme activation |

3. **Variables Eliminated by LASSO**
   - pH level (correlated with nutrients)
   - Temperature range (less important than average)
   - Humidity variables (avg and range)
   - Water amount (possibly optimized already)
   - CO₂ concentration (consistent across experiment)

**Screenshot Suggestion 12**: *Add coefficient comparison plot for LASSO vs Ridge*
```
📸 [Screenshot: coefficient_comparison.png]
Caption: Figure 12: Coefficient comparison between Ridge (all retained) and LASSO (sparse selection)
```

### Network Security Insights

1. **Failed Attempts: The Strongest Threat Indicator**
   - Odds Ratio: **2.267** (p < 0.001)
   - Each additional failed attempt **more than doubles** threat probability
   - Aligns with cybersecurity: brute force attacks generate multiple failures

2. **Packet Size: Smaller Packets Signal Threats**
   - Odds Ratio: 0.997 (p < 0.001)
   - Each byte decrease increases threat odds
   - Explanation: Reconnaissance probes, port scans use small packets

3. **Port Activity: Scanning Behavior**
   - Odds Ratio: 1.176 (p = 0.002)
   - Each additional port accessed increases threat odds by 17.6%
   - Attackers probe multiple ports for vulnerabilities

4. **Critical Limitation: Low Recall**
   - Recall: **42.5%**—model misses **57.5% of actual threats**
   - Of 127 threats, only 54 detected, 73 missed
   - False Negative Rate: 57.5%—unacceptable for security

5. **False Positive Rate**
   - Precision: 70.1%—when alarm sounds, 70% chance it's real
   - False Positive Rate: 6.2% (23/373 normal sessions)
   - Low false alarms but at cost of missing real threats

---

## 🧭 Business & Scientific Recommendations

### For Agricultural Researchers (Plant Growth)

#### 1. ✅ Optimize Light Duration
**Finding**: Light duration has largest coefficient (2.618)
**Recommendation**: Extend daily light exposure to **12-15 hours**
- Use LED grow lights for energy efficiency
- Consider automated light timers
- Monitor for light saturation effects

#### 2. 🧪 Nutrient Management Protocol

| Nutrient | Target Range | Current Avg | Action |
|----------|--------------|-------------|--------|
| Nitrogen | 50-100 ppm | 61.4 ppm | Maintain/increase slightly |
| Phosphorus | 15-30 ppm | 25.2 ppm | Maintain current levels |
| Potassium | 100-200 ppm | 138.3 ppm | Monitor, may increase |

**Recommendation**: Implement weekly soil testing focusing on NPK levels

#### 3. 🌡️ Temperature Control
**Finding**: Average temperature coefficient = 0.689
**Recommendation**: Maintain **20-25°C** with minimal daily fluctuation
- Install automated ventilation
- Monitor with data loggers
- Avoid temperature range >5°C daily

#### 4. 📊 Experimental Design Improvements
- Collect data on additional varieties
- Include mycorrhizal treatments
- Test light spectrum variations
- Add soil microbiome measurements

**Screenshot Suggestion 13**: *Add optimal growth conditions infographic*
```
📸 [Screenshot: optimal_conditions.png]
Caption: Figure 13: Recommended optimal growing conditions based on LASSO model
```

### For Security Teams (Network Monitoring)

#### 1. 🚨 Immediate Action: Adjust Threshold
**Problem**: Current model misses 57.5% of threats
**Solution**: Lower classification threshold from 0.5 to **0.3**
```r
# Expected impact with threshold = 0.3
# More threats caught, more false alarms
```

| Threshold | Recall | Precision | Threats Caught | False Alarms |
|-----------|--------|-----------|----------------|--------------|
| 0.5 | 42.5% | 70.1% | 54/127 | 23 |
| 0.3 | ~65% | ~50% | ~83/127 | ~83 |
| 0.2 | ~80% | ~35% | ~102/127 | ~198 |

**Recommendation**: Implement tiered alerting
- Threshold 0.2: Log for investigation
- Threshold 0.5: Immediate alert
- Threshold 0.8: Automatic blocking

#### 2. 🎯 Focus on Failed Attempts
**Finding**: Most powerful predictor (OR = 2.27)
**Actions**:
- Set alert on >3 failed attempts in 5 minutes
- Implement account lockout after 5 failures
- Add CAPTCHA after 2 failures
- Log all failed attempts with timestamps

#### 3. 📦 Packet Size Monitoring
**Finding**: Smaller packets indicate threats
**Actions**:
- Create baseline of normal packet sizes by application
- Alert on packets <100 bytes from external sources
- Monitor for packet fragmentation patterns
- Track average packet size over rolling windows

#### 4. 🔌 Port Activity Rules
**Finding**: Each additional port increases threat odds
**Actions**:
- Baseline: Normal apps use 1-3 ports
- Alert on >5 unique ports in 60 seconds
- Block known vulnerable ports at firewall
- Implement port knocking for sensitive services

#### 5. 📊 Daily Operational Recommendations

**Morning Review** (Security Analyst):
```
1. Check failed attempts dashboard
2. Review top 10 sessions by threat probability
3. Investigate false negatives from previous day
4. Update rules based on new threat intel
```

**Automated Responses**:
```r
if(failed_attempts > 5 & port_activity > 10) {
  block_ip(address)
  send_alert("Possible port scan + brute force")
} else if(packet_size < 100 & bandwidth_usage > 100) {
  log_for_review(address)
  increase_monitoring(address)
}
```

**Screenshot Suggestion 14**: *Add ROC curve with threshold recommendations*
```
📸 [Screenshot: roc_curve.png]
Caption: Figure 14: ROC curve showing trade-off between true positive and false positive rates
```

---

## 🔮 Future Work

### For Plant Growth Modeling

#### 1. 🧠 Deep Learning Approaches
- Implement CNN for image-based growth assessment
- Use LSTM for time-series growth prediction
- Compare with regularised regression results

#### 2. 🔬 Additional Data Collection
```
Proposed New Variables:
├── Soil microbiome composition
├── Mycorrhizal colonization %
├── Light spectrum (RGB ratios)
├── Vapor pressure deficit
├── Electrical conductivity
└── Root zone temperature
```

#### 3. 📱 Precision Agriculture App
- Mobile app for farmers
- Input current conditions
- Get biomass predictions
- Receive optimization recommendations

### For Network Security

#### 1. 🤖 Ensemble Methods
```r
# Proposed ensemble approach
model_1 <- glmnet(alpha = 1)  # LASSO
model_2 <- randomForest()      # Random Forest
model_3 <- xgboost()           # XGBoost

# Weighted voting based on CV performance
final_pred <- 0.4 * pred_lasso + 
              0.3 * pred_rf + 
              0.3 * pred_xgb
```

#### 2. 📈 Real-Time Streaming Implementation
- Apache Kafka for event streaming
- Flink for real-time processing
- Elasticsearch for storage
- Kibana for dashboards

#### 3. 🎯 Feature Engineering Pipeline
```r
# Additional features to engineer
network_data %>%
  mutate(
    attempts_per_minute = failed_attempts / (duration / 60),
    packets_per_port = bandwidth_usage / port_activity,
    threat_composite = (failed_attempts * 2.267) + 
                      ((100 - packet_size) * 0.003) +
                      (port_activity * 0.176),
    time_of_day = hour(as.POSIXct(timestamp)),
    is_business_hours = time_of_day %in% 8:17
  )
```

#### 4. 🔄 Automated Retraining Pipeline
```r
# Weekly model retraining
auto_retrain <- function(new_data) {
  # Combine with historical data
  full_data <- bind_rows(historical_data, new_data)
  
  # Run cross-validation
  cv_results <- cross_validate(full_data)
  
  # Select best model
  best_model <- select_best(cv_results)
  
  # Update production model
  saveRDS(best_model, "production_model.rds")
  
  # Log performance metrics
  log_metrics(cv_results)
}
```

---

## 📥 Repository Access

```bash
git clone https://github.com/yourusername/STAT312-Practicals.git
cd STAT312-Practicals/02-Regularised-Regression-and-Classification
```

### Files Included:
- `practical_3_4_combined.Rmd` - Complete R Markdown document
- `practical_3_4_combined.pdf` - Knitted output with all results
- `data/plant_growth_data.csv` - Plant growth dataset (120 obs, 15 vars)
- `data/network_data.csv` - Network security dataset (500 obs, 7 vars)

### Quick Start
```r
# In RStudio
setwd("path/to/02-Regularised-Regression-and-Classification")
rmarkdown::render("practical_3_4_combined.Rmd")
```

---

## 🙏 Acknowledgements

- **Statistical Learning Community**: For glmnet package development
- **Cybersecurity Researchers**: For realistic threat simulation guidance
- **Agricultural Scientists**: For domain expertise in plant growth factors
- **STAT312 Students**: 2024-2025 cohorts for feedback on method comparisons

---

## 📊 Citation

If using this module in your teaching or research:

```
[Your Name] (2025). Module 02: Regularised Regression and Classification.
STAT312: Advanced Data Analysis in R. Nelson Mandela University.
https://github.com/yourusername/STAT312-Practicals
```

---

## 📋 Summary Table: When to Use Each Method

| **Method** | **Best For** | **Pros** | **Cons** | **Our Result** |
|------------|--------------|----------|----------|----------------|
| **Ridge Regression** | Many correlated predictors | Retains all variables, handles multicollinearity | No variable selection | RMSE: 11.51 |
| **LASSO Regression** | Feature selection needed | Automatic selection, sparse models | Can be unstable with high correlations | **RMSE: 7.89** |
| **Logistic Regression** | Binary classification, interpretability | Clear odds ratios, p-values | Assumes linear log-odds | **F1: 0.526** |
| **Naive Bayes** | Fast, simple baseline | Fast training, handles missing data | Independence assumption often violated | F1: 0.477 |

---

**⬆️ [Back to Top](#-module-02-regularised-regression-and-classification)**
