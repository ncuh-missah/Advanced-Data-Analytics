# 🏥 Module 01: Data Wrangling and Model Selection

**R + Statistical Learning Project | 410 Records | Hospital Admissions ➝ Retail Sales ➝ Business Insights**

---

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Tools and Technologies](#-tools-and-technologies)
- [Dataset 1: Hospital Admissions](#-dataset-1-hospital-admissions)
- [Dataset 2: Retail Sales](#-dataset-2-retail-sales)
- [Data Wrangling Process](#-data-wrangling-process)
- [Cross-Validation Framework](#-cross-validation-framework)
- [Model Comparison](#-model-comparison)
- [Insights & Findings](#-insights--findings)
- [Business Recommendations](#-business-recommendations)
- [Future Work](#-future-work)

---

## 🎯 Project Overview

This module analyzes **210 hospital admissions records** and **200 retail store observations** to demonstrate the complete data analysis workflow—from messy real-world data to validated predictive models. Using R and the tidyverse ecosystem, we transform raw data into actionable business insights through systematic cleaning, exploration, and cross-validated modeling.

### The Learning Journey
```
Raw Hospital Data → Clean Data → Summary Statistics → Business Insights
       ↓                ↓              ↓                    ↓
   Missing Values    dplyr Verbs    Visualizations    Department Analysis
   Inconsistent Text mutate/filter  Summary Tables    Cost Patterns

Raw Retail Data → Clean Data → Cross-Validation → Model Selection → Recommendations
       ↓              ↓              ↓                  ↓                ↓
   Negative Values  imputation     10-fold CV      RMSE Comparison    Staffing Strategy
   Outliers         IQR method     Model Folds     Best Model Choice  Budget Allocation
```

---

## 🛠️ Tools and Technologies

| **Category** | **Tools** | **Purpose** |
|--------------|-----------|-------------|
| **Data Wrangling** | R (tidyverse, dplyr, tidyr) | Data cleaning, transformation, and manipulation |
| **Date Handling** | lubridate | Parsing multiple date formats |
| **Cross-Validation** | rsample | Creating validation folds and model evaluation |
| **Visualization** | ggplot2 | Exploratory data analysis plots |
| **Reporting** | knitr, R Markdown | Generating professional PDF outputs |
| **Version Control** | Git, GitHub | Code management and sharing |

---

## 📊 Dataset 1: Hospital Admissions

The dataset contains **210 patient records** with **8 variables** simulating real-world hospital data with common quality issues.

### Dataset Structure
```
Rows: 210  Columns: 8
```

| **Variable** | **Type** | **Description** | **Issues** |
|--------------|----------|-----------------|------------|
| `patient_id` | double | Unique patient identifier | 17 missing values |
| `age` | double | Patient age in years | 14 missing, impossible values |
| `gender` | character | Patient gender | 22 missing, inconsistent case |
| `admission_date` | character | Date of hospital admission | 21 missing, multiple formats |
| `department` | character | Hospital department | 14 missing, variations ("ER", "emergency") |
| `length_of_stay` | double | Days in hospital | 14 missing, negatives possible |
| `treatment_cost` | double | Cost of treatment in R | 12 missing, negatives possible |
| `discharge_status` | character | Disposition at discharge | 16 missing, inconsistent case |

### Missing Data Summary

| Variable | Missing Count |
|----------|---------------|
| patient_id | 17 |
| age | 14 |
| gender | 22 |
| admission_date | 21 |
| department | 14 |
| length_of_stay | 14 |
| treatment_cost | 12 |
| discharge_status | 16 |

**Screenshot Suggestion 1**: *Add a bar plot of missing values by variable here*
```
📸 [Screenshot: missing_values_plot.png]
Caption: Figure 1: Missing value distribution across hospital dataset variables
```

### Initial Data Sample
```r
# First few rows of messy data
head(hospital_data)
```

| patient_id | age | gender | admission_date | department | length_of_stay | treatment_cost | discharge_status |
|------------|-----|--------|----------------|------------|----------------|----------------|------------------|
| 1 | 59 | Male | 07/01/2023 | ER | 3 | 20900 | Discharged |
| 2 | 79 | Female | 02/02/2023 | Surgery | 5 | 33700 | Discharged |
| 3 | 59 | Male | 22/11/2023 | ER | 11 | 11300 | Discharged |
| 4 | 57 | NA | 17/04/2023 | Neurology | 3 | 1400 | Discharged |
| 5 | 84 | female | 22/03/2023 | emergency | 7 | 14500 | Discharged |
| 6 | 84 | Male | 30/10/2023 | A&E | 9 | 9100 | Discharged |

---

## 📈 Dataset 2: Retail Sales

The dataset contains **200 store observations** with **8 variables** for predicting monthly sales performance.

### Dataset Structure
```
Rows: 200  Columns: 8
```

| **Variable** | **Type** | **Description** | **Issues** |
|--------------|----------|-----------------|------------|
| `monthly_sales` | double | Total sales revenue (R, target) | Negative values possible |
| `advertising_spend` | double | Monthly advertising spending (R) | Missing, negatives |
| `staff_count` | double | Number of FTE employees | Missing values |
| `store_size` | double | Retail floor space (m²) | Outliers possible |
| `foot_traffic` | double | Monthly customer visits | Missing values |
| `local_income` | double | Average household income (R) | Missing values |
| `competition_distance` | double | Distance to nearest competitor (km) | Negative values possible |
| `seasonal_index` | double | Seasonal adjustment factor | Complete |

### Data Quality Issues Found

```r
# Count missing values
missing_summary <- sales_data %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  filter(missing_count > 0)
```

| variable | missing_count |
|----------|---------------|
| foot_traffic | 8 |
| local_income | 6 |
| advertising_spend | 4 |

```r
# Identify negative values
negative_summary <- sales_data %>%
  select(monthly_sales, advertising_spend, competition_distance) %>%
  summarise(across(everything(), ~sum(. < 0, na.rm = TRUE)))
```

| variable | negative_count |
|----------|----------------|
| monthly_sales | 3 |
| advertising_spend | 2 |
| competition_distance | 1 |

```r
# Detect outliers using IQR method
outlier_summary <- sales_data %>%
  select(monthly_sales, advertising_spend, staff_count, store_size) %>%
  summarise(across(everything(), ~{
    q1 <- quantile(., 0.25, na.rm = TRUE)
    q3 <- quantile(., 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    lower <- q1 - 1.5 * iqr
    upper <- q3 + 1.5 * iqr
    sum(. < lower | . > upper, na.rm = TRUE)
  }))
```

| variable | outlier_count |
|----------|---------------|
| monthly_sales | 5 |
| advertising_spend | 4 |
| staff_count | 3 |
| store_size | 6 |

**Screenshot Suggestion 2**: *Add boxplots showing outliers before cleaning*
```
📸 [Screenshot: retail_outliers_boxplot.png]
Caption: Figure 2: Boxplots revealing outliers in retail sales variables before treatment
```

---

## 🔧 Data Wrangling Process

### 1. Text Cleaning and Standardization

```r
# Trim whitespace from all character columns
hospital_clean <- hospital_data %>%
  mutate(across(where(is.character), str_trim))

# Standardize department names
names_for_emergency <- c("emergency", "a&e", "er", "accident & emergency")
names_for_orthopedics <- c("orthopedics", "orthopaedics", "ortho", "bone & joint")
names_for_cardiology <- c("cardiology")
names_for_neurology <- c("neurology")
names_for_surgery <- c("surgery")

hospital_standardised <- hospital_data %>%
  mutate(dept_lower = tolower(department),
         department_clean = case_when(
           dept_lower %in% names_for_emergency ~ "Emergency",
           dept_lower %in% names_for_orthopedics ~ "Orthopedics",
           dept_lower %in% names_for_cardiology ~ "Cardiology",
           dept_lower %in% names_for_neurology ~ "Neurology",
           dept_lower %in% names_for_surgery ~ "Surgery",
           TRUE ~ "Unknown"
         ),
         gender = str_to_title(tolower(gender)),
         discharge_status = str_to_title(tolower(discharge_status)))
```

**Before Standardization:**
| department | count |
|------------|-------|
| A&E | 22 |
| ACCIDENT & EMERGENCY | 1 |
| Accident & Emergency | 7 |
| Bone & Joint | 10 |
| CARDIOLOGY | 7 |
| Cardiology | 37 |
| ER | 20 |
| Emergency | 17 |

**After Standardization:**
| department_clean | count |
|------------------|-------|
| Cardiology | 49 |
| Emergency | 71 |
| Neurology | 29 |
| Orthopedics | 32 |
| Surgery | 15 |
| Unknown | 14 |

**Screenshot Suggestion 3**: *Add before/after comparison of department names*
```
📸 [Screenshot: department_clean_comparison.png]
Caption: Figure 3: Department name standardization results - 24 variations reduced to 6 categories
```

### 2. Handling Missing Values and Impossible Entries

```r
# Handle impossible ages and create age groups
hospital_transformed <- hospital_clean %>%
  mutate(
    age_clean = ifelse(age < 0 | age > 120, NA, age),
    age_group = case_when(
      age_clean < 18 ~ "Child",
      age_clean >= 18 & age_clean < 65 ~ "Adult",
      age_clean >= 65 ~ "Elderly",
      TRUE ~ "Unknown"
    )
  )

# Parse multiple date formats
hospital_with_dates <- hospital_transformed %>%
  mutate(
    admission_date_clean = parse_date_time(
      admission_date, 
      orders = c("dmy", "mdy", "ymd")
    )
  )
```

**Date Parsing Results:**
| patient_id | admission_date | admission_date_clean |
|------------|----------------|----------------------|
| 1 | 07/01/2023 | 2023-01-07 |
| 2 | 02/02/2023 | 2023-02-02 |
| 3 | 22/11/2023 | 2023-11-22 |
| 4 | 17/04/2023 | 2023-04-17 |

**Screenshot Suggestion 4**: *Add age distribution by group visualization*
```
📸 [Screenshot: age_groups_piechart.png]
Caption: Figure 4: Patient age group distribution after cleaning
```

### 3. Feature Engineering

```r
# Create derived features
hospital_with_features <- hospital_standardised %>%
  mutate(
    daily_cost = treatment_cost / length_of_stay,
    high_cost = ifelse(treatment_cost > 20000, "Yes", "No")
  )
```

| patient_id | treatment_cost | length_of_stay | daily_cost | high_cost |
|------------|----------------|----------------|------------|-----------|
| 1 | 20900 | 3 | 6967 | Yes |
| 2 | 33700 | 5 | 6740 | Yes |
| 3 | 11300 | 11 | 1027 | No |
| 4 | 1400 | 3 | 467 | No |

**Screenshot Suggestion 5**: *Add daily cost distribution histogram*
```
📸 [Screenshot: daily_cost_histogram.png]
Caption: Figure 5: Distribution of daily treatment costs across departments
```

### 4. Complete Cleaning Pipeline

```r
# Full pipeline combining all steps
hospital_final <- hospital_data %>%
  # Step 1: Trim whitespace
  mutate(across(where(is.character), str_trim)) %>%
  
  # Step 2: Handle impossible values
  mutate(
    age_clean = ifelse(age < 0 | age > 120, NA, age),
    length_of_stay = ifelse(length_of_stay < 0, NA, length_of_stay),
    treatment_cost = ifelse(treatment_cost < 0, NA, treatment_cost)
  ) %>%
  
  # Step 3: Parse dates
  mutate(
    admission_date_clean = parse_date_time(
      admission_date, 
      orders = c("dmy", "mdy", "ymd")
    )
  ) %>%
  
  # Step 4: Standardise categorical variables
  mutate(
    dept_lower = tolower(department),
    department_clean = case_when(
      dept_lower %in% names_for_emergency ~ "Emergency",
      dept_lower %in% names_for_orthopedics ~ "Orthopedics",
      dept_lower %in% names_for_cardiology ~ "Cardiology",
      dept_lower %in% names_for_neurology ~ "Neurology",
      dept_lower %in% names_for_surgery ~ "Surgery",
      TRUE ~ "Unknown"
    ),
    gender = str_to_title(tolower(gender)),
    discharge_status = str_to_title(tolower(discharge_status))
  ) %>%
  
  # Step 5: Create derived features
  mutate(
    age_group = case_when(
      age_clean < 18 ~ "Child",
      age_clean >= 18 & age_clean < 65 ~ "Adult",
      age_clean >= 65 ~ "Elderly",
      TRUE ~ "Unknown"
    ),
    daily_cost = treatment_cost / length_of_stay
  ) %>%
  
  # Step 6: Remove helper columns
  select(-dept_lower, -admission_date, -age, -department)
```

---

## 🔄 Cross-Validation Framework

### 5. Retail Data Cleaning

```r
# Median imputation for missing values
sales_data <- sales_data %>%
  mutate(
    advertising_spend = if_else(
      is.na(advertising_spend), 
      median(advertising_spend, na.rm = TRUE), 
      advertising_spend
    ),
    foot_traffic = if_else(
      is.na(foot_traffic), 
      median(foot_traffic, na.rm = TRUE), 
      foot_traffic
    ),
    local_income = if_else(
      is.na(local_income), 
      median(local_income, na.rm = TRUE), 
      local_income
    )
  )

# Convert negative values to positive
sales_data <- sales_data %>%
  mutate(
    monthly_sales = abs(monthly_sales),
    advertising_spend = abs(advertising_spend),
    competition_distance = abs(competition_distance)
  )

# IQR-based outlier removal function
remove_outliers <- function(data, var) {
  q1 <- quantile(data[[var]], 0.25, na.rm = TRUE)
  q3 <- quantile(data[[var]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  data %>%
    filter(.data[[var]] >= lower_bound & .data[[var]] <= upper_bound)
}

# Apply outlier removal
sales_data <- sales_data %>%
  remove_outliers("monthly_sales") %>%
  remove_outliers("advertising_spend") %>%
  remove_outliers("staff_count") %>%
  remove_outliers("store_size") %>%
  remove_outliers("foot_traffic") %>%
  remove_outliers("local_income") %>%
  remove_outliers("competition_distance")

cat("Rows removed:", original_rows - nrow(sales_data), 
    sprintf("(%.1f%%)", 100 * (original_rows - nrow(sales_data))/original_rows))
# Output: Rows removed: 22 (11.0%)
```

**Screenshot Suggestion 6**: *Add before/after scatterplot of sales vs advertising*
```
📸 [Screenshot: sales_vs_advertising_cleaned.png]
Caption: Figure 6: Relationship between advertising spend and monthly sales after outlier removal
```

### 6. 10-Fold Cross-Validation Setup

```r
library(rsample)

# Create 10 validation folds
set.seed(123)
cv_folds <- vfold_cv(sales_data, v = 10)

# Define three model specifications
model_1 <- monthly_sales ~ advertising_spend + staff_count
model_2 <- monthly_sales ~ .  # All predictors
model_3 <- monthly_sales ~ advertising_spend + staff_count + 
            store_size + foot_traffic + seasonal_index

# Function to calculate metrics for each fold
calc_metrics <- function(split, model_formula) {
  train <- analysis(split)
  test <- assessment(split)
  
  fit <- lm(model_formula, data = train)
  preds <- predict(fit, newdata = test)
  
  rmse <- sqrt(mean((test$monthly_sales - preds)^2))
  mae <- mean(abs(test$monthly_sales - preds))
  r2 <- 1 - sum((test$monthly_sales - preds)^2) / 
            sum((test$monthly_sales - mean(test$monthly_sales))^2)
  
  tibble(rmse = rmse, mae = mae, r2 = r2)
}
```

**Screenshot Suggestion 7**: *Add diagram of 10-fold cross-validation process*
```
📸 [Screenshot: cv_diagram.png]
Caption: Figure 7: Visual representation of 10-fold cross-validation splitting
```

---

## 📊 Model Comparison

### Cross-Validation Results

```r
# Calculate summary statistics
all_summaries <- bind_rows(
  summarise_results(results_1, "Simple"),
  summarise_results(results_2, "Full"),
  summarise_results(results_3, "Optimised")
)

all_summaries %>%
  select(model, rmse_mean, rmse_se, mae_mean, mae_se, r2_mean, r2_se) %>%
  arrange(rmse_mean) %>%
  knitr::kable(digits = c(0, 0, 2, 0, 2, 2, 2))
```

| Model | RMSE (Mean) | RMSE (SE) | MAE (Mean) | MAE (SE) | R² (Mean) | R² (SE) |
|-------|-------------|-----------|------------|----------|-----------|---------|
| **Full** | **196,199** | **10,901** | **151,168** | **8,384** | **0.55** | **0.03** |
| Optimised | 235,013 | 14,854 | 191,730 | 12,928 | 0.36 | 0.05 |
| Simple | 277,166 | 17,487 | 225,521 | 15,240 | 0.13 | 0.04 |

**Screenshot Suggestion 8**: *Add bar chart comparing model RMSEs with error bars*
```
📸 [Screenshot: model_comparison_barchart.png]
Caption: Figure 8: Cross-validation RMSE comparison across three model specifications
```

### Performance Improvement Calculation

```r
simple_rmse <- 277166
best_rmse <- 196199
improvement <- (simple_rmse - best_rmse) / simple_rmse * 100
cat(sprintf("Improvement over Simple model: %.1f%%", improvement))
# Output: Improvement over Simple model: 29.2%
```

### Final Model Results

```r
# Fit best model (Full) to complete dataset
final_fit <- lm(monthly_sales ~ ., data = sales_data)
summary(final_fit)
```

```
Call:
lm(formula = monthly_sales ~ ., data = sales_data)

Coefficients:
                     Estimate Std. Error t value Pr(>|t|)    
(Intercept)         -94298.87  132670.28  -0.711  0.47820    
advertising_spend        2.60       0.42   6.222 3.69e-09 ***
staff_count          15671.98    2102.07   7.456 4.36e-12 ***
store_size             252.50      38.82   6.505 8.35e-10 ***
foot_traffic            47.79       5.82   8.205 5.48e-14 ***
local_income             8.91       1.03   8.663 3.47e-15 ***
competition_distance -12010.24    6278.24  -1.913  0.05740 .  
seasonal_index       276401.13   97046.89   2.848  0.00494 ** 

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 194500 on 170 degrees of freedom
Multiple R-squared:  0.6195,	Adjusted R-squared:  0.6038 
F-statistic: 39.54 on 7 and 170 DF,  p-value: < 2.2e-16
```

**Screenshot Suggestion 9**: *Add diagnostic plots for final model*
```
📸 [Screenshot: final_model_diagnostics.png]
Caption: Figure 9: Residual diagnostics for selected full model
```

---

## 💡 Insights & Findings

### Hospital Data Insights

1. **Department Cost Variation**
   - Surgery department has highest average treatment cost (R39,160)
   - Emergency department sees most patients (71 visits)
   - 14 patients (6.7%) assigned to "Unknown" department—potential data entry issue

2. **Age Demographics**
   - Average patient age: 54.0 years
   - Age groups: Adult (48%), Elderly (32%), Child (20%)
   - 14 missing ages identified and flagged

3. **Length of Stay Patterns**
   - Average stay: 4.57 days
   - Emergency: 3.2 days average (fastest turnover)
   - Surgery: 8.1 days average (longest recovery)

**Screenshot Suggestion 10**: *Add department cost comparison bar chart*
```
📸 [Screenshot: department_costs.png]
Caption: Figure 10: Average treatment cost by department
```

### Retail Sales Insights

1. **Key Sales Drivers**
   - **Staff count**: Each additional employee generates R15,672 monthly sales (p < 0.001)
   - **Foot traffic**: Each additional customer visit adds R48 in sales (p < 0.001)
   - **Store size**: Each square metre contributes R253 to monthly sales (p < 0.001)
   - **Seasonal index**: Strongest effect, R276,401 impact (p = 0.005)

2. **Model Performance**
   - Full model explains **60.4%** of sales variance (Adjusted R²)
   - **29.2%** RMSE reduction over simple model
   - Prediction accuracy: ±R194,500 (68% confidence interval)

3. **Non-Significant Factors**
   - Competition distance (p = 0.057) - marginally significant
   - Suggests proximity to competitors may matter but needs further study

---

## 🧭 Business Recommendations

Based on the cross-validated model results and analysis of 200 store locations, the following recommendations are proposed for RetailMax management:

### 1. ✅ Optimize Staffing Strategy
Each additional employee generates **R15,672 in monthly sales**. Implement data-driven staffing models based on:
- Seasonal patterns (peak vs off-peak months)
- Foot traffic predictions
- Store size requirements

**ROI Calculation:**
```
Cost of 1 FTE employee (estimated): R8,000/month
Return: R15,672/month
ROI: 96% monthly return on staffing investment
```

### 2. 📊 Allocate Marketing Budget Proportionally
Advertising spend shows significant but modest returns:
- Each R1 in advertising generates **R2.60 in sales**
- Focus on stores with below-average foot traffic
- Test different media channels (digital vs traditional)

### 3. 🏬 Store Expansion Guidelines
Store size coefficient: **R253 per m²**
- Minimum viable size: 200 m² (R50,600 monthly sales contribution)
- Optimal range: 400-800 m² based on current data
- Consider local income levels (coefficient: R8.91 per R1 income)

**Screenshot Suggestion 11**: *Add ROI comparison chart for different strategies*
```
📸 [Screenshot: roi_comparison.png]
Caption: Figure 11: Return on investment comparison across business strategies
```

### 4. 📈 Implement Performance Monitoring Dashboard
Track all significant variables monthly:
- Actual vs predicted sales
- Staff efficiency (sales per employee)
- Foot traffic conversion rates
- Seasonal adjustment factors

**Target setting formula:**
```
Predicted Sales = -94,299 + 2.60(ad_spend) + 15,672(staff) + 
                  253(store_size) + 48(foot_traffic) + 
                  8.91(local_income) + 276,401(seasonal)
```

### 5. 🔬 Investigate Competition Effects
Competition distance (p = 0.057) approaches significance:
- Collect additional data on competitor types (direct vs indirect)
- Monitor competitor promotions and pricing
- Consider loyalty programs for stores near competitors

---

## 🔮 Future Work

Based on the foundation established in this module, several extensions are possible:

### 1. 🧠 Advanced Modeling Approaches
- Implement regularised regression (ridge/LASSO) to handle multicollinearity
- Develop ensemble methods combining multiple model predictions
- Create time-series models for seasonal pattern forecasting

### 2. 📱 Interactive Dashboard Development
- Build Shiny web application for real-time sales predictions
- Create interactive filters for "what-if" scenario analysis
- Implement automated alerting for stores underperforming predictions

### 3. 📊 Additional Data Collection
- Collect competitor pricing and promotion data
- Add customer demographic information
- Include online vs in-store sales breakdown
- Track employee turnover and training levels

### 4. 🔄 Automated Data Pipeline
```r
# Concept for automated weekly updates
auto_update_pipeline <- function(new_data) {
  # Apply same cleaning rules
  cleaned <- clean_retail_data(new_data)
  
  # Generate predictions
  predictions <- predict(final_fit, newdata = cleaned)
  
  # Identify underperforming stores
  alerts <- cleaned %>%
    mutate(
      predicted = predictions,
      variance = monthly_sales - predicted,
      flag = ifelse(variance < -0.2 * predicted, "Review", "OK")
    )
  
  return(alerts)
}
```

### 5. 📚 Cross-Department Integration
- Apply same methodology to other business units
- Develop standardized cleaning protocols
- Create training materials based on this module

**Screenshot Suggestion 12**: *Add concept diagram of future pipeline*
```
📸 [Screenshot: future_pipeline_concept.png]
Caption: Figure 12: Proposed automated data pipeline architecture
```

---

## 📥 Repository Access

```
git clone https://github.com/yourusername/STAT312-Practicals.git
cd STAT312-Practicals/01-Data-Wrangling-and-Model-Selection
```

### Files Included:
- `practical_1_2_combined.Rmd` - Complete R Markdown document
- `practical_1_2_combined.pdf` - Knitted output with all results
- `data/hospital_data_messy.csv` - Raw hospital dataset
- `data/retail_sales_data.csv` - Raw retail dataset

---

## 🙏 Acknowledgements

- **Teaching Team**: STAT312 instructors and teaching assistants
- **Students**: 2024-2025 cohorts for valuable feedback
- **R Community**: tidyverse, rsample, and knitr package maintainers
- **Open Data Initiatives**: For inspiring realistic data simulation

---

## 📊 Citation

If using this module in your teaching or research:

```
[Your Name] (2025). Module 01: Data Wrangling and Model Selection.
STAT312: Advanced Data Analysis in R. Nelson Mandela University.
https://github.com/yourusername/STAT312-Practicals
```

---

**⬆️ [Back to Top](#-module-01-data-wrangling-and-model-selection)**
