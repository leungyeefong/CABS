###############################################################################
# AACT Clinical Trial Termination Prediction
# Data Preparation for Machine Learning
#
# Output:
#   AACT_prediction_data.csv
#
# Purpose:
#   Prepare one-row-per-trial dataset for:
#   - Logistic Regression
#   - Random Forest
#   - XGBoost
#   - SHAP
#   - AI Agent
###############################################################################


# =============================================================================
# 1. Packages
# =============================================================================

library(data.table)


# =============================================================================
# 2. Working directory
# =============================================================================

setwd(" ") # set to your own path


# =============================================================================
# 3. Import AACT datasets
# =============================================================================

studies <- fread(
  "studies.txt",
  sep = "|",
  na.strings = c("", "NULL", "NA"))

sponsors <- fread(
  "sponsors.txt",
  sep = "|",
  na.strings = c("", "NULL", "NA"))

interventions <- fread(
  "interventions.txt",
  sep = "|",
  na.strings = c("", "NULL", "NA"))

conditions <- fread(
  "conditions.txt",
  sep = "|",
  na.strings = c("", "NULL", "NA"))


# Quick check
dim(studies)
dim(sponsors)
dim(interventions)
dim(conditions)


# =============================================================================
# 4. Create study-level base dataset
# =============================================================================

prediction_data <- copy(studies)


# -----------------------------------------------------------------------------
# Outcome: trial termination
# -----------------------------------------------------------------------------

prediction_data[, terminated := as.integer(overall_status == "TERMINATED")]


# -----------------------------------------------------------------------------
# Start year
# -----------------------------------------------------------------------------

prediction_data[, start_year := as.integer(substr(start_date, 1, 4))]


# =============================================================================
# 5. Restrict to trials with a known final outcome
# =============================================================================

prediction_data <- prediction_data[overall_status %in% c("COMPLETED", "TERMINATED")]

# Restrict prediction analysis to interventional trials
prediction_data <- prediction_data[study_type == "INTERVENTIONAL"]

# Check outcome distribution

table(prediction_data$overall_status, useNA = "ifany")

table(prediction_data$terminated, useNA = "ifany")



# =============================================================================
# 6. Lead sponsor
# =============================================================================

lead_sponsor <- sponsors[lead_or_collaborator == "lead",
  .(agency_class = first(agency_class)), by = nct_id]


prediction_data <- merge(prediction_data, lead_sponsor, by = "nct_id", all.x = TRUE)


# Missing sponsor category

prediction_data[is.na(agency_class), agency_class := "UNKNOWN"]


# =============================================================================
# 7. Intervention types
#
# Create one-hot indicators:
# DRUG
# DEVICE
# BIOLOGICAL
# BEHAVIORAL
# PROCEDURE
# etc.
# =============================================================================

intervention_wide <- dcast(
  unique(
    interventions[
      !is.na(intervention_type),
      .(nct_id, intervention_type)]),
  nct_id ~ intervention_type,
  fun.aggregate = length,
  value.var = "intervention_type")


intervention_cols <- setdiff(names(intervention_wide), "nct_id")


# Convert counts to 0/1

intervention_wide[,(intervention_cols) :=
    lapply(.SD, function(x) as.integer(x > 0)), .SDcols = intervention_cols]


# Merge into trial dataset

prediction_data <- merge(prediction_data,
  intervention_wide, by = "nct_id", all.x = TRUE)


# Trials without a recorded intervention:
# set intervention indicators to 0

prediction_data[,
  (intervention_cols) :=
    lapply(.SD,
      function(x) {
        x[is.na(x)] <- 0
        x}),.SDcols = intervention_cols]


# =============================================================================
# 8. Disease features
#
# Select Top 30 diseases by number of trials.
# Then create disease one-hot indicators.
# =============================================================================

top30_diseases <- conditions[!is.na(name),
  .(trials = uniqueN(nct_id)),by = name][order(-trials)][1:30]


top30_disease_names <- top30_diseases$name


# Keep only Top 30 diseases

condition_top30 <- conditions[name %in% top30_disease_names,.(nct_id, name)]


# One-hot disease indicators

disease_wide <- dcast(unique(condition_top30),
  nct_id ~ name, fun.aggregate = length, value.var = "name")


disease_cols <- setdiff(names(disease_wide), "nct_id")


# Convert to 0/1

disease_wide[,(disease_cols) :=
    lapply(.SD, function(x) as.integer(x > 0)),.SDcols = disease_cols]


# Add DISEASE_ prefix so disease variables are easy to identify in Python

new_disease_names <- paste0("DISEASE_", disease_cols)

setnames(disease_wide, old = disease_cols, new = new_disease_names)

disease_cols <- new_disease_names


# Merge disease features

prediction_data <- merge(prediction_data, disease_wide, by = "nct_id", all.x = TRUE)


# Trials not belonging to one of the Top 30 diseases get 0

prediction_data[,(disease_cols) :=
    lapply(.SD, function(x) {
        x[is.na(x)] <- 0
        x}),.SDcols = disease_cols]


# =============================================================================
# 9. Clean categorical variables
# =============================================================================

prediction_data[is.na(phase), phase := "UNKNOWN"]

prediction_data[is.na(study_type), study_type := "UNKNOWN"]


# =============================================================================
# 10. Select variables for machine-learning dataset
# =============================================================================

base_features <- c(
  "nct_id", "terminated", "phase", "study_type", "enrollment", "agency_class", "start_year")

final_columns <- c(base_features, intervention_cols, disease_cols)

prediction_data_final <- prediction_data[,..final_columns]


# =============================================================================
# 11. Remove impossible enrollment values
#
# Keep missing enrollment for now.
# We will impute missing values using TRAINING data in Python
# to avoid data leakage.
# =============================================================================

prediction_data_final[!is.na(enrollment) & enrollment <= 0, enrollment := NA_real_]


# =============================================================================
# 12. Final checks
# =============================================================================

dim(prediction_data_final)

table(prediction_data_final$terminated)

prop.table(table(prediction_data_final$terminated))

summary(prediction_data_final$enrollment)

table(prediction_data_final$phase)

table(prediction_data_final$study_type)

# Number of duplicated trial IDs should be 0

sum(duplicated(prediction_data_final$nct_id))


# Missing values

colSums(is.na(prediction_data_final))


# =============================================================================
# 13. Export modeling dataset
# =============================================================================

fwrite(prediction_data_final, "AACT_prediction_data.csv")


# Also export the Top 30 disease list for reference

fwrite(top30_diseases, "AACT_top30_diseases.csv")




# =============================================================================
# DONE
# =============================================================================

cat("\nPrediction dataset created successfully.\n")

cat("Rows:", nrow(prediction_data_final), "\n")

cat("Columns:", ncol(prediction_data_final), "\n")

cat("Saved as: AACT_prediction_data.csv\n")









# =============================================================================
# Descriptive analysis
# Cohort Flow Summary for Presentation
# Purpose:
# Count the number of trials remaining after each major
# eligibility restriction used to create the modeling cohort.
# =============================================================================

# Step 1. All studies in AACT
n_raw <- nrow(studies)

# Step 2. Keep trials with known final outcome
n_final_outcome <- nrow(
  studies[
    overall_status %in% c("COMPLETED", "TERMINATED")
  ]
)

# Step 3. Restrict to interventional trials
n_interventional <- nrow(
  studies[
    overall_status %in% c("COMPLETED", "TERMINATED") &
      study_type == "INTERVENTIONAL"
  ]
)

# Create presentation table
cohort_flow <- data.frame(
  Step = c(
    "All AACT studies",
    "Completed or terminated trials",
    "Interventional trials",
    "Final analytic cohort"
  ),
  N = c(
    n_raw,
    n_final_outcome,
    n_interventional,
    nrow(prediction_data_final)
  )
)

# Calculate number excluded at each step
cohort_flow$Excluded_from_previous <- c(
  NA,
  n_raw - n_final_outcome,
  n_final_outcome - n_interventional,
  n_interventional - nrow(prediction_data_final)
)

print(cohort_flow)
