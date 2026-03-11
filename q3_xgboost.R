# Q3-Q5 script - run on supercomputer

library(disdat)
library(dplyr)
library(ggplot2)
library(xgboost)
library(precrec)

# load and prep data
nz05df <- bind_cols(select(disPa("NZ"), nz05), disEnv("NZ")) |>
    rename(occ = nz05)
nz05pa <- nz05df |>
    select(!c(group, siteid, x, y, toxicats))

# helper function
F1_score <- function(predicted_class, true_class) {
    confusion <- table(predicted_class, true_class)
    if (nrow(confusion) < 2) return(0)
    recall <- confusion[2, 2] / sum(confusion[, 2])
    precision <- confusion[2, 2] / sum(confusion[2, ])
    score <- 2 * recall * precision / (recall + precision)
    return(score)
}

# fold setup
set.seed(1234)
n <- nrow(nz05pa)
k <- 5
fold_id <- sample(rep(1:k, length.out = n))

#**Q3\.** Train and tune a gradient boosting model across these six boosting parameters 
#( `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `nrounds`, `scale_pos_weight`). 
#The goal is to find a model with excellent predictive performance (not necessarily "the best") 
#and especially to beat the random forest benchmark above. Tune models to maximize the F1 score.

# Q3: Tuned XGBoost with 5-fold CV grid search
# Goal: beat the Q2 benchmark F1 of 0.613

# calculate absence:presence ratio for scale_pos_weight
# this tells us how imbalanced the data is
n_pres_total <- sum(nz05pa$occ)        # count total presences
n_abs_total  <- sum(nz05pa$occ == 0)   # count total absences
ab_pr_ratio  <- n_abs_total / n_pres_total  # ratio e.g. 10 absences per presence

# coarse grid of parameter combinations to try
# expand.grid creates every possible combination of these values
# 2 learning rates x 2 depths x 2 subsamples x 2 colsamples x 2 nrounds x 2 weights = 64 combos
param_grid <- expand.grid(
    learning_rate    = c(0.01, 0.1),    # how big a step each tree takes (small = careful)
    max_depth        = c(3, 6),          # how deep each tree grows (deeper = more complex)
    subsample        = c(0.5, 0.8),      # fraction of rows used per tree (like RF bootstrapping)
    colsample_bytree = c(0.5, 0.8),      # fraction of columns used per tree (like RF's mtry)
    nrounds          = c(500, 1000),     # total number of trees to build (like RF's ntree)
    scale_pos_weight = c(1, ab_pr_ratio) # 1 = no correction, ab_pr_ratio = class imbalance corrected
)

cat("Grid has", nrow(param_grid), "combinations\n")

# empty container to store the average F1 for each parameter combination
grid_f1 <- numeric(nrow(param_grid))

# outer loop: try each parameter combination
for (g in 1:nrow(param_grid)) {
    
    # empty container for F1 across the 5 folds for this combo
    fold_f1 <- numeric(k)
    
    # inner loop: 5-fold CV for this parameter combination
    for (fold in 1:k) {
        
        # split data into train and test for this fold
        train <- nz05pa[fold_id != fold, ]
        test  <- nz05pa[fold_id == fold, ]
        
        # convert to numeric matrix - xgboost requires this format
        x_train <- as.matrix(sapply(train[, -1], as.numeric))
        x_test  <- as.matrix(sapply(test[, -1], as.numeric))
        
        # train xgboost model with this combo's parameters
        model <- xgboost(
            data             = x_train,
            label            = train$occ,
            max_depth        = param_grid$max_depth[g],
            eta              = param_grid$learning_rate[g],      # eta = learning_rate in older xgboost
            subsample        = param_grid$subsample[g],
            colsample_bytree = param_grid$colsample_bytree[g],
            nrounds          = param_grid$nrounds[g],
            scale_pos_weight = param_grid$scale_pos_weight[g],
            nthread          = 4,                 # use 4 CPU threads for speed
            objective        = "binary:logistic", # classification problem
            verbose          = 0                  # suppress output while running
        )
        
        # get predicted probabilities for test data
        prob <- predict(model, newdata = x_test)
        # convert probabilities to 0/1 using 0.5 threshold
        pred <- 1 * (prob > 0.5)
        # store F1 for this fold
        fold_f1[fold] <- F1_score(pred, test$occ)
    }
    
    # average F1 across all 5 folds for this parameter combo
    grid_f1[g] <- mean(fold_f1)
    cat("Combo", g, "of", nrow(param_grid), "| F1:", grid_f1[g], "\n")
}

# find the parameter combo with the highest F1
best_g <- which.max(grid_f1)
best_params <- param_grid[best_g, ] # save best parameters for Q4/Q5
cat("\n--- Q3: Best XGBoost Parameters ---\n")
print(best_params)
cat("Best F1:", grid_f1[best_g], "\n")
# Best parameters: learning_rate=0.1, max_depth=6, subsample=0.5, 
# colsample_bytree=0.8, nrounds=500, scale_pos_weight=4.914
# Best F1: 0.6013 - close to but not quite beating RF benchmark of 0.613
# This is a coarse grid search - narrowing in could improve further

#**Q4\.** Variable importance plot

# Q4: Variable Importance
# train final model on ALL data using best parameters from grid search
# (CV models only used 80% of data - final model uses 100% for best predictions)

x_all <- as.matrix(sapply(nz05pa[, -1], as.numeric))

nz05_final <- xgboost(
    data             = x_all,
    label            = nz05pa$occ,
    max_depth        = best_params$max_depth,
    eta              = best_params$learning_rate,
    subsample        = best_params$subsample,
    colsample_bytree = best_params$colsample_bytree,
    nrounds          = best_params$nrounds,
    scale_pos_weight = best_params$scale_pos_weight,
    nthread          = 4,
    objective        = "binary:logistic",
    verbose          = 0
)

# plot which environmental variables were most important for predictions
importance_matrix <- xgb.importance(model = nz05_final)
xgb.plot.importance(importance_matrix, top_n = 15,
                    main = "XGBoost Variable Importance")

#**Q5\.** Plot the prediction both as a presence map and a map of probabilities.

# Q5: Prediction Maps
# map the final model's predictions across all of New Zealand

# read in the grid of predictor variables (2km pixels across NZ)
NZ_grid <- read.csv("data/NZ_predictors.csv")

# prepare grid for xgboost (exclude x,y coordinates)
NZ_grid_xgb <- as.matrix(sapply(select(NZ_grid, !c(x, y)), as.numeric))

# predict probability of presence for each grid cell
nz05_grid_prob    <- predict(nz05_final, newdata = NZ_grid_xgb)
# convert to presence/absence using 0.5 threshold
nz05_grid_present <- 1 * (nz05_grid_prob > 0.5)

# map probability of presence
NZ_grid |>
    bind_cols(prob = nz05_grid_prob) |>
    ggplot() +
    geom_tile(aes(x = x, y = y, fill = prob)) +
    scale_fill_viridis_c() +
    coord_equal() +
    theme_void() +
    labs(title = "Predicted Probability of Presence", fill = "Probability")

# map predicted presence/absence
NZ_grid |>
    bind_cols(present = nz05_grid_present) |>
    ggplot() +
    geom_tile(aes(x = x, y = y, fill = factor(present))) +
    coord_equal() +
    theme_void() +
    labs(title = "Predicted Presence/Absence", fill = "Present")