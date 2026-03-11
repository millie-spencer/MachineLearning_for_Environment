# Assignment 4: Random Forest and Boosting
# Author: Millie Spencer
# Due: March 5, 2026

# Libraries
library(disdat)
library(dplyr)
library(ggplot2)
# library(sf) #momentarily hashing out- having issue w package in supercomputer
library(randomForest)
library(gbm)
library(xgboost)
library(precrec)

# ── Load Data ─────────────────────────────────────────────────────────────────
nz05df <- bind_cols(select(disPa("NZ"), nz05), disEnv("NZ")) |>
    rename(occ = nz05)
head(nz05df)

# # Outline of New Zealand
# nzpoly <- disBorder("NZ")
# class(nzpoly)

# # Plot presence/absence data
# nz05df |>
#     arrange(occ) |>
#     ggplot() +
#     geom_sf(data = nzpoly, fill = "lightgray") +
#     geom_point(aes(x = x, y = y, col = factor(occ)), shape = 1, alpha = 0.2) +
#     theme_void()
# #1 = presence, 0 = absence

# Data for modeling (drop non-predictor columns)
nz05pa <- nz05df |>
    select(!c(group, siteid, x, y, toxicats)) # variables we want to remove
head(nz05pa)
# occ is present or absent (0 or 1)
# age, deficit, DEM, hillshade... remaining variables 

# ── Helper Functions ──────────────────────────────────────────────────────────
F1_score <- function(predicted_class, true_class) {
    confusion <- table(predicted_class, true_class)
    # if model predicts all absent, return 0 (no presences predicted)
    if (nrow(confusion) < 2) return(0)
    recall <- confusion[2, 2] / sum(confusion[, 2])
    precision <- confusion[2, 2] / sum(confusion[2, ])
    score <- 2 * recall * precision / (recall + precision)
    return(score)
}

get_AUC <- function(scores, labels) {
    ROC_curve <- evalmod(scores = scores, labels = labels) #ROC curve is a way to visualize how well the model performs 
    #the model outputs a probability (e.g. 0.3, 0.7, 0.95). You then pick a threshold to decide "above this = present, below = absent." The default is 0.5 but you could pick any number.
    #At each threshold it calculates the true presence rate and false presence rate, and plots that as a single point. Connect all those points and you get the ROC curve.
    auc(ROC_curve)[1, 4] #The AUC is just the area under that curve — one number that summarizes the whole thing. Higher = better across all thresholds.
}

# ── 5-Fold CV Setup ───────────────────────────────────────────────────────────
set.seed(1234)
n <- nrow(nz05pa)
k <- 5
fold_id <- sample(rep(1:k, length.out = n))
table(fold_id)
# I now have 19,120 total rows split evenly into 5 folds of 3,824 each

## Assignment questions
#**Q1\.**  Use 5-fold cross validation to estimate overall accuracy, F1 score, and AUC for the basic random forest model above. 
#Don't bother tuning the model. 

# empty containers to store results for each fold
rf_accuracy <- numeric(k)
rf_f1 <- numeric(k)
rf_auc <- numeric(k)

for (fold in 1:k) { # runs 5 times, once per fold
    # rows NOT in this fold = training data, rows IN this fold = test data
    train <- nz05pa[fold_id != fold, ]
    test  <- nz05pa[fold_id == fold, ]
    # random forest needs occ to be a factor (category) not a number
    train_rf <- train
    train_rf$occ <- factor(train_rf$occ)
    # train the model using all predictor variables with 500 trees
    model <- randomForest(occ ~ ., data = train_rf, ntree = 500)
    # get probability of presence for each test row
    prob <- predict(model, newdata = test[, -1], type = "prob")[, 2]
    # convert probability to 0/1 using 0.5 threshold
    pred <- 1 * (prob > 0.5)
    # store accuracy, F1, and AUC for this fold
    rf_accuracy[fold] <- mean(pred == test$occ) # proportion correctly classified
    rf_f1[fold]       <- F1_score(pred, test$occ) # using helper function defined above
    rf_auc[fold]      <- get_AUC(prob, test$occ)  # using helper function defined above
    cat("Fold", fold, "done\n") # progress update
}

# average results across all 5 folds
cat("\n--- Q1: Basic Random Forest ---\n")
cat("Accuracy:", mean(rf_accuracy), "\n")
cat("F1 Score:", mean(rf_f1), "\n")
cat("AUC:     ", mean(rf_auc), "\n")

#--- Q1: Basic Random Forest ---
#Accuracy: 0.8721234 
#F1 Score: 0.5490505 
#AUC:      0.8981595 


#**Q2\.**  Use 5-fold cross validation to estimate overall accuracy, F1 score, and AUC for the class imbalance 
#random forest model above. Again, don't bother tuning the model. How much better or worse does this model perform on the metrics? 
# This is your benchmark model. Can you beat this model with xgboost?

#class imbalance version of random forest adds sampsize=c(n_pres, n_pres) to force equal sampling of presences and absences at each tree.
#  so even though there are way more absences in the real data, each tree sees the same amount of both. This way the model gets a fair chance to learn what presence looks like.

# empty containers to store results for each fold
rf2_accuracy <- numeric(k)
rf2_f1 <- numeric(k)
rf2_auc <- numeric(k)

for (fold in 1:k) { # runs 5 times, once per fold
    
    # rows NOT in this fold = training data, rows IN this fold = test data
    train <- nz05pa[fold_id != fold, ]
    test  <- nz05pa[fold_id == fold, ]
    
    # random forest needs occ to be a factor (category) not a number
    train_rf <- train
    train_rf$occ <- factor(train_rf$occ)
    
    # count presences in training data for equal sampling
    n_pres <- sum(train$occ)
    
    # train with equal sampling of presences and absences at each tree
    model <- randomForest(occ ~ ., data = train_rf, ntree = 500,
                          sampsize = c(n_pres, n_pres))
    
    # get probability of presence for each test row
    prob <- predict(model, newdata = test[, -1], type = "prob")[, 2]
    # convert probability to 0/1 using 0.5 threshold
    pred <- 1 * (prob > 0.5)
    
    # store accuracy, F1, and AUC for this fold
    rf2_accuracy[fold] <- mean(pred == test$occ)
    rf2_f1[fold]       <- F1_score(pred, test$occ)
    rf2_auc[fold]      <- get_AUC(prob, test$occ)
    
    cat("Fold", fold, "done\n") # progress update
}

# average results across all 5 folds
cat("\n--- Q2: Class-Imbalance Random Forest ---\n")
cat("Accuracy:", mean(rf2_accuracy), "\n")
cat("F1 Score:", mean(rf2_f1), "\n")
cat("AUC:     ", mean(rf2_auc), "\n")

mean(rf2_accuracy)
mean(rf2_f1)
mean(rf2_auc)

#--- Q2: Class-Imbalance Random Forest ---
#Accuracy: 0.8286611 
#F1 Score: 0.6132771 
#AUC:      0.8996246 
# mean(rf2_accuracy)
#[1] 0.8286611
#mean(rf2_f1)
#[1] 0.6132771
#mean(rf2_auc)
#[1] 0.8996246

###COMPARING Q1 and Q2: 
#Accuracy went DOWN from 0.87 to 0.83 - Accuracy drops because the model is now making more mistakes on the absence class — it's calling some absences "present" when they're actually absent. Since there are way more absences than presences, those extra mistakes hurt overall accuracy.
#F1 score went UP from 0.55 to 0.61 - F1 goes up because F1 only cares about how well the model does on the presence class — and now the model is finding more true presences it was missing before.
#AUC stayed about the same (0.898 vs 0.900) - AUC stays the same because AUC measures performance across all thresholds, so the overall ranking ability of the model didn't really change.


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