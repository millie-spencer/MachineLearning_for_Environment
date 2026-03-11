# Q3 Narrow Grid Search
# Best coarse params: learning_rate=0.1, max_depth=6, subsample=0.5, 
# colsample_bytree=0.8, nrounds=500, scale_pos_weight=4.914

library(disdat)
library(dplyr)
library(xgboost)

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

ab_pr_ratio <- sum(nz05pa$occ == 0) / sum(nz05pa$occ)

# narrow grid - zoom in around best coarse values
param_grid <- expand.grid(
    learning_rate    = c(0.05, 0.1, 0.15),  # zoom in around 0.1
    max_depth        = c(5, 6, 7),            # zoom in around 6
    subsample        = c(0.4, 0.5, 0.6),      # zoom in around 0.5
    colsample_bytree = c(0.7, 0.8, 0.9),      # zoom in around 0.8
    nrounds          = c(500, 1000),           # try more trees
    scale_pos_weight = ab_pr_ratio             # keep best value
)

cat("Narrow grid has", nrow(param_grid), "combinations\n")

grid_f1 <- numeric(nrow(param_grid))

for (g in 1:nrow(param_grid)) {
    fold_f1 <- numeric(k)
    for (fold in 1:k) {
        train <- nz05pa[fold_id != fold, ]
        test  <- nz05pa[fold_id == fold, ]
        x_train <- as.matrix(sapply(train[, -1], as.numeric))
        x_test  <- as.matrix(sapply(test[, -1], as.numeric))
        model <- xgboost(
            data             = x_train,
            label            = train$occ,
            max_depth        = param_grid$max_depth[g],
            eta              = param_grid$learning_rate[g],
            subsample        = param_grid$subsample[g],
            colsample_bytree = param_grid$colsample_bytree[g],
            nrounds          = param_grid$nrounds[g],
            scale_pos_weight = param_grid$scale_pos_weight[g],
            nthread          = 4,
            objective        = "binary:logistic",
            verbose          = 0
        )
        prob <- predict(model, newdata = x_test)
        pred <- 1 * (prob > 0.5)
        fold_f1[fold] <- F1_score(pred, test$occ)
    }
    grid_f1[g] <- mean(fold_f1)
    cat("Combo", g, "of", nrow(param_grid), "| F1:", grid_f1[g], "\n")
}

best_g <- which.max(grid_f1)
cat("\n--- Narrow Grid Best Parameters ---\n")
print(param_grid[best_g, ])
cat("Best F1:", grid_f1[best_g], "\n")