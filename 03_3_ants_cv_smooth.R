#' ---
#' title: "Ant data: smoothing spline model"
#' author: Brett Melbourne
#' date: 29 Jan 2024 (updated 20 Jan 2026)
#' output:
#'     github_document
#' ---

#' Investigate cross-validation with the ants data and a smoothing-spline model

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
source("source/random_partitions.R") #Function is now in our custom library

#' Forest ant data:

forest_ants <- read.csv("data/ants.csv") |> 
    filter(habitat=="forest")

#' ## Model + training algorithms

#' Example of a smoothing spline model. The training algorithm is penalized
#' least squares. Try running this next block of code to visualize the model
#' predictions for different values of `df`. Here is df=7.

smooth_trained <- smooth.spline(forest_ants$latitude, forest_ants$richness, df=7)
grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
preds <- data.frame(predict(smooth_trained, x=grid_latitude))
forest_ants |> 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=x, y=y)) +
    coord_cartesian(ylim=c(0,20))

#' Using `predict` to ask for predictions from the trained smoothing spline
#' model.

predict(smooth_trained, x=43.2)
predict(smooth_trained, x=forest_ants$latitude)
predict(smooth_trained, x=seq(41, 45, by=0.5))

#' ## Inference algorithm

#' Since it's a small dataset, leave one out cross validation (LOOCV) is a good
#' choice. LOOCV is deterministic for this model.
#' 

#' The CV function is essentially the same as for the polynomial model (largely
#' copy and paste), except we have switched out the polynomial model for the
#' smoothing spline.

# Function to perform k-fold CV for a smoothing spline on ants data
# forest_ants: forest ants dataset (dataframe)
# k:           number of partitions (scalar, integer)
# df:          degrees of freedom in smoothing spline (scalar, integer)
# return:      CV error as MSE (scalar, numeric)
#
cv_smooth_ants <- function(forest_ants, k, df) {
    forest_ants$partition <- random_partitions(nrow(forest_ants), k)
    e <- rep(NA, k)
    for ( i in 1:k ) {
        test_data <- subset(forest_ants, partition == i)
        train_data <- subset(forest_ants, partition != i)
        smooth_trained <- smooth.spline(train_data$latitude, train_data$richness, df=df)
        pred_richness <- predict(smooth_trained, test_data$latitude)$y
        e[i] <- mean((test_data$richness - pred_richness) ^ 2)
    }
    cv_error <- mean(e)
    return(cv_error)
}

#' Test/use the function (in LOOCV mode)

nrow(forest_ants) #22 data points
cv_smooth_ants(forest_ants, k=22, df=7)

#' Explore a grid of values for df (k is always 22 for LOOCV)

grid <- expand.grid(k=nrow(forest_ants), df=2:16)
grid

cv_error <- rep(NA, nrow(grid))
for ( i in 1:nrow(grid) ) {
    cv_error[i] <- cv_smooth_ants(forest_ants, grid$k[i], grid$df[i])
}
result <- cbind(grid, cv_error)

#' Plot the result.

result |>
    ggplot() +
    geom_line(aes(x=df, y=cv_error)) +
    labs(title="LOOCV")

#' We see that MSE prediction error (cv_error) increases dramatically for df
#' beyond 8 or so.

result |> 
    ggplot() +
    geom_line(aes(x=df, y=cv_error)) +
    coord_cartesian(xlim=c(2,8), ylim=c(12,18)) +
    labs(title="LOOCV")

#' Table of results

result

#' LOOCV (k=22) identifies df=3 as the best performing model. Compared to the
#' polynomial we had from before, this model has slightly better prediction
#' accuracy. The best performing smoothing spline and best performing polynomial
#' have about the same degree of wiggliness. You can think of degrees of freedom
#' as representing the number of effective parameters, so a smoothing spline
#' with 3 degrees of freedom is in the same ballpark as a polynomial with 3
#' parameters (polynomial of order 2).
#' 

#' | Model              |   LOOCV   |
#' |--------------------|-----------|
#' | Polynomial 2       |   12.88   |
#' | Smoothing spline 3 |   12.52   |
