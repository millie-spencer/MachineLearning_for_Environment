#' ---
#' title: "Ant data: k-fold cross validation"
#' author: Brett Melbourne
#' date: 13 Jan 2026
#' output:
#'     github_document
#' ---
#' MS note: command+Enter to run code chunks 

#' Explore the cross-validation **inference algorithm** from scratch with the
#' ants data and a polynomial model. Our goal is to predict richness of forest
#' ants from latitude. What order of a polynomial **model algorithm** gives the
#' most accurate predictions?

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(tidyr) #for pivot_longer()

#' Ant data:

ants <- read.csv("data/ants.csv")
head(ants)

#' Forest ant data:

forest_ants <- ants |> 
    filter(habitat=="forest")

forest_ants |>
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    ylim(0,20)


#' ### Model algorithm

#' Here's one way we could code a 3rd order polynomial by first creating new
#' variables for the quadratic (squared) and cubic (cubed) terms, and using R's
#' model formula syntax to train the model by minimizing the SSQ with the
#' function `lm`.

forest_ants$latitude_2 <- forest_ants$latitude ^ 2 # beta2, or latitude_2
forest_ants$latitude_3 <- forest_ants$latitude ^ 3 # beta3, or latitude_3
head(forest_ants)
lm(richness ~ latitude + latitude_2 + latitude_3, data=forest_ants)

#' A model formula provides a shorthand notation for (mostly) linear models,
#' e.g. `y ~ x + z` is shorthand for the model:
#' 
#' $$
#' y = \beta_0 + \beta_1 x + \beta_2 z
#' $$
#' 

#' Here's another way to code the same model that eliminates the need to create
#' new variables for higher order terms.

lm(richness ~ latitude + I(latitude^2) + I(latitude^3), data=forest_ants)

#' The `I()` function ensures that `^` is not interpreted as model formula
#' syntax. See `?formula` for more details about model formulae.

#' An even more convenient way uses the function `poly()`, which creates a
#' matrix of the polynomial terms.

poly(forest_ants$latitude, degree=3, raw=TRUE) 
# degree = 3 is controlling the degree of model flexibility 
# raw=TRUE just means its a form of the algorithm which is the same as the funcion lm before it 

#' We can use this directly within a model formula

lm(richness ~ poly(latitude, degree=3, raw=TRUE), data=forest_ants)

#' A potential problem with polynomial models is that the higher order terms can
#' become almost perfectly correlated with one another, leading to models where
#' the parameters can't all be uniquely estimated. For example, for these data
#' the fourth order polynomial can be trained but for the fifth order polynomial
#' we can't determine a unique value for the highest order parameter, and the
#' parameter estimates remain the same as the fourth order model. We have
#' essentially run out of uniqueness among the polynomial terms due to the high
#' correlations.

lm(richness ~ poly(latitude, degree=4, raw=TRUE), data=forest_ants)
lm(richness ~ poly(latitude, degree=5, raw=TRUE), data=forest_ants)
cor(poly(forest_ants$latitude, degree=5, raw=TRUE))

#' This problem can be markedly reduced by using orthogonal polynomials, which
#' remove the correlation among the polynomial terms. Orthogonal polynomials are
#' the default type for `poly()`.

lm(richness ~ poly(latitude, degree=5), data=forest_ants)
cor(poly(forest_ants$latitude, degree=5))

#' Orthogonal polynomials give the same predictions as the raw polynomials. It's
#' just a difference in parameterization of the same model. In machine learning
#' we don't care about the parameter values, just the resulting prediction, so
#' it's best to choose the more robust parameterization.
#' 

#' R's `lm()` function contains a **training algorithm** that finds the
#' parameters that minimize the sum of squared deviations of the data from the
#' model. The following code trains the order 4 polynomial and plots the fitted
#' model. Use this block of code to try different values for the order of the
#' polynomial. For this small dataset, we can get up to order 16, after which we
#' can no longer form orthogonal polynomials.

order <- 8 #integer ## flexibility of model comes from changing the integer, can change to 1,2,3 etc. eg.
poly_trained <- lm(richness ~ poly(latitude, order), data=forest_ants)
grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
nd <- data.frame(latitude=grid_latitude)
pred_richness <- predict(poly_trained, newdata=nd)
preds <- cbind(nd, richness=pred_richness)

ggplot(data=NULL, aes(x=latitude, y=richness)) +
    geom_point(data=forest_ants) +
    geom_line(data=preds) +
    coord_cartesian(ylim=c(0,20)) +
    labs(title=paste("Polynomial order", order))

### what form of the model gives us the optimal amount of wiggliness so that the model is good, accurately predicts new data 
### we'll use cross validation model to determine the correct amount of wiggliness (flexibility)

#' Use `predict` to ask for predictions from the trained polynomial model. For
#' example, here we are asking for the prediction at latitude 43.2 and we find
#' the predicted richness is 5.45. We need to provide the predictor variable
#' `latitude` as a data frame even if it's just one value. See `?predict.lm`.

predict(poly_trained, newdata=data.frame(latitude=43.2))


#' ### Inference algorithm

#' Exploring the k-fold CV algorithm
#'
#' First, we need a function to divide the dataset up into partitions.

# Function to divide a dataset into random partitions for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of partitions (scalar, integer)
# return:  partition labels (vector, integer)
# 
random_partitions <- function(n, k) {
    min_n <- floor(n / k)
    extras <- n - k * min_n
    labels <- c(rep(1:k, each=min_n),rep(seq_len(extras)))
    partitions <- sample(labels, n)
    return(partitions)
}

#' What does the output of `random_partitions()` produce? It's a set of labels
#' that says which partition each data point belongs to.
# nrow = number of datapoints is the number of rows (22) in the ants dataset
random_partitions(nrow(forest_ants), k=5) #sets k = 5
# returns partition labels (a vector of integers) in this case, a list of numbers 1-5 allocated evenly and randomly to the rows
# we should see 22/5 of each (1-5), so we see each 4 times and for 2 we'll see 5
# if we run the code again the folds should look different, since random partition
random_partitions(nrow(forest_ants), k=nrow(forest_ants)) # sets k = 22 
# if k=22, then we should see each number 1-22 once, again in random order
# will get the same result (error) every time though, since there's one label per row

#' Now code up the k-fold CV algorithm (from our pseudocode to R code) to
#' estimate the prediction mean squared error for one order of the polynomial.
#' Try 5-fold, 10-fold, and n-fold CV. Try different values for polynomial
#' order.

# divide dataset into k parts i = 1...k
forest_ants$partition <- random_partitions(nrow(forest_ants), k)

e <- rep(NA, k) # we'll need this to store the error from each test

# for each i
for ( i in 1:k ) {
#     test dataset = part i
    test_data <- subset(forest_ants, partition == i)
#     training dataset = remaining data
    train_data <- subset(forest_ants, partition != i)
#     find f using training dataset
    f_trained <- lm(richness ~ poly(latitude, order), data=train_data)
#     use f to predict for test dataset
    pred_richness <- predict(f_trained, newdata=test_data)
#     e_i = prediction error (MSE)
    e[i] <- mean((test_data$richness - pred_richness) ^ 2)
}
# CV_error = mean(e)

