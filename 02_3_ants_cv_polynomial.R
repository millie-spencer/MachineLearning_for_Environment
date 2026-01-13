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
#' ### aka just remove the raw=true part 

lm(richness ~ poly(latitude, degree=5), data=forest_ants)
cor(poly(forest_ants$latitude, degree=5))

#' Orthogonal polynomials give the same predictions as the raw polynomials. It's
#' just a difference in parameterization of the same model. In machine learning
#' we don't care about the parameter values, just the resulting prediction, so
#' it's best to choose the more robust parameterization.
#' 

### all just says we'll need to use a different form of the model ...? 
### relatively small dataset, information content is not that great
### polynomial model is not able to caputure the information 
### can be improved by reparameterizing the model to capture more info 
### called an orthagonal polynomial, each new term is mathmatically orthagonal to the previous one 


#' R's `lm()` function contains a **training algorithm** that finds the
#' parameters that minimize the sum of squared deviations of the data from the
#' model. The following code trains the order 4 polynomial and plots the fitted
#' model. Use this block of code to try different values for the order of the
#' polynomial. We can get up to order 16, after which we can no longer form
#' orthogonal polynomials.

order <- 12 #integer ## flexibility of model comes from changing the integer, can change to 1,2,3 etc. eg.
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

#' Exploring the k-fold CV algorithm
#'
#' First, we need a function to divide the dataset up into partitions.

# Function to divide a data set into random partitions for cross-validation
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

#' What does the output of `random_partitions()` look like? It's a set of labels
#' that says which partition each data point belongs to.

random_partitions(nrow(forest_ants), k=5)
random_partitions(nrow(forest_ants), k=nrow(forest_ants))


#' Now code up the k-fold CV algorithm (from our pseudocode to R code) to
#' estimate the prediction mean squared error for one order of the polynomial.
#' Try 5-fold, 10-fold, and n-fold CV. Try different values for polynomial
#' order.

# divide dataset into k parts i = 1...k
# initiate vector to hold e
# for each i
#     test dataset = part i
#     training dataset = remaining data
#     find f using training dataset
#     use f to predict for test dataset 
#     e_i = prediction error (MSE)
# CV_error = mean(e)

