# Assignment 1 Millie Spencer

**Due:** Tue 27 Jan 3:30 PM

**Grading criteria:** Complete all the check boxes below. On time submission.

**Percent of grade:** 7%

**Push your files to your GitHub repository**


## **Learning goals:** 

* Understand the steps of the cross validation (CV) algorithm
* Build competence in translating algorithms to code
* Practice tuning a machine learning algorithm using the CV inference algorithm

Investigating algorithms line by line or coding algorithms from scratch gives you much deeper understanding for how they work, provides detailed knowledge of what they are actually doing, and builds intuition that you can draw on throughout your career.


## Understanding the random partitions algorithm

Here is the random partitions algorithm that we used in class:

The R version

```R
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
```

The Python version

```python
# Function to divide a data set into random partitions for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of partitions (scalar, integer)
# rng:     numpy random generator, set ahead rng = np.random.default_rng()
# return:  partition labels ranging from 0 to k-1 (vector, integer)
# 
def random_partitions(n, k, rng):
    min_n = n // k
    extras = n - k * min_n
    labels = np.concatenate([np.repeat(np.arange(k), min_n), 
             np.arange(extras)])
    partitions = rng.choice(labels, n, replace=False)
    return partitions
```

- [ ] Choose either the R or Python version, and describe what each line of code does in the above random partitions algorithm

# For the R version: 

# Function to divide a dataset into random partitions for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of partitions (scalar, integer)
# return:  partition labels (vector, integer)
# 
random_partitions <- function(n, k) { #creates a function called random partitions, inputs are total number of data points and the number of folds we want
    min_n <- floor(n / k) #calculates min number of data points per partition. 
    extras <- n - k * min_n # calculates remainder. how many partitions will get an extra data point?
    labels <- c(rep(1:k, each=min_n),rep(seq_len(extras))) # allocates each number to a partition, including the remainders, and makes them into a vector
    partitions <- sample(labels, n) #randomly assigns data points to partitions 
    return(partitions) #return/print the vector 
}

## Coding the LOOCV algorithm 


The leave one out cross validation algorithm is a special case of the k-fold CV algorithm. We can use the k-fold CV algorithm that we coded in class to do LOOCV by setting k equal to the number of data points. But LOOCV is a special case that suggests an even simpler algorithm. This algorithm **does not need** the function `random_partitions()`. Code up the LOOCV algorithm in R or Python from the following pseudocode (literally translate the pseudocode line by line).

```
# LOOCV algorithm
# for each data point
#     fit model without point
#     predict for that point
#     measure prediction error (compare to observed)
# CV_error = mean error across points
```

Use the first section of [02_3_ants_cv_polynomial.R](02_3_ants_cv_polynomial.R) or [02_3_ants_cv_polynomial.py](02_3_ants_cv_polynomial.py) to get going with reading in the data and using a polynomial model.

- [ ] As we did for coding the k-fold CV algorithm, first code the LOOCV algorithm line by line. Include this line-by-line version in your submission.

- [ ] Then turn it into a function. Include the function separately from your original code.

- [ ] Finally, use the function to investigate the LOOCV error for different orders of the polynomial model to determine the order with the best predictive accuracy. This code will be substantially similar to the code we wrote in class but you'll be using the LOOCV function you just wrote.

## 1. First read in the data and build the polynomial 
#load packages: 
library(ggplot2)
library(dplyr)
library(tidyr) #for pivot_longer()

#load ant data: 
ants <- read.csv("data/ants.csv")
head(ants) # prints first 6 rows 

#plot ant data for observation, ensure correct loading, change y lim to see full data range
forest_ants |>
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    ylim(0,20)


## 2. Then turn pseudo code into a real R code for LOOCV line by line: 
# LOOCV algorithm
# for each data point
#     fit model without point
#     predict for that point
#     measure prediction error (compare to observed)
# CV_error = mean error across points

order <- 2 # set number of degrees in polynomial
n <- nrow(forest_ants) # n=22 because doing LOOCV 
e <- rep(NA, n) #creates an empty vector with 22 spaces to store the errors for each row

# for each data point
for (i in 1:n) {
    # fit model without point i (leave one out)
    train_data <- forest_ants[-i, ]
    f_trained <- lm(richness ~ poly(latitude, order), data=train_data)
    
    # predict for that point (the left-out point)
    test_data <- forest_ants[i, ]
    pred_richness <- predict(f_trained, newdata=test_data)
    
    # measure prediction error
    e[i] <- (test_data$richness - pred_richness)^2
}

# CV_error = mean error
cv_error <- mean(e)
cv_error
# shows us that with order=2 in our LOOCV, prediction error is 12.88
# can experiment with different orders to find lowest error 


## 3. Turn above code into a function 

loocv_poly_ants <- function(forest_ants, order) {
    n <- nrow(forest_ants)
    e <- rep(NA, n)
    
    for (i in 1:n) {
        train_data <- forest_ants[-i, ]
        f_trained <- lm(richness ~ poly(latitude, order), data=train_data)
        test_data <- forest_ants[i, ]
        pred_richness <- predict(f_trained, newdata=test_data)
        e[i] <- (test_data$richness - pred_richness)^2
    }
    
    cv_error <- mean(e)
    return(cv_error)
}

# Test if the function works:
loocv_poly_ants(forest_ants, order=2)  # Should still give 12.88 like in #2 
loocv_poly_ants(forest_ants, order=3)  # Try a different order e.g. 3, here order 3 increases the error compared to order 2

## 4. Use function to investigate LOOCV error for different orders (degrees) of the poly model. ID order with best predictive accuracy (lowest e)

# Test orders 1 through 16 (maximum for this dataset)
orders <- 1:16
cv_error <- rep(NA, length(orders))

for (i in 1:length(orders)) {
    cv_error[i] <- loocv_poly_ants(forest_ants, order=orders[i])
}
# prints the errors for orders 1-16. allows us to compare how wiggly is best at predicting the data. 

# Create results table
results <- data.frame(order=orders, cv_error=cv_error)
results
# big increase in errors after order=8, 16 yields NA 


# Plot the full range to see the dramatic error increase
results |>
    ggplot() +
    geom_line(aes(x=order, y=cv_error)) +
    geom_point(aes(x=order, y=cv_error)) +
    labs(title="LOOCV Error for Different Polynomial Orders",
         x="Polynomial Order",
         y="CV Error (MSE)")

# Scale is bad because of exponential decrease, zoom in on the reasonable range (1-8)
results |>
    filter(order <= 8) |>
    ggplot() +
    geom_line(aes(x=order, y=cv_error)) +
    geom_point(aes(x=order, y=cv_error)) +
    coord_cartesian(ylim=c(10,25)) +
    labs(title="LOOCV Error (zoomed to orders 1-8)",
         x="Polynomial Order",
         y="CV Error (MSE)")

# Identify the best order
best_error <- min(results$cv_error, na.rm=TRUE) #prints error results from diff orders from min error to max, ignoring NA values 
best_order <- results$order[which.min(results$cv_error)] # which order yielded the lowest error?

cat("Lowest CV error:", best_error, "\n") # returns lowest error value (12.87)) 
cat("Best polynomial order:", best_order, "\n") #tells us which order number yielded the lowest error
