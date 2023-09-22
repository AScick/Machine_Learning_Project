library(MASS)
require(boot)
library("class")
library("nnet")
library("glmnet")
set.seed(17)
path_train <- paste0(getwd(), "/zip.train.gz")
path_test <- paste0(getwd(), "/zip.test.gz")
zip_train <- read.csv(path_train, header = FALSE, sep = " ")
zip_test <- read.csv(path_test, header = FALSE, sep = " ")
im <- matrix(as.numeric(unlist(zip_train[4, 2:257])), nrow = 16, ncol = 16)
image(t(apply(-im, 1, rev)), col = gray((0:32) / 32))
zip_train <- zip_train[zip_train[, 1] %in% c(2, 3), ]
zip_test <- zip_test[zip_test[, 1] %in% c(2, 3), ]
y_train <- zip_train[, 1]
y_test <- zip_test[, 1]
neighbors <- c(1, 3, 5, 7, 15)
knn_results <- data.frame(k = integer(), train_error = numeric(), test_error = numeric(), stringsAsFactors = FALSE)

for (i in 1:length(neighbors)) {
  k <- neighbors[i]
  
  train_estimate <- knn(train = zip_train[, 2:257], test = zip_train[, 2:257], cl = y_train, k = k)
  
  test_estimate <- knn(train = zip_train[, 2:257], test = zip_test[, 2:257], cl = y_train, k = k)
  
  knn_results <- rbind(
    knn_results,
    data.frame(
      k = k,
      train_error = 1 - sum(train_estimate == y_train) / length(y_train),
      test_error = 1 - sum(test_estimate == y_test) / length(y_test),
      stringsAsFactors = FALSE
    )
  )
}
lModel <- multinom(y_train ~ ., data = zip_train[, 2:257], MaxNWts = 5000, maxit = 1000)
test_estimate_multil <- predict(lModel, zip_test[, 2:257], type = "class")
train_estimate_multil <- predict(lModel, zip_train[, 2:257], type = "class")
test_error_multil <- 1 - sum(test_estimate_multil == y_test) / length(y_test)
train_error_multil <- 1 - sum(train_estimate_multil == y_train) / length(y_train)
train_error_multil
test_error_multil
y_train_log <- ifelse(y_train == 2, 0, 1)
y_test_log <- ifelse(y_test == 2, 0, 1)
logistic <- glm(y_train_log ~ ., data = zip_train[, 2:257], family = binomial)
test_estimate_log <- predict(logistic, zip_test[, 2:257], type = "response")
train_estimate_log <- predict(logistic, zip_train[, 2:257], type = "response")
length(test_estimate_log)
length(train_estimate_log)
contrasts(as.factor(y_train_log))
train_estimate_log <- ifelse(train_estimate_log < 0.5, 0, 1)
test_estimate_log <- ifelse(test_estimate_log < 0.5, 0, 1)
test_error_log <- 1 - sum(test_estimate_log == y_test_log) / length(y_test_log)
train_error_log <- 1 - sum(train_estimate_log == y_train_log) / length(y_train_log)
train_error_log
test_error_log
y_train_lm <- ifelse(y_train == 2, -1, 1)
y_test_lm <- ifelse(y_test == 2, -1, 1)
linear_model <- lm(y_train_lm ~ ., data = zip_train[, 2:257])
test_estimate_lm <- predict(linear_model, zip_test[, 2:257])
train_estimate_lm <- predict(linear_model, zip_train[, 2:257])
train_estimate_lm <- ifelse(train_estimate_lm < 0, -1, 1)
test_estimate_lm <- ifelse(test_estimate_lm < 0, -1, 1)
test_error_lm <- 1 - sum(test_estimate_lm == y_test_lm) / length(y_test_lm)
train_error_lm <- 1 - sum(train_estimate_lm == y_train_lm) / length(y_train_lm)
train_error_lm
test_error_lm
regression_results <- data.frame(name_method = numeric(), train_error = numeric(), test_error = numeric(), stringsAsFactors = FALSE)
regression_results <- rbind(
  regression_results,
  data.frame(
    name_method = "multinomial logistic regression",
    train_error = train_error_multil,
    test_error = test_error_multil,
    stringsAsFactors = FALSE
  )
)
regression_results <- rbind(
  regression_results,
  data.frame(
    name_method = "logistic regression",
    train_error = train_error_log,
    test_error = test_error_log,
    stringsAsFactors = FALSE
  )
)
regression_results <- rbind(
  regression_results,
  data.frame(
    name_method = "linear model",
    train_error = train_error_lm,
    test_error = test_error_lm,
    stringsAsFactors = FALSE
  )
)
x_train <- model.matrix(y_train_lm ~ ., data = zip_train[, 2:257])[, -1]
y_train = y_train_lm
x_test <- model.matrix(y_test_lm ~ ., data = zip_test[, 2:257])[, -1]
y_test <- y_test_lm
lasso.mod <- cv.glmnet(as.matrix(zip_train[, 2:257]), y_train_lm, alpha = 1)
best_lambda <- lasso.mod$lambda.min
lasso.coef <- predict(lasso.mod, type = "coefficients", s = best_lambda)
train_estimate_lm_lasso <- predict(lasso.mod, newx = as.matrix(zip_train[, 2:257]), s = best_lambda)
test_estimate_lm_lasso <- predict(lasso.mod, newx = as.matrix(zip_test[, 2:257]), s = best_lambda)
test_estimate_lm_lasso <- ifelse(test_estimate_lm_lasso < 0, -1, 1)
train_estimate_lm_lasso <- ifelse(train_estimate_lm_lasso < 0, -1, 1)
test_error_lm_lasso <- 1 - sum(as.numeric(unlist(test_estimate_lm_lasso)) == y_test_lm) / length(y_test_lm)
train_error_lm_lasso <- 1 - sum(as.numeric(unlist(train_estimate_lm_lasso)) == y_train_lm) / length(y_train_lm)
train_error_lm_lasso
test_error_lm_lasso
regression_results <- rbind(
  regression_results,
  data.frame(
    name_method = "lasso regression",
    train_error = train_error_lm_lasso,
    test_error = test_error_lm_lasso,
    stringsAsFactors = FALSE
  )
)
regression_results
knn_results

set.seed(5)
nasa_path <- paste0(getwd(), "/NASA.csv")
data <- read.csv(nasa_path)
attach(data)
# a)
for (i in 1:25) {
  if (distress[i] == "None") {
    distress[i] <- "No Incident"
  } else {
    distress[i] <- "Incident"
  }
}
distress <- as.factor(distress)
glm.fits <- glm(distress ~ date + temp, family = binomial)
summary(glm.fits)
contrasts(distress)
glm.probs <- predict(glm.fits, type = "response")
glm.pred <- rep("Incident", 25)
glm.pred[glm.probs > 0.5] <- "No Incident"
table(glm.pred, distress)
year <- as.Date(rep(0, 25), format = "%m/%d/%Y", origin = "1960-01-01")
for (i in 1:25) {
  year[i] <- as.Date(date[i], origin = "1960-01-01")
}
year <- substring(year, 0, 4)
train <- (year < 1986)
data.1986 <- data[!train, ]
dim(data.1986)
for (i in 1:length(distress.1986)) {
  if (distress.1986[i] == "None") {
    distress.1986[i] <- "No Incident"
  } else {
    distress.1986[i] <- "Incident"
  }
}
distress.1986 <- as.factor(distress.1986)
distress.1986
glm.fits2 <- glm(distress ~ date + temp, family = binomial, subset = train)
summary(glm.fits2)
glm.probs2 <- predict(glm.fits2, data.1986, type = "response")
contrasts(distress)
glm.pred2 <- rep("Incident", 2)
glm.pred2[glm.probs2 > 0.5] <- "No Incident"
table(glm.pred2, distress.1986)
lda.fit <- lda(distress ~ date + temp, subset = train)
lda.fit$prior
lda.fit$means
lda.fit$scaling
lda.pred <- predict(lda.fit, data.1986)
lda.class <- lda.pred$class
table(lda.class, distress.1986)
qda.fit <- qda(distress ~ date + temp, subset = train)
qda.fit$prior
qda.fit$means
qda.class <- predict(qda.fit, data.1986)$class
table(qda.class, distress.1986)
x <- rnorm(10000)
mean.boot <- function(x, ind) {
  mean(x[ind])
}
true_mean <- mean(x)
num_obs <- c(10, 100, 1000, 10000)
num_boot <- c(10, 100, 1000, 10000)
bias <- matrix(0, length(num_boot), length(num_obs))
colnames(bias) <- c("10", "100", "1000", "10000")
rownames(bias) <- c("10", "100", "1000", "10000")
for (i in 1:length(num_obs)) {
  for (j in 1:length(num_boot)) {
    obs <- num_obs[i]
    result <- boot(data = sample(x, obs), mean.boot, R = num_boot[j])
    bias[j, i] <- mean(result$t) - result$t0
  }
}
bias
size <- 1000
true_intercept <- 0
true_beta <- 2
x <- rnorm(size, sd = 10)
y <- true_intercept + true_beta * x + rnorm(size, sd = 10)
ols <- lm(y ~ x)
summary(ols)
ols.boot <- function(matrix, ind) {
  y <- matrix[, 1]
  x <- matrix[, 2]
  ols <- lm(y[ind] ~ x[ind])
  coef(ols)
}
result <- boot(data = cbind(y, x), ols.boot, R = size)
boot_intercept <- mean(result$t[, 1])
boot_slope <- mean(result$t[, 2])
bias_lm <- matrix(0, 2, 2)
colnames(bias_lm) <- c("OLS", "Boot")
rownames(bias_lm) <- c("Intercept", "Slope")
bias_lm[1, 1] = ols$coefficients[1] - true_intercept
bias_lm[1, 2] <- boot_intercept - true_intercept
bias_lm[2, 1] <- ols$coefficients[2] - true_beta
bias_lm[2, 2] <- boot_slope - true_beta
bias_lm
ols.boot_beta <- function(matrix, ind) {
  y <- matrix[, 1]
  x <- matrix[, 2]
  ols <- lm(y[ind] ~ x[ind])
  ols$coefficients[2]
}
result_2 <- boot(data = cbind(y, x), ols.boot_beta, R = size)
confidence_levels <- c(0.9, 0.95, 0.99)
boot.ci(
  conf = confidence_levels,
  boot.out = result_2,
  type = c(
    "all"
  )
)
plot(result_2, main = "Distribution of the bootstrap slope")
abline(v = ols$coefficients[2], col = "red")
abline(v = mean(result_2$t[, 1]), col = "orange")
abline(v = boot.ci(conf = 0.9, boot.out = result_2, type = c("all"))$basic[1], col = "blue")
abline(v = boot.ci(conf = 0.95, boot.out = result_2, type = c("all"))$basic[1], col = "green")
abline(v = boot.ci(conf = 0.99, boot.out = result_2, type = c("all"))$basic[1], col = "black")