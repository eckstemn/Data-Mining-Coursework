set.seed(13023260)

library(MASS)
data(Boston)
index <- sample(nrow(Boston),nrow(Boston)*0.70)
bostonTrain <- Boston[index,]
bostonTest <- Boston[-index,]

# Linear Model
Boston1 <- lm(medv~., bostonTrain)
summary(Boston1)
bestmod <- step(Boston1)
summary(bestmod)
pred.train <- predict(bestmod)
MSE.lm <- mean((pred.train-bostonTrain$medv)^2)
MSE.lm
pred.test <- predict(bestmod, bostonTest)
MSPE.lm <- mean((pred.test-bostonTest$medv)^2)
MSPE.lm

# Regression Tree
library(rpart)
boston.tree<- rpart(bestmod, data = bostonTrain)
plot(boston.tree, uniform=TRUE, 
     main="Regression Tree")
text(boston.tree, use.n=TRUE, all=TRUE, cex=.7)
boston.tree.train.pred <- predict(boston.tree)
boston.tree.pred<- predict(boston.tree, newdata = bostonTest)
MSE.tree <- mean((bostonTrain$medv - boston.tree.train.pred)^2)
MSE.tree
MSPE.tree <- mean((bostonTest$medv-boston.tree.pred)^2)
MSPE.tree

# Bagging
library(ipred)

boston.bag<- bagging(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat, 
                     data = bostonTrain, nbagg=100)
boston.bag
boston.bag.pred.train <- predict(boston.bag, newdata = bostonTrain)
in_samp_err <- mean((bostonTrain$medv-boston.bag.pred.train)^2)
in_samp_err
boston.bag.pred.test <- predict(boston.bag, newdata = bostonTest)
out_samp_err <- mean((bostonTest$medv-boston.bag.pred.test)^2)
out_samp_err

# OOB
boston.bag.oob<- bagging(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat,
                         data = bostonTrain, coob=T, nbagg=100)
boston.bag.oob



# RANDOM FORESTS
library(randomForest)
boston.rf<- randomForest(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat, 
                         data = bostonTrain, importance=TRUE, mtry = sqrt(11))
boston.rf
boston.rf$importance

boston.rf.pred.train<- predict(boston.rf, bostonTrain)
RF.MSE <- mean((bostonTrain$medv-boston.rf.pred.train)^2)
RF.MSE
boston.rf.pred.test <- predict(boston.rf, bostonTest)
RF.MSPE <- mean((bostonTest$medv-boston.rf.pred.test)^2)
RF.MSPE


oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~.,
                     data = BostonTest, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((BostonTest$medv-predict(fit, BostonTest))^2)
  cat(i, " ")
}
matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))


# BOOSTING TREE
library(gbm)
boston.boost<- gbm(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat, data = bostonTrain, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)

# prediction on training
boston.boost.pred.train<- predict(boston.boost, bostonTrain, n.trees = 10000)
ise <- mean((bostonTrain$medv-boston.boost.pred.train)^2)
ise
# prediction on testing
boston.boost.pred.test<- predict(boston.boost, bostonTest, n.trees = 10000)
ose <- mean((bostonTest$medv-boston.boost.pred.test)^2)
ose
