library(MASS) #this data is in MASS package
boston.data <- data(Boston)
sample_index <- sample(nrow(Boston),nrow(Boston)*0.90)
boston.train <- Boston[sample_index,]
boston.test <- Boston[-sample_index,]

library(rpart)
library(rpart.plot)

boston.rpart <- rpart(formula = medv ~ ., data = boston.train)
boston.rpart

prp(boston.rpart,digits = 4, extra = 1)

boston.train[10,]

boston.train.pred.tree = predict(boston.rpart)
head(boston.train.pred.tree)
boston.test.pred.tree = predict(boston.rpart,boston.test)
head(boston.test.pred.tree)

MSE.tree<- mean((boston.train.pred.tree - boston.train$medv)^2)
MSPE.tree <- mean((boston.test.pred.tree - boston.test$medv)^2)
  
boston.reg = lm(medv~., data = boston.train)
boston.test.pred.reg = predict(boston.reg, boston.test)
mean((boston.test.pred.reg - boston.test$medv)^2)

boston.lm<- lm(medv~., data = boston.train)
boston.train.pred.lm<- predict(boston.lm)
boston.test.pred.lm<- predict(boston.lm, boston.test)
MSE.lm<- mean((boston.train.pred.lm - boston.train$medv)^2)
MSPE.lm<- mean((boston.test.pred.lm - boston.test$medv)^2)
MSE.lm; MSPE.lm

credit.data <- read.csv(file = "/Users/MeganEckstein/Documents/2019SpringSemester/DataMining/credit_default.csv", 
                        header=T)
# rename
library(dplyr)
credit.data<- rename(credit.data, default=default.payment.next.month)
# convert categorical data to factor
credit.data$SEX<- as.factor(credit.data$SEX)
credit.data$EDUCATION<- as.factor(credit.data$EDUCATION)
credit.data$MARRIAGE<- as.factor(credit.data$MARRIAGE)

index <- sample(nrow(credit.data),nrow(credit.data)*0.80)
credit.train = credit.data[index,]
credit.test = credit.data[-index,]

credit.rpart0 <- rpart(formula = default ~ ., data = credit.train, method = "class")
credit.rpart0
credit.rpart <- rpart(formula = default ~ . , data = credit.train, method = "class", parms = list(loss=matrix(c(0,10,1,0), nrow = 2)))

# Prediction
pred0<- predict(credit.rpart0, type="class")
head(predict(credit.rpart0, type="prob"))
pred_ind <- (predict(credit.rpart0, type="prob")[,2]) >= 0.5
table(pred_ind, credit.train$default)

table(credit.train$default, pred0, dnn = c("True", "Pred"))

credit.rpart <- rpart(formula = default ~ . , data = credit.train, method = "class", parms = list(loss=matrix(c(0,5,1,0), nrow = 2)))

credit.rpart
prp(credit.rpart, extra = 1)

library(ROCR)
credit.rpart <- rpart(formula = default ~ ., data = credit.train, method = "class", parms = list(loss=matrix(c(0,10,1,0), nrow = 2)))
#Probability of getting 1
credit.test.prob.rpart = predict(credit.rpart,credit.test, type="prob")
pred = prediction(credit.test.prob.rpart[,2], credit.test$default)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]

credit.test.pred.rpart = as.numeric(credit.test.prob.rpart[,2] > 1/11)
table(credit.test$default, credit.test.pred.rpart, dnn=c("Truth","Predicted"))


cost <- function(r, pi){
  weight1 = 5
  weight0 = 1
  c1 = (r==1)&(pi==0) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi==1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}



#Fit logistic regression model
credit.glm<- glm(default~., data = credit.train, family=binomial)
#Get binary prediction
credit.test.pred.glm<- as.numeric(predict(credit.glm, credit.test, type="response")>0.21)
#Calculate cost using test set
cost(credit.test$default,credit.test.pred.glm)
#Confusion matrix
table(credit.test$default, credit.test.pred.glm, dnn=c("Truth","Predicted"))

# Prune 
prp(prune(credit.rpart, cp = 0.001))
plotcp(credit.rpart)

