credit.data <- read.csv("/Users/MeganEckstein/Documents/2019SpringSemester/DataMiningII/credit0.csv", header=T)
credit.data$X9 = NULL
credit.data$id = NULL
credit.data$Y = as.factor(credit.data$Y)
id_train <- sample(nrow(credit.data),nrow(credit.data)*0.75)
credit.train = credit.data[id_train,]
credit.test = credit.data[-id_train,]
creditcost <- function(observed, predicted){
  weight1 = 10
  weight0 = 1
  c1 = (observed==1)&(predicted == 0) #logical vector - true if actual 1 but predict 0
  c0 = (observed==0)&(predicted == 1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

#GAM
library(mgcv)

## Create a formula for a model with a large number of variables:
# s(X2)+ s(X3) + ... 
gam_formula <- as.formula(paste("Y~X2+X3+s(X4)+s(X5)+", paste(colnames(credit.train)[6:61], collapse= "+")))

credit.gam <- gam(formula = gam_formula, family=binomial,data=credit.train);
summary(credit.gam)

plot(credit.gam, shade=TRUE,seWithMean=TRUE,scale=0, pages = 1)

AIC(credit.gam)
BIC(credit.gam)
credit.gam$deviance

pcut.gam <- .08
prob.gam.in<-predict(credit.gam,credit.train,type="response")
pred.gam.in<-(prob.gam.in>=pcut.gam)*1
table(credit.train$Y,pred.gam.in,dnn=c("Observed","Predicted"))

mean(ifelse(credit.train$Y != pred.gam.in, 1, 0))
AIC(credit.gam)
BIC(credit.gam)

#define the searc grid from 0.01 to 0.20
searchgrid = seq(0.01, 0.20, 0.01)
#result.gam is a 99x2 matrix, the 1st col stores the cut-off p, the 2nd column stores the cost
result.gam = cbind(searchgrid, NA)
#in the cost function, both r and pi are vectors, r=Observed, pi=predicted probability
cost1 <- function(r, pi){
  weight1 = 10
  weight0 = 1
  c1 = (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

for(i in 1:length(searchgrid))
{
  pcut <- result.gam[i,1]
  #assign the cost to the 2nd col
  result.gam[i,2] <- cost1(credit.train$Y, predict(credit.gam,type="response"))
}
plot(result.gam, ylab="Cost in Training Set")

index.min<-which.min(result.gam[,2])#find the index of minimum value
result.gam[index.min,2] #min cost

result.gam[index.min,1] #optimal cutoff probability

pcut <-  result.gam[index.min,1] 
prob.gam.out<-predict(credit.gam,credit.test,type="response")
pred.gam.out<-(prob.gam.out>=pcut)*1
table(credit.test$Y,pred.gam.out,dnn=c("Observed","Predicted"))

mean(ifelse(credit.test$Y != pred.gam.out, 1, 0))

creditcost(credit.test$Y, pred.gam.out)

library(MASS)
credit.train$Y = as.factor(credit.train$Y)
credit.lda <- lda(Y~.,data=credit.train)
prob.lda.in <- predict(credit.lda,data=credit.train)
pcut.lda <- .15
pred.lda.in <- (prob.lda.in$posterior[,2]>=pcut.lda)*1
table(credit.train$Y,pred.lda.in,dnn=c("Obs","Pred"))

mean(ifelse(credit.train$Y != pred.lda.in, 1, 0))

lda.out <- predict(credit.lda,newdata=credit.test)
cut.lda <- .12
pred.lda.out <- as.numeric((lda.out$posterior[,2]>=cut.lda))
table(credit.test$Y,pred.lda.out,dnn=c("Obs","Pred"))

mean(ifelse(credit.test$Y != pred.lda.out, 1, 0))

creditcost(credit.test$Y, pred.lda.out)

library(MASS)
maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

scaled <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))
index <- sample(1:nrow(Boston),round(0.75*nrow(Boston)))

train_ <- scaled[index,]
test_ <- scaled[-index,]

library(neuralnet)

n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
# you can use this package for classification, just set linear.output = F
plot(nn)


pr.nn <- compute(nn,test_[,1:13])

pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
test.r <- (test_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
# MSE of testing set
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
MSE.nn

library(nnet)
credit.nnet <- nnet(Y~., data=credit.train, size=1, maxit=500)

prob.nnet= predict(credit.nnet,credit.test)
pred.nnet = as.numeric(prob.nnet > 0.08)
table(credit.test$Y,pred.nnet, dnn=c("Observed","Predicted"))

mean(ifelse(credit.test$Y != pred.nnet, 1, 0))

creditcost(credit.test$Y, pred.nnet)


library(e1071)

credit.svm = svm(Y ~ ., data = credit.train, cost = 1, gamma = 1/length(credit.train), probability= TRUE)
prob.svm = predict(credit.svm, credit.test, probability = TRUE)
prob.svm = attr(prob.svm, 'probabilities')[,2] #This is needed because prob.svm gives a matrix
pred.svm = as.numeric((prob.svm >= 0.08))
table(credit.test$Y,pred.svm,dnn=c("Obs","Pred"))

mean(ifelse(credit.test$Y != pred.svm, 1, 0))
creditcost(credit.test$Y, pred.svm)

data(iris)
id_train <- sample(nrow(iris),nrow(iris)*0.80)
iris.train = iris[id_train,]
iris.test = iris[-id_train,]
iris.svm = svm(Species ~ ., data = iris.train)
table(iris.test$Species, predict(iris.svm, iris.test), dnn=c("Observed","Predicted"))

library(caret) #this package contains the german data with its numeric format
data(GermanCredit)
