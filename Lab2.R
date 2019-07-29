data(iris)
iris
head(iris)
dim(iris)
summary(iris)
ncol(iris)
nrow(iris)
names(iris)
str(iris)
colnames(iris)
class(iris[,1])
class(iris[,5])
table(iris$Species)
setosa <- subset(iris, Species=="setosa")

# group mean
aggregate(.~Species, iris, mean)

# group standard deviation
aggregate(.~Species, iris, sd)

train_id <- sample(x = nrow(iris), size = nrow(iris)*0.80)
test_id <- c(1:nrow(iris))[! c(1:nrow(iris)) %in% train_id]
iris_train <- iris[train_id,]
iris_test <- iris[test_id,]
summary(iris_train$Sepal.Length)

library(class)

iris_knn <- knn(train = iris_train[,-5], test = iris_test[,-5], cl = iris_train[,5])
iris_knn
table(iris_knn, iris_test[,5], dnn = c("Predict", "True"))

# Clustering analysis
## k means clustering
library(fpc)
aggregate(.~Species, data = iris, mean)
fit <- kmeans(x = iris[,1:4], centers = 5)
plotcluster(iris[,1:4], fit$cluster)
fit$centers
