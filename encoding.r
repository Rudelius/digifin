# Dependencies: install.packages("dplyr", "utf8", "tree", "randomForest", "gbm", "keras")
# Install keras with: devtools::install_github("rstudio/keras")
# Also: install_keras()
library(dplyr)
library(utf8)
library(tree)
library(randomForest)
library(gbm)
library(keras)
library(reticulate)
use_python("/usr/bin/python")
#set.seed(1)

# Plot two plots at a time.
par(mfrow= c(1,2))

# Clean environment.
#rm(list = ls())

# Import unmodified loandata. 
loandata <- read.csv(file="/Users/johan/Dropbox/Handels/digifin/data/loandata.csv", header = TRUE, sep=",")

# Import prepared municipality -> region dataset.
# https://www.dropbox.com/s/04puwtchxprr1w3/lan.csv?dl=0
municipalities <- read.csv(file="/Users/johan/Dropbox/Handels/digifin/data/lan.csv", header = TRUE, sep=";")



######################################## Data cleaning part #######################################

# Reformat dates into year numerics.
loandata$creationdateuserloan = 2000+(unclass(as.Date(loandata$creationdateuserloan))-unclass(as.Date("2000/01/01")))/365 #creationdateuserloan
loandata$lastucrequest = 2000+(unclass(as.Date(loandata$lastucrequest))-unclass(as.Date("2000/01/01")))/365 #lastucrequest

# Replace "None", "NA" and empty string with NA. Requires dplyr package.
for (i in 1:length(loandata)){
  loandata[,i] <- na_if(loandata[,i], "None")
  loandata[,i] <- na_if(loandata[,i], "")
  loandata[,i] <- na_if(loandata[,i], "NA")
}

# Clean faulty jobtype data.
x <- loandata$jobtype
x <- replace(x, x == "Agricul", "Agriculture")
x <- replace(x, x == "Custome", "CustomerServ")
x <- replace(x, x == "Economi", "Economics")
x <- replace(x, x == "Educati", "Educational")
x <- replace(x, x == "ImportE", "ImportExport")
x <- replace(x, x == "Logisti", "Logistics")
x <- replace(x, x == "Managem", "Management")
x <- replace(x, x == "Product", "Production")
x <- replace(x, x == "Psychol", "Psychology")
x <- replace(x, x == "Publish", "Publishing")
x <- replace(x, x == "Realest", "Realestate")
x <- replace(x, x == "Researc", "Research")
x <- replace(x, x == "Securit", "Security")
loandata$jobtype <- factor(x)

# Clean faulty jobstatus data.
x <- loandata$jobstatus
x <- replace(x, x == "TemporaryEmployme", "TemporaryEmployment")
loandata$jobstatus <- factor(x)

# Fix encoding of Swedish special characters for municipality.
a = as.character(loandata$municipality)
Encoding(a) <- "latin1"
loandata$municipality <- as.factor(utf8_encode(a))

# VLOOKUP municipality -> region to reduce number of geographical data factor levels.
loandata <- (merge(municipalities, loandata, by = 'municipality'))

# Optional to save clean data to new csv file.
#write.csv(loandata, file = "/Users/johan/Dropbox/Handels/digifin/data/loandata_clean.csv")

# Drop unwanted attributes.
loandata$key <- NULL
loandata$addresstype <- NULL
loandata$lastaddresschange <- NULL
loandata$municipality <- NULL



######################################## Data splitting and final cleaning part #######################################

# Create a vector to randomly split the dataset into training and test, ratio 8:2.
trainingRows <- sample(1:nrow(loandata), 0.8*length(loandata[,1]))

# Split loandata into training and test sets.
loandata.train <- loandata[trainingRows, ]
loandata.test <- loandata[-trainingRows, ]

# Handle NA values in training set; remove rows with NA as lastucrequest, and create a NA factor level for categorical variables.
loandata.train <- loandata.train[!is.na(loandata.train$lastucrequest),]
loandata.train <- na.tree.replace(loandata.train)



####################################### Tree part ######################################

# Generate a tree on the training data.
tree.loandata <- tree(formula = newpayingremark ~ .-newpayingremark, data = loandata, subset = trainingRows)

# Cross validation testing.
cv.loandata <- cv.tree(tree.loandata)

# Print results.
cv.loandata 

# Plot cv results.
plot(cv.loandata$size, cv.loandata$dev, type = "b", main = "CV deviation depending on tree size")
plot(cv.loandata$k, cv.loandata$dev, type = "b", main = "CV deviation depending on k")

# Prune the tree to remove nodes that do not improve prediction.
tree.loandata.pruned <- prune.tree(tree.loandata, best = 3)

# Plot trees.
plot(tree.loandata)
text(tree.loandata, pretty = 0)
title(main = "Unpruned")
plot(tree.loandata.pruned)
text(tree.loandata.pruned, pretty = 0)
title(main = "Pruned")

# Calculate percentage of correct predictions for unpruned tree.
pred.tree.loandata <- predict(tree.loandata, loandata.test, type = "vector")
correctPredictions = 0
for (i in 1:length(pred.tree.loandata)){
  if(abs(pred.tree.loandata[i]-loandata.test$newpayingremark[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  } 
}
correctPredictions.tree.loandata <- correctPredictions / length(loandata.test[,1])

# Calculate the mean error of predictions for unpruned tree.
meanError.tree.loandata <- mean(abs(pred.tree.loandata-loandata.test$newpayingremark))

# Calculate percentage of correct predictions for pruned tree.
pred.tree.loandata.pruned <- predict(tree.loandata.pruned, loandata.test, type = "vector")
correctPredictions = 0
for (i in 1:length(pred.tree.loandata.pruned)){
  if(abs(pred.tree.loandata.pruned[i]-loandata.test$newpayingremark[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  } 
}
correctPredictions.tree.loandata.pruned <- correctPredictions / length(loandata.test[,1])

# Calculate the mean error of predictions for pruned tree.
meanError.tree.loandata.pruned <- mean(abs(pred.tree.loandata.pruned-loandata.test$newpayingremark))

# Create statistics table for unpruned and pruned tree.
stats <- matrix(c(meanError.tree.loandata, correctPredictions.tree.loandata,
                  meanError.tree.loandata.pruned, correctPredictions.tree.loandata.pruned)
                ,ncol = 2)
colnames(stats) <- c('unpruned tree', 'pruned tree')
rownames(stats) <- c('meanError', 'correctPredictions')
as.table(stats)



#################################### Bagging part ######################################

# Necessary fixing of NA values for random forest to work, drop lastucrequest, and add NA factor level.
loandata.test$lastucrequest <- NULL
loandata.train$lastucrequest <- NULL
loandata.test <- na.tree.replace(loandata.test)

# Make a bagging model.
bagg.loandata = randomForest(x = loandata.train[,names(loandata.train) != "newpayingremark"], y = loandata.train$newpayingremark,
                             mtry = length(loandata.train)-1, importance = TRUE, ntree = 100)

# Plot the OOB error to number of trees.
par(mfrow= c(1,1))
plot(bagg.loandata, main = "Bagging OOB error to number of trees")
par(mfrow= c(1,2))

# Calculate percentage of correct predictions for bagging model.
pred.bagg.loandata <- predict(bagg.loandata, newdata = loandata.test)
correctPredictions = 0
for (i in 1:length(pred.bagg.loandata)){
  if(abs(pred.bagg.loandata[i]-loandata.test$newpayingremark[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  } 
}
correctPredictions.bagg.loandata <- correctPredictions / length(loandata.test[,1])

# Calculate the mean error of predictions for bagging model.
meanError.bagg.loandata <- mean(abs(pred.bagg.loandata-loandata.test$newpayingremark))

# Add the new statistics to the stats table.
tempMatrix <- matrix(c(meanError.bagg.loandata, correctPredictions.bagg.loandata))
stats <- cbind(stats, tempMatrix)
colnames(stats) <- c('unpruned tree', 'pruned tree', 'bagging')
as.table(stats)

# Plot relative importance of input features.
varImpPlot (bagg.loandata )



#################################### Random forest part ######################################

# Make a random forest model.
randomForest.loandata <- randomForest(x = loandata.train[,names(loandata.train) != "newpayingremark"],
                                      y = loandata.train$newpayingremark, importance = TRUE, ntree = 100)

# Plot the OOB error to number of trees.
par(mfrow= c(1,1))
plot(randomForest.loandata, main = "Random forest OOB error to number of trees")
par(mfrow= c(1,2))

# Calculate percentage of correct predictions for random forest model.
pred.randomForest.loandata <- predict(randomForest.loandata, newdata = loandata.test)
correctPredictions = 0
for (i in 1:length(pred.randomForest.loandata)){
  if(abs(pred.randomForest.loandata[i]-loandata.test$newpayingremark[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  } 
}
correctPredictions.randomForest.loandata <- correctPredictions / length(loandata.test[,1])

# Calculate the mean error of predictions for bagging model.
meanError.randomForest.loandata <- mean(abs(pred.randomForest.loandata-loandata.test$newpayingremark))

# Add the new statistics to the stats table.
tempMatrix <- matrix(c(meanError.randomForest.loandata, correctPredictions.randomForest.loandata) , ncol = 1)
stats <- cbind(stats, tempMatrix)
colnames(stats) <- c('unpruned tree', 'pruned tree', 'bagging', 'randomForest')
as.table(stats)

# Plot relative importance of input features.
varImpPlot (randomForest.loandata )


#################################### Boosting part ######################################

boost.loandata = gbm(newpayingremark~ ., data = loandata.train, cv.folds = 10,
                 n.trees = 100, interaction.depth = 3, distribution = "gaussian")
 
summary(boost.loandata)

par(mfrow=c(1,2))
plot(boost.loandata ,i="noinquiries")
plot(boost.loandata ,i="ucriskperson")
plot(boost.loandata ,i="creationdateuserloan")

# Calculate percentage of correct predictions for boosting model.
pred.boost.loandata <- predict(boost.loandata, newdata = loandata.test, n.trees = 85)
correctPredictions = 0
for (i in 1:length(pred.boost.loandata)){
  if(abs(pred.boost.loandata[i]-loandata.test$newpayingremark[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  } 
}
correctPredictions.boost.loandata <- correctPredictions / length(loandata.test[,1])

# Calculate the mean error of predictions for boosting model.
meanError.boost.loandata <- mean(abs(pred.boost.loandata-loandata.test$newpayingremark))

# Add the new statistics to the stats table.
tempMatrix <- matrix(c(meanError.boost.loandata, correctPredictions.boost.loandata), ncol = 1)
stats <- cbind(stats, tempMatrix)
colnames(stats) <- c('unpruned tree', 'pruned tree', 'bagging', 'randomForest', 'boosting')

# Print info on boost model.
boost.loandata

# Print stats table.
as.table(stats)

######################################## Data splitting and final cleaning part #######################################

loandata$lastucrequest <- NULL
loandata <- na.tree.replace(loandata)

# Replace region data with one hot encoding.
loandata <- cbind(to_categorical(as.numeric(loandata$lan)), loandata)
loandata$lan <- NULL

# Replace category data with one hot encoding.
loandata <- cbind(to_categorical(as.numeric(loandata$category)), loandata)
loandata$category <- NULL

# Replace gender data with one hot encoding.
loandata <- cbind(to_categorical(as.numeric(loandata$gender)), loandata)
loandata$gender <- NULL

# Replace gender data with one hot encoding.
loandata <- cbind(to_categorical(as.numeric(loandata$maritalstatus)), loandata)
loandata$maritalstatus <- NULL

# Replace jobstatus data with one hot encoding.
loandata <- cbind(to_categorical(as.numeric(loandata$jobstatus)), loandata)
loandata$jobstatus <- NULL

# Replace jobtype data with one hot encoding.
loandata <- cbind(to_categorical(as.numeric(loandata$jobtype)), loandata)
loandata$jobtype <- NULL

# Replace livingstatus data with one hot encoding.
loandata <- cbind(to_categorical(as.numeric(loandata$livingstatus)), loandata)
loandata$livingstatus <- NULL

# Standardise data.
loandata.scaled <- scale(loandata)

# Split loandata into training and test sets.
x.train <- as.matrix( loandata[trainingRows, 1:114] )
y.train <- as.matrix( loandata$newpayingremark[trainingRows] )

x.test <- as.matrix( loandata[-trainingRows, 1:114] )
y.test <- as.matrix( loandata$newpayingremark[-trainingRows] )



######################################## Neural network part #######################################

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 3000, activation = 'sigmoid', kernel_initializer = 'normal') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1500, activation = 'relu', kernel_initializer = 'normal') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 750, activation = 'relu', kernel_initializer = 'normal') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 375, activation = 'relu', kernel_initializer = 'normal') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'linear', kernel_initializer = 'normal')

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_rmsprop()
)

history <- model %>% fit(
  x.train, y.train, 
  epochs = 30,
  batch_size = 128, 
  validation_split = 0.1
)
plot(history)
model %>% evaluate(x.test, y.test)

pred.nn.loandata <- model %>% predict(x.test)

# Plot distributions from different models.
plot(pred.bagg.loandata)
plot(pred.randomForest.loandata)
plot(pred.boost.loandata)
plot(pred.nn.loandata)

# Calculate percentage of correct predictions for boosting model.
correctPredictions = 0
for (i in 1:length(pred.nn.loandata)){
  if(abs(pred.nn.loandata[i]- y.test[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  }
}
correctPredictions.nn.loandata <- correctPredictions / length(y.test)

meanError.nn.loandata <- mean(abs(pred.nn.loandata-y.test))

# Add the new statistics to the stats table.
tempMatrix <- matrix(c(meanError.nn.loandata, correctPredictions.nn.loandata), ncol = 1)
stats <- stats[,-6]
stats <- cbind(stats, tempMatrix)
colnames(stats) <- c('unpruned tree', 'pruned tree', 'bagging', 'randomForest', 'boosting', 'neural net')

# Print stats table.
as.table(stats)

