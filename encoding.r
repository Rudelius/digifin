library(dplyr)
library(tree)
library(randomForest)
library(utf8)
set.seed(1)

# Plot two plots at a time.
par(mfrow= c(1,2))

# Clean environment.
rm(list = ls())

# Import loandata.
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
plot(cv.loandata$size, cv.loandata$dev, type = "b")
plot(cv.loandata$k, cv.loandata$dev, type = "b")

# Prune the tree to remove nodes that do not improve prediction.
tree.loandata.pruned <- prune.tree(tree.loandata, best = 3)

# Plot trees.
plot(tree.loandata)
text(tree.loandata, pretty = 0)
title(main = "Unpruned")
plot(tree.loandata.pruned)
text(tree.loandata.pruned, pretty = 0)
title(main = "Pruned")



####################################### Tree evaluating part ######################################

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
colnames(stats) <- c('unpruned', 'pruned')
rownames(stats) <- c('meanError', 'correctPredictions')
as.table(stats)



#################################### Bagging part ######################################

# Make a bagging model.
bagg.loandata = randomForest(x = loandata.train[,1:25], y = loandata.train$newpayingremark, 
                            subset = trainingRows, mtry = length(loandata.train)-1,
                            importance = TRUE, ntree = 10)

# Necessary fixing of NA values for random forest no work, same as for training data in the tree model.
loandata.test <- loandata.test[!is.na(loandata.test$lastucrequest),]
loandata.test <- na.tree.replace(loandata.test)

# Evaluate Bagging model.
pred.bagg.loandata <- predict(bagg.loandata, newdata = loandata.test)

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
colnames(stats) <- c('unpruned', 'pruned', 'bagging')
as.table(stats)


