#1. Import Data
votes <- read.table("house-votes.txt", sep = ",", header = FALSE, stringsAsFactors = FALSE)

#2. Prepare Data
colnames(votes) <- c("class_name", "handicapped_infants", "water_project_cost_sharing", "adoption_of_the_budget_resolution", "physician_fee_freeze", "el_salvador_aid", "religious_groups_in_schools", "anti_satellite_test_ban", "aid_to_nicaraguan_contras", "mx_missle", "immigration", "synfuels_coporation_cutback", "education_spending", "superfund_right_to_sue", "crime", "duty_free_exports", "export_administration_act_south_africa")
votes <- as.data.frame(lapply(votes,function(x){
gsub("\\?",NA,x)
}))
str(votes)

#Impute values...
library(gmodels)
#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$handicapped_infants,votes$class_name)
votes$handicapped_infants <- ifelse(is.na(votes$handicapped_infants),
ifelse(votes$class_name == "democrat",2,1),votes$handicapped_infants)

#Both parties voted yes more than no
CrossTable(votes$water_project_cost_sharing ,votes$class_name)
votes$water_project_cost_sharing <- ifelse(is.na(votes$water_project_cost_sharing),
2,votes$water_project_cost_sharing )

#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$adoption_of_the_budget_resolution,votes$class_name)
votes$adoption_of_the_budget_resolution <- ifelse(is.na(votes$adoption_of_the_budget_resolution),
ifelse(votes$class_name == "democrat",2,1),votes$adoption_of_the_budget_resolution)

#Republicans largely voted yes, democrats largely voted no
CrossTable(votes$physician_fee_freeze,votes$class_name)
votes$physician_fee_freeze <- ifelse(is.na(votes$physician_fee_freeze),
ifelse(votes$class_name == "democrat",1,2),votes$physician_fee_freeze)

#Republicans largely voted yes, democrats largely voted no
CrossTable(votes$el_salvador_aid ,votes$class_name)
votes$el_salvador_aid  <- ifelse(is.na(votes$el_salvador_aid ),
ifelse(votes$class_name == "democrat",1,2),votes$el_salvador_aid )

#Republicans largely voted yes, democrats largely voted no
CrossTable(votes$religious_groups_in_schools ,votes$class_name)
votes$religious_groups_in_schools  <- ifelse(is.na(votes$religious_groups_in_schools ),
ifelse(votes$class_name == "democrat",1,2),votes$religious_groups_in_schools )

#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$anti_satellite_test_ban,votes$class_name)
votes$anti_satellite_test_ban <- ifelse(is.na(votes$anti_satellite_test_ban),
ifelse(votes$class_name == "democrat",2,1),votes$anti_satellite_test_ban)

#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$aid_to_nicaraguan_contras,votes$class_name)
votes$aid_to_nicaraguan_contras <- ifelse(is.na(votes$aid_to_nicaraguan_contras),
ifelse(votes$class_name == "democrat",2,1),votes$aid_to_nicaraguan_contras)

#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$mx_missle ,votes$class_name)
votes$mx_missle  <- ifelse(is.na(votes$mx_missle ),
ifelse(votes$class_name == "democrat",2,1),votes$mx_missle )

#Republicans largely voted yes, democrats largely voted no
CrossTable(votes$immigration,votes$class_name)
votes$immigration <- ifelse(is.na(votes$immigration),
ifelse(votes$class_name == "democrat",1,2),votes$immigration)

#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$synfuels_coporation_cutback,votes$class_name)
votes$synfuels_coporation_cutback <- ifelse(is.na(votes$synfuels_coporation_cutback),
ifelse(votes$class_name == "democrat",2,1),votes$synfuels_coporation_cutback)

#Republicans largely voted yes, democrats largely voted no
CrossTable(votes$education_spending,votes$class_name)
votes$education_spending <- ifelse(is.na(votes$education_spending),
ifelse(votes$class_name == "democrat",2,1),votes$education_spending)

#Republicans largely voted yes, democrats largely voted no
CrossTable(votes$superfund_right_to_sue,votes$class_name)
votes$superfund_right_to_sue <- ifelse(is.na(votes$superfund_right_to_sue),
ifelse(votes$class_name == "democrat",1,2),votes$superfund_right_to_sue)

#Republicans largely voted yes, democrats largely voted no
CrossTable(votes$crime,votes$class_name)
votes$crime <- ifelse(is.na(votes$crime),
ifelse(votes$class_name == "democrat",1,2),votes$crime)

#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$duty_free_exports,votes$class_name)
votes$duty_free_exports <- ifelse(is.na(votes$duty_free_exports),
ifelse(votes$class_name == "democrat",2,1),votes$duty_free_exports)

#Republicans largely voted no, democrats largely voted yes
CrossTable(votes$export_administration_act_south_africa,votes$class_name)
votes$export_administration_act_south_africa <- ifelse(is.na(votes$export_administration_act_south_africa),
ifelse(votes$class_name == "democrat",2,1),votes$export_administration_act_south_africa)

#################### Model 1: Decision Tree ####################

#3a. Begin decision tree model
set.seed(123)
train_sample <- sample(435, 348)
votes_dt_train <- votes[train_sample,]
votes_dt_test <- votes[-train_sample,]
#For future use with SVM
votes_svm_train <- votes_dt_train
votes_svm_test <- votes_dt_test
prop.table(table(votes$class_name))
prop.table(table(votes_dt_train$class_name))
prop.table(table(votes_dt_test$class_name))
library(C50)
votes_dt_model <- C5.0(votes_dt_train[-1],votes_dt_train$class_name)
summary(votes_dt_model)
votes_pred <- predict(votes_dt_model,votes_dt_test[-1])
library(gmodels)
CrossTable(votes_pred,votes_dt_test$class_name)

#4a. Evaluate future performance via 10-fold CV
library(caret)
library(irr)
set.seed(123)
folds <- createFolds(votes$class_name)
cv_results <- lapply(folds,function(x){
votes_train <- votes[-x,]
votes_test <- votes[x,]
votes_dt_cv_model <- C5.0(class_name ~ ., data = votes_train)
votes_pred <- predict(votes_dt_cv_model, votes_test)
votes_actual <- votes_test$class_name
kappa <- kappa2(data.frame(votes_actual,votes_pred))$value
return(kappa)
})
str(cv_results)
#Very Good Agreement - kappa = .938
mean(unlist(cv_results))

#5a.  Perform automated parameter tuning
grid <- expand.grid(.model = "tree",
.trials = c(1,5,10,15,20,25,30,35),
.winnow = c("TRUE","FALSE")
)
ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
set.seed(123)
votes_caret_dt_model <- train(class_name ~ ., data = votes, method = "C5.0",
	metric = "Kappa",
	trControl = ctrl,
	tuneGrid = grid)
#trials = 15 produces the best model...
votes_caret_dt_model

#6a.  Try to improve with ensemble learning
library(ipred)
set.seed(123)
votes_dt_bag_model <- bagging(class_name ~ .,data = votes, nbagg = 25)
votes_dt_bag_pred <- predict(votes_dt_bag_model, votes)
#Only 1 error
table(votes_dt_bag_pred, votes$class_name)
# check to see how it will generalize...
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)
train(class_name ~ ., data = votes, method = "treebag", trControl = ctrl)

######################### Model 2: SVM #########################

#3b. Begin Support Vector Machine model
library(kernlab)
votes_svm_model <- ksvm(class_name ~ ., data = votes_svm_train, kernel = "vanilladot")
votes_svm_model
votes_svm_pred <- predict(votes_svm_model,votes_svm_test)
#6.9% error rate...
CrossTable(votes_svm_pred,votes_svm_test$class_name)

#4b. Evaluate future performance via 10-fold CV
library(caret)
library(irr)
set.seed(123)
folds <- createFolds(votes$class_name)
cv_results <- lapply(folds,function(x){
votes_train <- votes[-x,]
votes_test <- votes[x,]
votes_svm_cv_model <- ksvm(class_name ~ ., data = votes_train, kernel = "vanilladot")
votes_pred <- predict(votes_svm_cv_model, votes_test)
votes_actual <- votes_test$class_name
kappa <- kappa2(data.frame(votes_actual,votes_pred))$value
return(kappa)
})
str(cv_results)
#Very Good Agreement - kappa = .937
mean(unlist(cv_results))

#5b. Perform automated parameter tuning
grid <- expand.grid(.C = c(0,1,2,3,5,10,15,20))
ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
set.seed(123)
votes_caret_svm_model <- train(class_name ~ ., data = votes, method = "svmLinear",
	metric = "Kappa",
	trControl = ctrl,
	tuneGrid = grid)
#cost = 1 produces the best model...
votes_caret_svm_model

#6b. Try to improve with ensemble learning
bagctrl <- bagControl(fit = svmBag$fit,
predict = svmBag$pred,
aggregate = svmBag$aggregate)
ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
svmBag$pred
predfunct <- function (object, x) 
{
    if (is.character(lev(object))) {
        out <- predict(object, as.matrix(x), type = "probabilities")
        colnames(out) <- lev(object)
        rownames(out) <- NULL
    }
    else out <- predict(object, as.matrix(x))[, 1]
    out
}
bagctrl <- bagControl(fit = svmBag$fit,
		#add new variable...
		predict = predfunct,
		aggregate = svmBag$aggregate)
ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
set.seed(123)
votes_svm_bag_model <- train(class_name ~ ., data = votes, "bag", trControl = ctrl, bagControl = bagctrl)
#Kappa = .938
votes_svm_bag_model