#Setting working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#remove any data in the environment
#rm(list=ls())

#load necessary libraries
library(raster)
library(dismo)
library(rgeos)
library(rgdal)
library(sp)
library(sf)
library(maptools)
library(caret)
library(ggplot2)
library(GGally)
library(rJava)
#library(corrplot)

#To install maxent, uncomment the next line of code
#if( !file.exists(paste0(system.file("java", package = "dismo"), "/maxent.jar"))) {
#  utils::download.file(url="https://raw.githubusercontent.com/mrmaxent/Maxent/master/ArchivedReleases/3.3.3k/maxent.jar",
#                       destfile = paste0(system.file("java", package = "dismo"), "/maxent.jar"),
#                       mode = "wb") #wb for binary file, otherwise maxent,jar cannot execute
#}

##### PART A - CURRENT PREDICTIONS #####

#load all the predictors and stack 
#Final Predictors: Bio1, Bio3, bio7, bio8, bio9, bio12, bio13, bio14, elevation, EVT, GHM, Imperv, LC, NDVI, Sagebrush
predic_list<- list.files("PredictorsFinal", pattern=".tif$", full.names = T)
predictors <- raster::stack(predic_list)

#load presence/absence shapefile
sage <- read_sf("Sage_Final/sage_final.shp")

sage$ID <- seq.int(nrow(sage)) #give ID to the sage data

training <- raster::extract(predictors, sage, df=TRUE) #extract the rasters and sage into a df
pder <- merge(training, sage, by.x="ID", by.y="ID") #merge the columns

#remove extra columns not needed (leave only rasters and class_pa)
pder[,c("ID","species", "eventDt", "day", "month", "year", "issue", "season", "geometry")] <- NULL 

anyNA(pder)
pder_clean <- na.omit(pder) #omit rows with NA values in them - necessary for the models
anyNA(pder_clean)
summary(pder_clean$class_pa==1) #how many presences and pseudo-absences we have

#### plots of  variables ####
# Store X and y
x = pder_clean[, 1:15]
y = as.factor(pder_clean$class_pa)

# importance of variables using featurePlot()
featurePlot(x,
            y,
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")))

# density plots for each attribute by class value
featurePlot(x,
            y,
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")))


# pairs plot
pairs(pder_clean[,1:15], cex=0.1, fig=TRUE)

ggpairs(pder_clean[1:15], ggplot2::aes(colour = as.factor(pder_clean$class_pa), alpha = 0.4))

##### preparation for Models #####
#part the data into train and test
set.seed(100)
trainids <- createDataPartition(pder_clean$class_pa,list=FALSE,p=0.7) 
trainData <- pder_clean[trainids,] #70%
testData <- pder_clean[-trainids,] #30%

#Maxent cannot use the p/a as factor, so we transform only for the other models
trainfactor<-trainData
trainfactor$class_pa <- as.factor(trainfactor$class_pa)
testfactor<-testData
testfactor$class_pa <- as.factor(testfactor$class_pa)

#### Models ####

### Regular parameters ###
cross_validation <- trainControl(method="cv",search="random", number=10)
metric <- "Accuracy"
Number_of_tune_tries <- 15
preprocessed <- c("scale", "center")

####### SVM Model #######
set.seed(123) #for reproducibility
svmmodel <- caret::train(class_pa~.,data=trainfactor,
                         method = "svmRadial",
                         metric=metric,
                         tuneLength= Number_of_tune_tries,
                         trControl=cross_validation,
                         preProcess=preprocessed)

svmmodel

svmprediction <- predict(predictors, svmmodel, filename="pred_svm", progress ="text", overwrite=TRUE)
svmprediction
plot(svmmodel)
plot(svmprediction, main="SVM - caret package")
plot(varImp(svmmodel), main="Importance of Each Variable") #importance of each variable

#trying the model on the test data.
svm_pred_test <- predict(svmmodel, testfactor)
mean(svm_pred_test == testfactor$class_pa)
svm_cm <- confusionMatrix(svm_pred_test, as.factor(testfactor$class_pa), positive="1")
svm_cm

####### RandomForest Model #######
set.seed(123)
rfmodel <- caret::train(class_pa~.,data=trainfactor,
                        method = "rf",
                        metric=metric,
                        ntree=1000,
                        tuneLength= Number_of_tune_tries,
                        trControl=cross_validation,
                        preProcess=preprocessed)
rfmodel
rfprediction <- predict(predictors,rfmodel, filename="pred_rf", progress ="text", overwrite=TRUE)
plot(rfmodel)
plot(rfprediction, main="RF - caret package")
plot(varImp(rfmodel)) #importance of each variable

#trying the model on the test data.
rf_pred_test <- predict(rfmodel, testfactor)
mean(rf_pred_test == testfactor$class_pa)
rf_cm <- confusionMatrix(rf_pred_test, as.factor(testfactor$class_pa), positive = "1")
rf_cm


####### NeuralNetworks Model #######
set.seed(123)
nnetmodel <- caret::train(class_pa~.,data=trainfactor,
                          method = "nnet",
                          metric=metric,
                          tuneLength= Number_of_tune_tries,
                          trControl=cross_validation,
                          preProcess=preprocessed)
nnetmodel

nnetprediction <- predict(predictors,nnetmodel, filename="pred_nnet", progress ="text", overwrite=TRUE)

plot(nnetmodel)
plot(nnetprediction, main="NNET - caret package")
plot(varImp(nnetmodel)) #importance of each variable


#trying the model on the test data.
nnet_pred_test <- predict(nnetmodel, testfactor)
mean(nnet_pred_test == testfactor$class_pa)
nnet_cm <- confusionMatrix(nnet_pred_test, as.factor(testfactor$class_pa), positive="1")

nnet_pred_test
nnet_cm


####### MaxEnt Model ######

#PREPARE DATA FOR MAXENT
#1. We need to create a df with only clim data (no other columns)
test_clean <- testData
test_clean[,c("class_pa")] <- NULL
train_clean <- trainData
train_clean[,c("class_pa")] <- NULL

#2.Maxent implementation (get the model)
maxentmodel<- dismo::maxent(x=train_clean, #env. conditions
                    p=trainData$class_pa, #presence/absence vector
                    factors="lcmask1","evtUT",
                    path=paste0(getwd(),"/maxent_outputs4"), #folder to put outputs
                    args=c("responsecurves") #parameter specification
)

maxentmodel
plot(maxentmodel)

#Predict and project the data onto different layers
class(predictors)

maxent_pred<-predict(maxentmodel,predictors, progress="text")

plot(maxent_pred)

# MAXENT MODEL EVALUATION
#For the evaluation, we need to separate the train and test data into presence and absence
X<-split(trainData, trainData$class_pa) #split the df into 2 lists
names(X) <- c("a_all", "p_all") #gives a new name to the lists
list2env(X, envir = .GlobalEnv) #sends the lists to the environment as df

a <- a_all
a[,c("class_pa")]<-NULL

p <- p_all
p[,c("class_pa")]<-NULL

#We repeat for the test data
X_test<-split(testData, testData$class_pa) #split the df into 2 lists
names(X_test) <- c("a_all_test", "p_all_test") #gives a new name to the lists
list2env(X_test, envir = .GlobalEnv) #sends the lists to the environment as df

a_test <- a_all_test
a_test[,c("class_pa")]<-NULL

p_test <- p_all_test
p_test[,c("class_pa")]<-NULL

#1 Evaluate the model with the training data
maxenteval_train<- dismo::evaluate(p=p,a=a,model=maxentmodel)
print(maxenteval_train)

#2. Evaluate the model with the test data
maxenteval<- dismo::evaluate(p=p_test,a=a_test,model=maxentmodel)
print(maxenteval)

#Confusion matrix and kappa
tss<-maxenteval@TPR+maxenteval@TNR-1
thd<-maxenteval@t[which.max(tss)]
print(maxenteval@t)

maxent_cm<-maxenteval@confusion[158,] #change the number of the row according to the row for the thd, they are the same row
print(maxent_cm)
kappa<-max(maxenteval@kappa)
print(kappa)


#Adding thresholds (threshold for best model is TPR+TNR, thus highest TSS)
thd2<- threshold(maxenteval,stat="spec_sens") #highest TSS

maxent_finalpred<-maxent_pred>thd2
plot(maxent_finalpred)


##### Models Evaluation #####

#SVM
evsvm <- evaluate(p=p_all_test, a=a_all_test, model=svmmodel) #test data
evsvm

#RF
evrf <- evaluate(p=p_all_test, a=a_all_test, model=rfmodel) #test data
evrf

#ANN
evnnet <- evaluate(p=p_all_test, a=a_all_test, model=nnetmodel) #test data
evnnet

#summarize accuracy of caret models#

results <- resamples(list(svm=svmmodel, rf=rfmodel, nnet=nnetmodel))
summary(results)

#plot all the prediction maps and average mean
models <- stack(svmprediction, rfprediction, nnetprediction, maxent_pred)
names(models) <- c("SVM", "RandomForest", "ANN", "Maxent")

m <- mean(models)
plot(m, main='average score')

auc <- sapply(list(evsvm, evrf, evnnet, maxenteval), function(x) x@auc)
w <- (auc-0.5)^2

m2 <- weighted.mean(models[[c("SVM", "RandomForest", "ANN", "Maxent")]], w)
plot(m2, main='Weighted Mean of All Prediction Models')

#### EXTERNAL VALIDATION ####
predictors2.list<-list.files("idaho/predictors", pattern=".tif$", full.names = T)
predictors2 <- raster::stack(predictors2.list)

sage_idaho <- shapefile("idaho/sage_clean_idaho.shp")
rfprediction_va <- predict(predictors2, rfmodel, filename="predval_rf", overwrite=T)
sage_idaho_transformed <- spTransform(sage_idaho, crs(rfprediction_va))
plot(rfprediction_va)
plot(sage_idaho_transformed, add=T)

idaho_counts <- raster::extract(rfprediction_va, sage_idaho_transformed, df=TRUE)
table(idaho_counts$layer)


##### PART B - FUTURE PREDICTIONS #####

#### Present Data ####
biopredic_list<- list.files("PredictorsFinalBioclimOnly", pattern=".tif$", full.names = T)
biopredictors <- raster::stack(biopredic_list)

#merge new list of predictors with presence/absence data
biotraining <- raster::extract(biopredictors, sage, df=TRUE) #extract the rasters and sage into a df
biopder <- merge(biotraining, sage, by.x="ID", by.y="ID") #merge the columns

#remove extra columns not needed (leave only rasters and class_pa)
biopder[,c("ID","species", "eventDt", "day", "month", "year", "issue", "season", "geometry")] <- NULL 


anyNA(biopder) #make sure there are still no NAs
biopder_clean <- biopder #for making it uniform

#part the data into train and test
set.seed(100)
biotrainids <- createDataPartition(biopder_clean$class_pa,list=FALSE,p=0.7) 
biotrainData <- biopder_clean[biotrainids,] #70%
biotestData <- biopder_clean[-biotrainids,] #30%

#Maxent cannot use the p/a as factor, so we transform only for the other models
biotrainfactor<-biotrainData
biotrainfactor$class_pa <- as.factor(biotrainfactor$class_pa)
biotestfactor<-biotestData
biotestfactor$class_pa <- as.factor(biotestfactor$class_pa)

#### Models ####

### Regular parameters - same as previous, here for readability ###
cross_validation <- trainControl(method="cv",search="random", number=10)
metric <- "Accuracy"
Number_of_tune_tries <- 15
preprocessed <- c("scale", "center")

####### SVM Model #######
set.seed(123) #for reproducibility
biosvmmodel <- caret::train(class_pa~.,data=biotrainfactor,
                         method = "svmRadial",
                         metric=metric,
                         tuneLength= Number_of_tune_tries,
                         trControl=cross_validation,
                         preProcess=preprocessed)

biosvmmodel

biosvmprediction <- predict(biopredictors, biosvmmodel, filename="future predictions/pred_svm_present", progress ="text", overwrite=TRUE)
biosvmprediction
plot(biosvmmodel)
plot(biosvmprediction, main="SVM - caret package")
plot(varImp(biosvmmodel), main="Importance of Each Variable") #importance of each variable

#trying the model on the test data.
bio_svm_pred_test <- predict(biosvmmodel, biotestfactor)
mean(bio_svm_pred_test == biotestfactor$class_pa)
bio_svm_cm <- confusionMatrix(bio_svm_pred_test, as.factor(biotestfactor$class_pa), positive="1")
bio_svm_cm

####### RandomForest Model #######
set.seed(123)
biorfmodel <- caret::train(class_pa~.,data=biotrainfactor,
                        method = "rf",
                        metric=metric,
                        ntree=1000,
                        tuneLength= Number_of_tune_tries,
                        trControl=cross_validation,
                        preProcess=preprocessed)
biorfmodel
biorfprediction <- predict(biopredictors,biorfmodel, filename="future predictions/pred_rf_present", progress ="text", overwrite=TRUE)

plot(biorfmodel)
plot(biorfprediction, main="RF - caret package")
plot(varImp(biorfmodel)) #importance of each variable

#trying the model on the test data.
bio_rf_pred_test <- predict(biorfmodel, biotestfactor)
mean(bio_rf_pred_test == biotestfactor$class_pa)
biorf_cm <- confusionMatrix(bio_rf_pred_test, as.factor(biotestfactor$class_pa), positive = "1")
biorf_cm


####### NeuralNetworks Model #######
set.seed(123)
bionnetmodel <- caret::train(class_pa~.,data=biotrainfactor,
                          method = "nnet",
                          metric=metric,
                          tuneLength= Number_of_tune_tries,
                          trControl=cross_validation,
                          preProcess=preprocessed)
bionnetmodel
bionnetmodel$bestTune

bionnetprediction <- predict(biopredictors,bionnetmodel, filename="future predictions/prednoncor_nnet_present", progress ="text", overwrite=TRUE)

plot(bionnetmodel)
plot(bionnetprediction, main="NNET - caret package")
plot(varImp(bionnetmodel)) #importance of each variable


#trying the model on the test data.
bio_nnet_pred_test <- predict(bionnetmodel, biotestfactor)
mean(bio_nnet_pred_test == biotestfactor$class_pa)
bionnet_cm <- confusionMatrix(bio_nnet_pred_test, as.factor(biotestfactor$class_pa), positive="1")

bio_nnet_pred_test
bionnet_cm


####### MaxEnt Model ######

#PREPARE DATA FOR MAXENT
#1. We need to create a df with only clim data (no other columns)
biotest_clean <- biotestData
biotest_clean[,c("class_pa")] <- NULL
biotrain_clean <- biotrainData
biotrain_clean[,c("class_pa")] <- NULL

#2.Maxent implementation (get the model)
biomaxentmodel<- dismo::maxent(x=biotrain_clean, #env. conditions
                            p=biotrainData$class_pa, #presence/absence vector
                            factors="lcmask1","evtUT",
                            path=paste0(getwd(),"/maxent_outputs4"), #folder to put outputs
                            args=c("responsecurves") #parameter specification
)

biomaxentmodel
plot(biomaxentmodel)

#Predict and project the data onto different layers
class(biopredictors)

biomaxent_pred<-predict(biomaxentmodel,biopredictors, progress="text")

plot(biomaxent_pred)

# MAXENT MODEL EVALUATION
#For the evaluation, we need to separate the train and test data into presence and absence
#We do it for the test data
bioX_test<-split(biotestData, biotestData$class_pa) #split the df into 2 lists
names(bioX_test) <- c("bio_a_all_test", "bio_p_all_test") #gives a new name to the lists
list2env(bioX_test, envir = .GlobalEnv) #sends the lists to the environment as df

bio_a_test <- bio_a_all_test
bio_a_test[,c("class_pa")]<-NULL
bio_p_test <- bio_p_all_test
bio_p_test[,c("class_pa")]<-NULL

# Evaluate the model with the test data
biomaxenteval<- dismo::evaluate(p=bio_p_test,a=bio_a_test,model=biomaxentmodel)
print(biomaxenteval)

#Adding thresholds (threshold for best model is TPR+TNR, thus highest TSS)
biothd2<- threshold(biomaxenteval,stat="spec_sens") #highest TSS

##downloading the data from Worldclim.org for ssp 370##

bio_url<-"http://biogeo.ucdavis.edu/data/worldclim/v2.1/fut/2.5m/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.zip"

download.file(bio_url, "bioclimfuture/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.zip")

unzip("bioclimfuture/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.zip", exdir=".",  unzip = "internal")


##### Future Layers - ssp370 #####
bio1_mean_temp <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=1)
bio3_Isothermality <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=3)
bio7_temp_annual_range <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=7)
bio8_mean_temp_wettest_qtr <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=8)
bio9_mean_temp_driest_qtr <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=9)
bio12_annual_precip <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=12)
bio13_precip_wettest_month <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=13)
bio14_precip_driest_month <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp370/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp370_2041-2060.tif", band=14)
elevation <- raster("bioclimfuture/elevhillshade.tif")
landCover <- raster("bioclimfuture/lcmask1.tif")


names(bio1_mean_temp) <- "bio1_mean_temp"
names(bio3_Isothermality) <- "bio3_Isothermality"
names(bio7_temp_annual_range) <- "bio7_temp_annual_range"
names(bio8_mean_temp_wettest_qtr) <- "bio8_mean_temp_wettest_qtr"
names(bio9_mean_temp_driest_qtr) <- "bio9_mean_temp_driest_qtr"
names(bio12_annual_precip) <- "bio12_annual_precip"
names(bio13_precip_wettest_month) <- "bio13_precip_wettest_month"
names(bio14_precip_driest_month) <- "bio14_precip_driest_month"

#resample elevation and landcover to sae as bioclim layers

elevation <- resample(elevation, bio1_mean_temp, progress="text")
landCover <- resample(landCover, bio1_mean_temp, progress="text")

# resample future predictor layers to the same as the bio raster present layers
predictors_future <- raster::stack(bio1_mean_temp, bio3_Isothermality, bio7_temp_annual_range, bio8_mean_temp_wettest_qtr, bio9_mean_temp_driest_qtr, bio12_annual_precip, bio13_precip_wettest_month, bio14_precip_driest_month, elevation, landCover)
predictors_future_res <- resample(predictors_future, biopredictors, progress="text")
predictors_future_ext <- extend(predictors_future_res, extent(biopredictors))


##### FUTURE PREDICTIONS - ssp 370 #####

nnetprediction <- predict(predictors_future_res,bionnetmodel, filename="future predictions/2041_2060_prediction_nnet", progress ="text", overwrite=TRUE)
nnetprediction <- predict(predictors_future_res,bionnetmodel, filename="future predictions/2041_2060_prediction_nnet.tif", progress ="text", overwrite=TRUE)

rfprediction <- predict(predictors_future_res,biorfmodel, filename="future predictions/2041_2060_prediction_rf", progress ="text", overwrite=TRUE)
rfprediction <- predict(predictors_future_res,biorfmodel, filename="future predictions/2041_2060_prediction_rf.tif", progress ="text", overwrite=TRUE)

svmprediction <- predict(predictors_future_res,biosvmmodel, filename="future predictions/2041_2060_prediction_svm", progress ="text", overwrite=TRUE)
svmprediction <- predict(predictors_future_res,biosvmmodel, filename="future predictions/2041_2060_prediction_svm.tif", progress ="text", overwrite=TRUE)

maxentprediction <- predict(predictors_future_res,biomaxentmodel, progress ="text", overwrite=TRUE)

###CREATING MAXENT BY THRESHHOLD Adding thresholds (threshold for best model is TPR+TNR, thus highest TSS)

maxentpredictionfinal<-maxentprediction>biothd2

plot(maxentpredictionfinal)

writeRaster(maxentpredictionfinal, "future predictions/2041_2060_prediction_maxent.grd", "raster", overwrite=TRUE)


##downloading the data from Worldclim.org for ssp 245##

bio2_url<-"http://biogeo.ucdavis.edu/data/worldclim/v2.1/fut/2.5m/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.zip"

download.file(bio2_url, "bioclimfuture/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.zip")

unzip("bioclimfuture/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.zip", exdir=".",  unzip = "internal")


##### Future Layers - ssp245 #####
bio1_mean_temp <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=1)
bio3_Isothermality <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=3)
bio7_temp_annual_range <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=7)
bio8_mean_temp_wettest_qtr <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=8)
bio9_mean_temp_driest_qtr <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=9)
bio12_annual_precip <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=12)
bio13_precip_wettest_month <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=13)
bio14_precip_driest_month <- raster("./share/spatial03/worldclim/cmip6/7_fut//2.5m/BCC-CSM2-MR/ssp245/wc2.1_2.5m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif", band=14)
elevation <- raster("bioclimfuture/elevhillshade.tif")
landCover <- raster("bioclimfuture/lcmask1.tif")

names(bio1_mean_temp) <- "bio1_mean_temp"
names(bio3_Isothermality) <- "bio3_Isothermality"
names(bio7_temp_annual_range) <- "bio7_temp_annual_range"
names(bio8_mean_temp_wettest_qtr) <- "bio8_mean_temp_wettest_qtr"
names(bio9_mean_temp_driest_qtr) <- "bio9_mean_temp_driest_qtr"
names(bio12_annual_precip) <- "bio12_annual_precip"
names(bio13_precip_wettest_month) <- "bio13_precip_wettest_month"
names(bio14_precip_driest_month) <- "bio14_precip_driest_month"

#resample elevation and landcover to sae as bioclim layers

elevation <- resample(elevation, bio1_mean_temp, progress="text")
landCover <- resample(landCover, bio1_mean_temp, progress="text")

# resample future predictor layers to the same as the bio raster present layers
predictors_future <- raster::stack(bio1_mean_temp, bio3_Isothermality, bio7_temp_annual_range, bio8_mean_temp_wettest_qtr, bio9_mean_temp_driest_qtr, bio12_annual_precip, bio13_precip_wettest_month, bio14_precip_driest_month, elevation, landCover)
predictors_future_ext <- extend(predictors_future, extent(biopredictors))
predictors_future_res <- resample(predictors_future_ext, biopredictors, progress="text")

#### FUTURE PREDICTIONS - ssp245 ####

nnetprediction <- predict(predictors_future_res,bionnetmodel, filename="future predictions/245/2041_2060_prediction_nnet", progress ="text", overwrite=TRUE)
nnetprediction <- predict(predictors_future_res,bionnetmodel, filename="future predictions/245/2041_2060_prediction_nnet.tif", progress ="text", overwrite=TRUE)

rfprediction <- predict(predictors_future_res,biorfmodel, filename="future predictions/245/2041_2060_prediction_rf", progress ="text", overwrite=TRUE)
rfprediction <- predict(predictors_future_res,biorfmodel, filename="future predictions/245/2041_2060_prediction_rf.tif", progress ="text", overwrite=TRUE)

svmprediction <- predict(predictors_future_res,biosvmmodel, filename="future predictions/245/2041_2060_prediction_svm", progress ="text", overwrite=TRUE)
svmprediction <- predict(predictors_future_res,biosvmmodel, filename="future predictions/245/2041_2060_prediction_svm.tif", progress ="text", overwrite=TRUE)

maxentprediction <- predict(predictors_future_res,biomaxentmodel, progress ="text", overwrite=TRUE)

###CREATING MAXENT BY THRESHHOLD Adding thresholds (threshold for best model is TPR+TNR, thus highest TSS)

maxentpredictionfinal<-maxentprediction>biothd2

plot(maxentpredictionfinal)

writeRaster(maxentpredictionfinal, "future predictions/245/2041_2060_prediction_maxent.grd", "raster", overwrite=TRUE)
