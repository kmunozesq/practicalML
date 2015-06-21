# Activity Performance Classification
Karla Muñoz-Esquivel  
Sunday, June 21, 2015  

##Overview
This work focuses on creating a model that qualify how a person performs a physical activity usimg data from accelerometers located in the belt, forearm, arm and dumbell of six participants doing weight lifting. There are five possible classes "A" to "E", where "A" corresponds to the specific execution of the excercise and the other four classes correspond to common mistakes. The data for this investigation comes from http://groupware.les.inf.puc-rio.br/har. The selection of variables and its evaluation. In this case, The method employed for model validation is *3 fold-cross validation*, i.e. used to estimate the accuracy of the machine learning algorithms: *tree and naive bayes*. PCA was employed to avoid overfitting. The best model was the **naive bayes model**, since its accuracy of classification is 0.9447 and the accuracy of the **tree model** obtained was 0.47.

###Loading and Formatting the data
The data will be formated (remove columns with NA and timestamps) to create and validate the model using *"3-fold cross validation"*. The data provided for testing does not have the outcome *classe*, so it was not used either for training or testing the model.

```r
filenametrain <- file.path(getwd(), "PracticalMachineLearning", "pml-training.csv")
pro_data <- read.csv(filenametrain, header = TRUE, sep = ",", quote = "\"",dec = ".", fill = TRUE, stringsAsFactors = FALSE)

library(data.table)
pro_data = data.table(pro_data)
library(dplyr)
pro_data<- pro_data[, which(!grepl(".*timestamp.*$", colnames(pro_data))), with=FALSE]
pro_data<-  data.table(mutate(pro_data, user_name = as.factor(user_name),
                              classe = as.factor(classe)))
#Converting variables that are type character to numeric
data_num <-pro_data[, which(!grepl("^X|classe|user_name|new_window|num_window$", colnames(pro_data))), with=FALSE]
data_num2<- data.table(sapply(data_num, as.numeric))
tmp_data <- data.table(mutate(data_num2, X = pro_data$X, 
                              classe = pro_data$classe,
                              user_name = pro_data$user_name))

#variables with all NAs -> removed
na_cols_all<-sapply(tmp_data, function(x)all(is.na(x)))
cols <- na_cols_all[na_cols_all==TRUE]
rexp <- paste('^',paste(names(cols), collapse='|'),'$', sep='')
final_data <- data.table(tmp_data[, which(!grepl(rexp, colnames(tmp_data))), with=FALSE])

#variables with any NAs, which are 94 variables out of 149 -> leave only complete cases
final_data<- final_data[complete.cases(final_data)==TRUE]
#Deleting unused variables
rm(list = c("na_cols_all","cols", "rexp", "data_num", "data_num2", "tmp_data", "pro_data", "filenametrain"))
gc(c("na_cols_all","cols", "rexp", "data_num", "data_num2", "tmp_data", "pro_data", "filenametrain"))
```

```
##            used (Mb) gc trigger  (Mb) max used  (Mb)
## Ncells   625010 33.4    2373200 126.8  3708127 198.1
## Vcells 11691386 89.2   40835376 311.6 51024671 389.3
```

###Processing the data and creating data models using caret

```r
#Setting up the cross-validation
library(caret)
set.seed(181899)
#Remove variables with cero variance:  amplitude_yaw_belt, amplitude_yaw_dumbbell, amplitude_yaw_forearm
final_data <- final_data[, which(!grepl("^amplitude_yaw_forearm|amplitude_yaw_belt|amplitude_yaw_dumbbell$", colnames(final_data))), with=FALSE]
final_data_without_factors <- final_data[, which(!grepl("^X|classe|user_name$", colnames(final_data))), with=FALSE]
preProc <- preProcess(final_data_without_factors, method="pca")
training_pcs <- predict(preProc,final_data_without_factors) 
final_data <- data.table(mutate(training_pcs, user_name= final_data$user_name, classe=final_data$classe))
train_control <- trainControl(method="cv", number=3, returnData=FALSE)

# Training a TREE
modelTREE <- train(classe ~ ., data = final_data, method ="rpart", trControl = train_control)
# Obtaining the predictions
predictions <- predict(modelTREE, final_data[, which(!grepl("^classe$", colnames(final_data))), with=FALSE])
# summarize results
print(confusionMatrix(predictions, final_data$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  A  B  C  D  E
##          A 49 39 11 26 14
##          B  0  0  0  0  0
##          C  4  1 25  1  1
##          D  0  0  0  0  0
##          E  1  8  1  8 28
## 
## Overall Statistics
##                                           
##                Accuracy : 0.47            
##                  95% CI : (0.4021, 0.5388)
##     No Information Rate : 0.2488          
##     P-Value [Acc > NIR] : 1.514e-12       
##                                           
##                   Kappa : 0.3148          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9074   0.0000   0.6757   0.0000   0.6512
## Specificity            0.4479   1.0000   0.9611   1.0000   0.8966
## Pos Pred Value         0.3525      NaN   0.7813      NaN   0.6087
## Neg Pred Value         0.9359   0.7788   0.9351   0.8387   0.9123
## Prevalence             0.2488   0.2212   0.1705   0.1613   0.1982
## Detection Rate         0.2258   0.0000   0.1152   0.0000   0.1290
## Detection Prevalence   0.6406   0.0000   0.1475   0.0000   0.2120
## Balanced Accuracy      0.6776   0.5000   0.8184   0.5000   0.7739
```

```r
print(modelTREE$finalModel)
```

```
## n= 217 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 217 163 A (0.25 0.22 0.17 0.16 0.2)  
##   2) PC4>=-1.433578 171 118 A (0.31 0.23 0.21 0.16 0.088)  
##     4) PC8< 1.959597 139  90 A (0.35 0.28 0.079 0.19 0.1) *
##     5) PC8>=1.959597 32   7 C (0.12 0.031 0.78 0.031 0.031) *
##   3) PC4< -1.433578 46  18 E (0.022 0.17 0.022 0.17 0.61) *
```

```r
# Training a Naive Bayes
modelNB <- train(classe ~ ., data = final_data, method ="nb", trControl = train_control)
# Obtaining the predictions
predictions <- predict(modelNB, final_data[, which(!grepl("^classe$", colnames(final_data))), with=FALSE])
# summarize results
print(confusionMatrix(predictions, final_data$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  A  B  C  D  E
##          A 50  1  0  0  0
##          B  0 43  0  0  1
##          C  4  4 37  2  0
##          D  0  0  0 33  0
##          E  0  0  0  0 42
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9447          
##                  95% CI : (0.9054, 0.9711)
##     No Information Rate : 0.2488          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9306          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9259   0.8958   1.0000   0.9429   0.9767
## Specificity            0.9939   0.9941   0.9444   1.0000   1.0000
## Pos Pred Value         0.9804   0.9773   0.7872   1.0000   1.0000
## Neg Pred Value         0.9759   0.9711   1.0000   0.9891   0.9943
## Prevalence             0.2488   0.2212   0.1705   0.1613   0.1982
## Detection Rate         0.2304   0.1982   0.1705   0.1521   0.1935
## Detection Prevalence   0.2350   0.2028   0.2166   0.1521   0.1935
## Balanced Accuracy      0.9599   0.9450   0.9722   0.9714   0.9884
```

```r
print(modelNB$finalModel)
```

```
## $apriori
## grouping
##         A         B         C         D         E 
## 0.2488479 0.2211982 0.1705069 0.1612903 0.1981567 
## 
## $tables
## $tables$PC1
## $tables$PC1$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 1.964
## 
##        x                  y            
##  Min.   :-13.2186   Min.   :0.0001822  
##  1st Qu.: -6.9425   1st Qu.:0.0106764  
##  Median : -0.6664   Median :0.0349601  
##  Mean   : -0.6664   Mean   :0.0397865  
##  3rd Qu.:  5.6096   3rd Qu.:0.0632905  
##  Max.   : 11.8857   Max.   :0.1037148  
## 
## $tables$PC1$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 1.706
## 
##        x                 y            
##  Min.   :-12.628   Min.   :0.0001001  
##  1st Qu.: -6.911   1st Qu.:0.0075199  
##  Median : -1.194   Median :0.0414989  
##  Mean   : -1.194   Mean   :0.0436840  
##  3rd Qu.:  4.522   3rd Qu.:0.0745245  
##  Max.   : 10.239   Max.   :0.1039850  
## 
## $tables$PC1$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 1.881
## 
##        x                  y            
##  Min.   :-12.7649   Min.   :0.0001467  
##  1st Qu.: -6.7145   1st Qu.:0.0081584  
##  Median : -0.6642   Median :0.0344624  
##  Mean   : -0.6642   Mean   :0.0412731  
##  3rd Qu.:  5.3862   3rd Qu.:0.0610813  
##  Max.   : 11.4366   Max.   :0.1231555  
## 
## $tables$PC1$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 2.214
## 
##        x                 y            
##  Min.   :-14.884   Min.   :0.0001468  
##  1st Qu.: -8.166   1st Qu.:0.0096330  
##  Median : -1.449   Median :0.0337604  
##  Mean   : -1.449   Mean   :0.0371687  
##  3rd Qu.:  5.269   3rd Qu.:0.0542236  
##  Max.   : 11.986   Max.   :0.1001868  
## 
## $tables$PC1$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 1.892
## 
##        x                  y            
##  Min.   :-11.9789   Min.   :0.0002663  
##  1st Qu.: -6.2005   1st Qu.:0.0122206  
##  Median : -0.4222   Median :0.0405279  
##  Mean   : -0.4222   Mean   :0.0432104  
##  3rd Qu.:  5.3562   3rd Qu.:0.0719384  
##  Max.   : 11.1346   Max.   :0.0975552  
## 
## 
## $tables$PC2
## $tables$PC2$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 1.538
## 
##        x                  y            
##  Min.   :-12.1656   Min.   :6.208e-05  
##  1st Qu.: -6.1826   1st Qu.:5.376e-03  
##  Median : -0.1996   Median :4.571e-02  
##  Mean   : -0.1996   Mean   :4.174e-02  
##  3rd Qu.:  5.7834   3rd Qu.:7.205e-02  
##  Max.   : 11.7664   Max.   :8.419e-02  
## 
## $tables$PC2$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 1.277
## 
##        x                  y            
##  Min.   :-10.3641   Min.   :7.348e-05  
##  1st Qu.: -5.3093   1st Qu.:5.909e-03  
##  Median : -0.2545   Median :4.355e-02  
##  Mean   : -0.2545   Mean   :4.941e-02  
##  3rd Qu.:  4.8002   3rd Qu.:8.444e-02  
##  Max.   :  9.8550   Max.   :1.279e-01  
## 
## $tables$PC2$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 1.408
## 
##        x                 y            
##  Min.   :-8.4121   Min.   :0.0001487  
##  1st Qu.:-3.7611   1st Qu.:0.0115774  
##  Median : 0.8898   Median :0.0657786  
##  Mean   : 0.8898   Mean   :0.0536907  
##  3rd Qu.: 5.5407   3rd Qu.:0.0840238  
##  Max.   :10.1916   Max.   :0.1118701  
## 
## $tables$PC2$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 1.598
## 
##        x                  y            
##  Min.   :-11.1966   Min.   :8.838e-05  
##  1st Qu.: -5.9859   1st Qu.:8.625e-03  
##  Median : -0.7752   Median :6.555e-02  
##  Mean   : -0.7752   Mean   :4.792e-02  
##  3rd Qu.:  4.4355   3rd Qu.:7.680e-02  
##  Max.   :  9.6462   Max.   :9.774e-02  
## 
## $tables$PC2$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 1.654
## 
##        x                  y            
##  Min.   :-12.0112   Min.   :0.0001209  
##  1st Qu.: -6.4429   1st Qu.:0.0096215  
##  Median : -0.8747   Median :0.0465708  
##  Mean   : -0.8747   Mean   :0.0448467  
##  3rd Qu.:  4.6936   3rd Qu.:0.0646396  
##  Max.   : 10.2618   Max.   :0.1061832  
## 
## 
## $tables$PC3
## $tables$PC3$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.9652
## 
##        x                 y            
##  Min.   :-31.196   Min.   :0.000e+00  
##  1st Qu.:-21.311   1st Qu.:8.000e-08  
##  Median :-11.426   Median :3.527e-03  
##  Mean   :-11.426   Mean   :2.526e-02  
##  3rd Qu.: -1.540   3rd Qu.:3.258e-02  
##  Max.   :  8.345   Max.   :1.538e-01  
## 
## $tables$PC3$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.9248
## 
##        x                  y            
##  Min.   :-7.63340   Min.   :0.0001016  
##  1st Qu.:-3.84096   1st Qu.:0.0102225  
##  Median :-0.04852   Median :0.0578431  
##  Mean   :-0.04852   Mean   :0.0658510  
##  3rd Qu.: 3.74392   3rd Qu.:0.1052782  
##  Max.   : 7.53636   Max.   :0.1827952  
## 
## $tables$PC3$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.6612
## 
##        x                y            
##  Min.   :-5.878   Min.   :0.0002571  
##  1st Qu.:-3.564   1st Qu.:0.0234474  
##  Median :-1.250   Median :0.1032218  
##  Mean   :-1.250   Mean   :0.1079292  
##  3rd Qu.: 1.063   3rd Qu.:0.1829029  
##  Max.   : 3.377   Max.   :0.2382676  
## 
## $tables$PC3$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.6439
## 
##        x                 y            
##  Min.   :-4.1665   Min.   :0.0002096  
##  1st Qu.:-1.5865   1st Qu.:0.0149275  
##  Median : 0.9935   Median :0.0633885  
##  Mean   : 0.9935   Mean   :0.0967967  
##  3rd Qu.: 3.5736   3rd Qu.:0.1756981  
##  Max.   : 6.1536   Max.   :0.2694219  
## 
## $tables$PC3$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.9142
## 
##        x                 y            
##  Min.   :-8.5655   Min.   :0.0001144  
##  1st Qu.:-4.0072   1st Qu.:0.0111601  
##  Median : 0.5512   Median :0.0412020  
##  Mean   : 0.5512   Mean   :0.0547870  
##  3rd Qu.: 5.1096   3rd Qu.:0.0851516  
##  Max.   : 9.6680   Max.   :0.1762635  
## 
## 
## $tables$PC4
## $tables$PC4$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.311
## 
##        x                 y           
##  Min.   :-22.749   Min.   :0.000000  
##  1st Qu.:-15.687   1st Qu.:0.000000  
##  Median : -8.624   Median :0.000000  
##  Mean   : -8.624   Mean   :0.035363  
##  3rd Qu.: -1.562   3rd Qu.:0.005938  
##  Max.   :  5.501   Max.   :0.476773  
## 
## $tables$PC4$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.8049
## 
##        x                 y            
##  Min.   :-6.8267   Min.   :0.0001179  
##  1st Qu.:-3.6198   1st Qu.:0.0117589  
##  Median :-0.4129   Median :0.0599390  
##  Mean   :-0.4129   Mean   :0.0778739  
##  3rd Qu.: 2.7940   3rd Qu.:0.1335245  
##  Max.   : 6.0009   Max.   :0.2069786  
## 
## $tables$PC4$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3754
## 
##        x                 y            
##  Min.   :-3.7519   Min.   :0.0003233  
##  1st Qu.:-2.0059   1st Qu.:0.0178028  
##  Median :-0.2599   Median :0.0745212  
##  Mean   :-0.2599   Mean   :0.1430312  
##  3rd Qu.: 1.4861   3rd Qu.:0.2784284  
##  Max.   : 3.2321   Max.   :0.4260528  
## 
## $tables$PC4$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.9371
## 
##        x                 y            
##  Min.   :-10.778   Min.   :0.0001371  
##  1st Qu.: -6.327   1st Qu.:0.0131019  
##  Median : -1.876   Median :0.0360143  
##  Mean   : -1.876   Mean   :0.0561060  
##  3rd Qu.:  2.576   3rd Qu.:0.0651983  
##  Max.   :  7.027   Max.   :0.2035024  
## 
## $tables$PC4$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 1.234
## 
##        x                 y            
##  Min.   :-11.183   Min.   :0.0000991  
##  1st Qu.: -6.538   1st Qu.:0.0078926  
##  Median : -1.894   Median :0.0542798  
##  Mean   : -1.894   Mean   :0.0537686  
##  3rd Qu.:  2.751   3rd Qu.:0.0933190  
##  Max.   :  7.396   Max.   :0.1164585  
## 
## 
## $tables$PC5
## $tables$PC5$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.7476
## 
##        x                 y            
##  Min.   :-15.963   Min.   :1.539e-05  
##  1st Qu.:-10.297   1st Qu.:3.602e-03  
##  Median : -4.631   Median :1.016e-02  
##  Mean   : -4.631   Mean   :4.407e-02  
##  3rd Qu.:  1.036   3rd Qu.:5.692e-02  
##  Max.   :  6.702   Max.   :1.868e-01  
## 
## $tables$PC5$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.945
## 
##        x                y            
##  Min.   :-8.767   Min.   :0.0001098  
##  1st Qu.:-4.950   1st Qu.:0.0110663  
##  Median :-1.133   Median :0.0566150  
##  Mean   :-1.133   Mean   :0.0654264  
##  3rd Qu.: 2.684   3rd Qu.:0.1123693  
##  Max.   : 6.501   Max.   :0.1643605  
## 
## $tables$PC5$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.556
## 
##        x                   y            
##  Min.   :-5.256079   Min.   :0.0002249  
##  1st Qu.:-2.628121   1st Qu.:0.0273397  
##  Median :-0.000162   Median :0.0690512  
##  Mean   :-0.000162   Mean   :0.0950306  
##  3rd Qu.: 2.627796   3rd Qu.:0.1512839  
##  Max.   : 5.255754   Max.   :0.2746382  
## 
## $tables$PC5$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 1.332
## 
##        x                y            
##  Min.   :-8.125   Min.   :9.608e-05  
##  1st Qu.:-2.589   1st Qu.:6.235e-03  
##  Median : 2.947   Median :2.284e-02  
##  Mean   : 2.947   Mean   :4.511e-02  
##  3rd Qu.: 8.483   3rd Qu.:6.676e-02  
##  Max.   :14.019   Max.   :1.505e-01  
## 
## $tables$PC5$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 1.15
## 
##        x                y            
##  Min.   :-8.520   Min.   :9.085e-05  
##  1st Qu.:-3.702   1st Qu.:6.678e-03  
##  Median : 1.115   Median :3.586e-02  
##  Mean   : 1.115   Mean   :5.184e-02  
##  3rd Qu.: 5.932   3rd Qu.:1.035e-01  
##  Max.   :10.750   Max.   :1.274e-01  
## 
## 
## $tables$PC6
## $tables$PC6$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.9495
## 
##        x                 y            
##  Min.   :-8.8221   Min.   :8.769e-05  
##  1st Qu.:-4.0193   1st Qu.:5.901e-03  
##  Median : 0.7835   Median :1.710e-02  
##  Mean   : 0.7835   Mean   :5.200e-02  
##  3rd Qu.: 5.5863   3rd Qu.:1.052e-01  
##  Max.   :10.3890   Max.   :1.612e-01  
## 
## $tables$PC6$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 1.02
## 
##        x                 y            
##  Min.   :-7.9993   Min.   :9.353e-05  
##  1st Qu.:-3.8665   1st Qu.:7.038e-03  
##  Median : 0.2662   Median :5.065e-02  
##  Mean   : 0.2662   Mean   :6.043e-02  
##  3rd Qu.: 4.3990   3rd Qu.:1.192e-01  
##  Max.   : 8.5318   Max.   :1.322e-01  
## 
## $tables$PC6$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 1.01
## 
##        x                y            
##  Min.   :-5.633   Min.   :0.0002268  
##  1st Qu.:-2.071   1st Qu.:0.0142383  
##  Median : 1.491   Median :0.0724595  
##  Mean   : 1.491   Mean   :0.0701074  
##  3rd Qu.: 5.053   3rd Qu.:0.1176432  
##  Max.   : 8.615   Max.   :0.1593726  
## 
## $tables$PC6$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.7716
## 
##        x                y            
##  Min.   :-9.976   Min.   :0.0001694  
##  1st Qu.:-5.747   1st Qu.:0.0232612  
##  Median :-1.519   Median :0.0334767  
##  Mean   :-1.519   Mean   :0.0590596  
##  3rd Qu.: 2.710   3rd Qu.:0.0770919  
##  Max.   : 6.938   Max.   :0.2052535  
## 
## $tables$PC6$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.9402
## 
##        x                 y            
##  Min.   :-6.8932   Min.   :0.0001137  
##  1st Qu.:-3.0230   1st Qu.:0.0124733  
##  Median : 0.8473   Median :0.0464967  
##  Mean   : 0.8473   Mean   :0.0645263  
##  3rd Qu.: 4.7175   3rd Qu.:0.1244693  
##  Max.   : 8.5878   Max.   :0.1610585  
## 
## 
## $tables$PC7
## $tables$PC7$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.7735
## 
##        x                 y            
##  Min.   :-10.521   Min.   :0.0001074  
##  1st Qu.: -6.059   1st Qu.:0.0032360  
##  Median : -1.597   Median :0.0194419  
##  Mean   : -1.597   Mean   :0.0559700  
##  3rd Qu.:  2.865   3rd Qu.:0.1026727  
##  Max.   :  7.327   Max.   :0.2244657  
## 
## $tables$PC7$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 1.219
## 
##        x                 y            
##  Min.   :-8.8821   Min.   :8.242e-05  
##  1st Qu.:-4.0568   1st Qu.:6.753e-03  
##  Median : 0.7684   Median :4.585e-02  
##  Mean   : 0.7684   Mean   :5.176e-02  
##  3rd Qu.: 5.5936   3rd Qu.:9.758e-02  
##  Max.   :10.4188   Max.   :1.176e-01  
## 
## $tables$PC7$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.5898
## 
##        x                 y            
##  Min.   :-2.9746   Min.   :0.0003657  
##  1st Qu.:-0.7656   1st Qu.:0.0240430  
##  Median : 1.4435   Median :0.0801813  
##  Mean   : 1.4435   Mean   :0.1130462  
##  3rd Qu.: 3.6526   3rd Qu.:0.1927606  
##  Max.   : 5.8616   Max.   :0.2975198  
## 
## $tables$PC7$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.668
## 
##        x                y            
##  Min.   :-7.008   Min.   :0.0001940  
##  1st Qu.:-4.245   1st Qu.:0.0144020  
##  Median :-1.482   Median :0.0716705  
##  Mean   :-1.482   Mean   :0.0903998  
##  3rd Qu.: 1.280   3rd Qu.:0.1511143  
##  Max.   : 4.043   Max.   :0.2537468  
## 
## $tables$PC7$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 1.011
## 
##        x                 y            
##  Min.   :-9.1699   Min.   :0.0001895  
##  1st Qu.:-5.0652   1st Qu.:0.0132624  
##  Median :-0.9604   Median :0.0442088  
##  Mean   :-0.9604   Mean   :0.0608390  
##  3rd Qu.: 3.1443   3rd Qu.:0.0854549  
##  Max.   : 7.2490   Max.   :0.1965633  
## 
## 
## $tables$PC8
## $tables$PC8$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.5976
## 
##        x                  y            
##  Min.   :-7.19681   Min.   :0.0001394  
##  1st Qu.:-3.55627   1st Qu.:0.0080623  
##  Median : 0.08428   Median :0.0328870  
##  Mean   : 0.08428   Mean   :0.0686008  
##  3rd Qu.: 3.72482   3rd Qu.:0.1112742  
##  Max.   : 7.36536   Max.   :0.2573174  
## 
## $tables$PC8$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.8077
## 
##        x                 y            
##  Min.   :-6.8956   Min.   :0.0001379  
##  1st Qu.:-3.8094   1st Qu.:0.0131859  
##  Median :-0.7231   Median :0.0772590  
##  Mean   :-0.7231   Mean   :0.0809191  
##  3rd Qu.: 2.3631   3rd Qu.:0.1442937  
##  Max.   : 5.4494   Max.   :0.1817728  
## 
## $tables$PC8$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.628
## 
##        x                 y           
##  Min.   :-3.7718   Min.   :0.000193  
##  1st Qu.:-0.8115   1st Qu.:0.018319  
##  Median : 2.1488   Median :0.060436  
##  Mean   : 2.1488   Mean   :0.084358  
##  3rd Qu.: 5.1092   3rd Qu.:0.115719  
##  Max.   : 8.0695   Max.   :0.255994  
## 
## $tables$PC8$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.7587
## 
##        x                y           
##  Min.   :-4.951   Min.   :0.000169  
##  1st Qu.:-1.668   1st Qu.:0.009135  
##  Median : 1.616   Median :0.028204  
##  Mean   : 1.616   Mean   :0.076055  
##  3rd Qu.: 4.900   3rd Qu.:0.157576  
##  Max.   : 8.183   Max.   :0.218340  
## 
## $tables$PC8$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.8911
## 
##        x                 y            
##  Min.   :-6.5069   Min.   :0.0001462  
##  1st Qu.:-3.0872   1st Qu.:0.0119427  
##  Median : 0.3326   Median :0.0619947  
##  Mean   : 0.3326   Mean   :0.0730276  
##  3rd Qu.: 3.7524   3rd Qu.:0.1427841  
##  Max.   : 7.1721   Max.   :0.1618477  
## 
## 
## $tables$PC9
## $tables$PC9$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.7286
## 
##        x                 y            
##  Min.   :-6.2257   Min.   :0.0001141  
##  1st Qu.:-2.6695   1st Qu.:0.0110927  
##  Median : 0.8866   Median :0.0443437  
##  Mean   : 0.8866   Mean   :0.0702288  
##  3rd Qu.: 4.4428   3rd Qu.:0.1219336  
##  Max.   : 7.9989   Max.   :0.2053583  
## 
## $tables$PC9$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.7329
## 
##        x                y            
##  Min.   :-6.026   Min.   :0.0001275  
##  1st Qu.:-2.422   1st Qu.:0.0103211  
##  Median : 1.182   Median :0.0558953  
##  Mean   : 1.182   Mean   :0.0692859  
##  3rd Qu.: 4.787   3rd Qu.:0.1190562  
##  Max.   : 8.391   Max.   :0.1975190  
## 
## $tables$PC9$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.7572
## 
##        x                   y            
##  Min.   :-5.405453   Min.   :0.0002301  
##  1st Qu.:-2.707154   1st Qu.:0.0193199  
##  Median :-0.008855   Median :0.0800086  
##  Mean   :-0.008855   Mean   :0.0925469  
##  3rd Qu.: 2.689444   3rd Qu.:0.1720631  
##  Max.   : 5.387743   Max.   :0.1890145  
## 
## $tables$PC9$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.9037
## 
##        x                 y            
##  Min.   :-8.2547   Min.   :0.0001419  
##  1st Qu.:-4.2541   1st Qu.:0.0112481  
##  Median :-0.2534   Median :0.0381528  
##  Mean   :-0.2534   Mean   :0.0624239  
##  3rd Qu.: 3.7472   3rd Qu.:0.1135890  
##  Max.   : 7.7479   Max.   :0.1801257  
## 
## $tables$PC9$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.9217
## 
##        x                y            
##  Min.   :-8.357   Min.   :0.0001464  
##  1st Qu.:-4.700   1st Qu.:0.0130712  
##  Median :-1.043   Median :0.0396322  
##  Mean   :-1.043   Mean   :0.0682912  
##  3rd Qu.: 2.614   3rd Qu.:0.1322938  
##  Max.   : 6.271   Max.   :0.1744250  
## 
## 
## $tables$PC10
## $tables$PC10$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.6752
## 
##        x                y           
##  Min.   :-6.172   Min.   :0.000123  
##  1st Qu.:-2.360   1st Qu.:0.008199  
##  Median : 1.453   Median :0.047668  
##  Mean   : 1.453   Mean   :0.065508  
##  3rd Qu.: 5.265   3rd Qu.:0.091984  
##  Max.   : 9.078   Max.   :0.239931  
## 
## $tables$PC10$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.8405
## 
##        x                 y            
##  Min.   :-6.5490   Min.   :0.0001296  
##  1st Qu.:-3.0894   1st Qu.:0.0112275  
##  Median : 0.3702   Median :0.0521284  
##  Mean   : 0.3702   Mean   :0.0721872  
##  3rd Qu.: 3.8298   3rd Qu.:0.1404506  
##  Max.   : 7.2895   Max.   :0.1849694  
## 
## $tables$PC10$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.6541
## 
##        x                 y            
##  Min.   :-4.4754   Min.   :0.0002801  
##  1st Qu.:-2.0945   1st Qu.:0.0214030  
##  Median : 0.2864   Median :0.0953809  
##  Mean   : 0.2864   Mean   :0.1048853  
##  3rd Qu.: 2.6674   3rd Qu.:0.1807817  
##  Max.   : 5.0483   Max.   :0.2491177  
## 
## $tables$PC10$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.7738
## 
##        x                 y            
##  Min.   :-5.8079   Min.   :0.0002385  
##  1st Qu.:-3.0819   1st Qu.:0.0191316  
##  Median :-0.3558   Median :0.0876337  
##  Mean   :-0.3558   Mean   :0.0916052  
##  3rd Qu.: 2.3703   3rd Qu.:0.1565043  
##  Max.   : 5.0964   Max.   :0.2083680  
## 
## $tables$PC10$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 1.016
## 
##        x                 y            
##  Min.   :-7.0807   Min.   :0.0002206  
##  1st Qu.:-3.3974   1st Qu.:0.0134230  
##  Median : 0.2858   Median :0.0592703  
##  Mean   : 0.2858   Mean   :0.0678000  
##  3rd Qu.: 3.9690   3rd Qu.:0.1121017  
##  Max.   : 7.6523   Max.   :0.1658910  
## 
## 
## $tables$PC11
## $tables$PC11$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.9099
## 
##        x                 y            
##  Min.   :-7.8505   Min.   :0.0001383  
##  1st Qu.:-4.0978   1st Qu.:0.0127966  
##  Median :-0.3452   Median :0.0506458  
##  Mean   :-0.3452   Mean   :0.0665497  
##  3rd Qu.: 3.4074   3rd Qu.:0.1278043  
##  Max.   : 7.1600   Max.   :0.1605305  
## 
## $tables$PC11$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.7765
## 
##        x                 y            
##  Min.   :-5.6413   Min.   :0.0001232  
##  1st Qu.:-2.5540   1st Qu.:0.0118938  
##  Median : 0.5333   Median :0.0703746  
##  Mean   : 0.5333   Mean   :0.0808921  
##  3rd Qu.: 3.6206   3rd Qu.:0.1464402  
##  Max.   : 6.7078   Max.   :0.1883518  
## 
## $tables$PC11$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.5311
## 
##        x                 y            
##  Min.   :-4.9451   Min.   :0.0002281  
##  1st Qu.:-2.6575   1st Qu.:0.0214401  
##  Median :-0.3699   Median :0.1003618  
##  Mean   :-0.3699   Mean   :0.1091698  
##  3rd Qu.: 1.9177   3rd Qu.:0.1667985  
##  Max.   : 4.2052   Max.   :0.2964363  
## 
## $tables$PC11$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.7041
## 
##        x                 y            
##  Min.   :-5.5799   Min.   :0.0001854  
##  1st Qu.:-2.9439   1st Qu.:0.0113273  
##  Median :-0.3078   Median :0.0836287  
##  Mean   :-0.3078   Mean   :0.0947393  
##  3rd Qu.: 2.3282   3rd Qu.:0.1738267  
##  Max.   : 4.9642   Max.   :0.2177379  
## 
## $tables$PC11$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.913
## 
##        x                 y            
##  Min.   :-7.3646   Min.   :0.0001296  
##  1st Qu.:-3.8906   1st Qu.:0.0115173  
##  Median :-0.4167   Median :0.0686449  
##  Mean   :-0.4167   Mean   :0.0718875  
##  3rd Qu.: 3.0573   3rd Qu.:0.1069057  
##  Max.   : 6.5313   Max.   :0.1876392  
## 
## 
## $tables$PC12
## $tables$PC12$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.7374
## 
##        x                y            
##  Min.   :-7.167   Min.   :0.0001204  
##  1st Qu.:-4.107   1st Qu.:0.0133830  
##  Median :-1.046   Median :0.0555183  
##  Mean   :-1.046   Mean   :0.0815981  
##  3rd Qu.: 2.015   3rd Qu.:0.1491885  
##  Max.   : 5.075   Max.   :0.2141256  
## 
## $tables$PC12$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.5299
## 
##        x                 y           
##  Min.   :-4.8245   Min.   :0.000188  
##  1st Qu.:-2.6558   1st Qu.:0.015893  
##  Median :-0.4871   Median :0.089813  
##  Mean   :-0.4871   Mean   :0.115157  
##  3rd Qu.: 1.6816   3rd Qu.:0.220845  
##  Max.   : 3.8503   Max.   :0.274487  
## 
## $tables$PC12$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.6473
## 
##        x                 y            
##  Min.   :-5.1201   Min.   :0.0001883  
##  1st Qu.:-2.6118   1st Qu.:0.0155026  
##  Median :-0.1035   Median :0.0951155  
##  Mean   :-0.1035   Mean   :0.0995610  
##  3rd Qu.: 2.4049   3rd Qu.:0.1564840  
##  Max.   : 4.9132   Max.   :0.2659212  
## 
## $tables$PC12$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.7885
## 
##        x                y            
##  Min.   :-4.289   Min.   :0.0001802  
##  1st Qu.:-1.445   1st Qu.:0.0156620  
##  Median : 1.400   Median :0.0871450  
##  Mean   : 1.400   Mean   :0.0877931  
##  3rd Qu.: 4.244   3rd Qu.:0.1495326  
##  Max.   : 7.089   Max.   :0.2000933  
## 
## $tables$PC12$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.6347
## 
##        x                 y            
##  Min.   :-6.1023   Min.   :0.0001645  
##  1st Qu.:-3.2220   1st Qu.:0.0126718  
##  Median :-0.3417   Median :0.0628673  
##  Mean   :-0.3417   Mean   :0.0867074  
##  3rd Qu.: 2.5386   3rd Qu.:0.1496157  
##  Max.   : 5.4188   Max.   :0.2492935  
## 
## 
## $tables$PC13
## $tables$PC13$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.6553
## 
##        x                 y            
##  Min.   :-8.7719   Min.   :0.0001272  
##  1st Qu.:-4.8287   1st Qu.:0.0060199  
##  Median :-0.8855   Median :0.0216743  
##  Mean   :-0.8855   Mean   :0.0633352  
##  3rd Qu.: 3.0577   3rd Qu.:0.1070567  
##  Max.   : 7.0009   Max.   :0.2367468  
## 
## $tables$PC13$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4726
## 
##        x                y            
##  Min.   :-4.544   Min.   :0.0001983  
##  1st Qu.:-1.743   1st Qu.:0.0059660  
##  Median : 1.058   Median :0.0352754  
##  Mean   : 1.058   Mean   :0.0891648  
##  3rd Qu.: 3.858   3rd Qu.:0.1486048  
##  Max.   : 6.659   Max.   :0.3385928  
## 
## $tables$PC13$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.6818
## 
##        x                  y            
##  Min.   :-5.46057   Min.   :0.0002058  
##  1st Qu.:-2.69575   1st Qu.:0.0149292  
##  Median : 0.06907   Median :0.0522525  
##  Mean   : 0.06907   Mean   :0.0903264  
##  3rd Qu.: 2.83389   3rd Qu.:0.1803368  
##  Max.   : 5.59870   Max.   :0.2332077  
## 
## $tables$PC13$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.5384
## 
##        x                 y            
##  Min.   :-5.5292   Min.   :0.0002386  
##  1st Qu.:-2.9149   1st Qu.:0.0154603  
##  Median :-0.3006   Median :0.0288542  
##  Mean   :-0.3006   Mean   :0.0955278  
##  3rd Qu.: 2.3137   3rd Qu.:0.1936102  
##  Max.   : 4.9280   Max.   :0.3110876  
## 
## $tables$PC13$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.8767
## 
##        x                 y            
##  Min.   :-6.8851   Min.   :0.0001966  
##  1st Qu.:-3.7649   1st Qu.:0.0151563  
##  Median :-0.6446   Median :0.0868300  
##  Mean   :-0.6446   Mean   :0.0800333  
##  3rd Qu.: 2.4756   3rd Qu.:0.1287457  
##  Max.   : 5.5959   Max.   :0.1751728  
## 
## 
## $tables$PC14
## $tables$PC14$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.5916
## 
##        x                y           
##  Min.   :-9.326   Min.   :0.000141  
##  1st Qu.:-5.684   1st Qu.:0.008210  
##  Median :-2.042   Median :0.016942  
##  Mean   :-2.042   Mean   :0.068570  
##  3rd Qu.: 1.600   3rd Qu.:0.123810  
##  Max.   : 5.242   Max.   :0.271066  
## 
## $tables$PC14$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.559
## 
##        x                y            
##  Min.   :-7.561   Min.   :0.0001676  
##  1st Qu.:-4.427   1st Qu.:0.0060098  
##  Median :-1.293   Median :0.0242270  
##  Mean   :-1.293   Mean   :0.0796885  
##  3rd Qu.: 1.841   3rd Qu.:0.1513966  
##  Max.   : 4.975   Max.   :0.2850155  
## 
## $tables$PC14$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.5194
## 
##        x                 y           
##  Min.   :-4.9473   Min.   :0.000233  
##  1st Qu.:-2.7123   1st Qu.:0.015341  
##  Median :-0.4774   Median :0.074220  
##  Mean   :-0.4774   Mean   :0.111741  
##  3rd Qu.: 1.7575   3rd Qu.:0.209957  
##  Max.   : 3.9925   Max.   :0.306855  
## 
## $tables$PC14$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.7155
## 
##        x                y            
##  Min.   :-5.323   Min.   :0.0001899  
##  1st Qu.:-2.534   1st Qu.:0.0131770  
##  Median : 0.255   Median :0.0655492  
##  Mean   : 0.255   Mean   :0.0895374  
##  3rd Qu.: 3.044   3rd Qu.:0.1414828  
##  Max.   : 5.833   Max.   :0.2626738  
## 
## $tables$PC14$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.6128
## 
##        x                 y            
##  Min.   :-6.3762   Min.   :0.0001706  
##  1st Qu.:-3.3476   1st Qu.:0.0176076  
##  Median :-0.3189   Median :0.0398873  
##  Mean   :-0.3189   Mean   :0.0824595  
##  3rd Qu.: 2.7097   3rd Qu.:0.1360368  
##  Max.   : 5.7383   Max.   :0.2790181  
## 
## 
## $tables$PC15
## $tables$PC15$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.6811
## 
##        x                y            
##  Min.   :-8.749   Min.   :0.0001221  
##  1st Qu.:-5.484   1st Qu.:0.0118623  
##  Median :-2.218   Median :0.0336388  
##  Mean   :-2.218   Mean   :0.0764703  
##  3rd Qu.: 1.048   3rd Qu.:0.1342417  
##  Max.   : 4.313   Max.   :0.2380855  
## 
## $tables$PC15$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.5228
## 
##        x                 y           
##  Min.   :-7.2555   Min.   :0.000179  
##  1st Qu.:-3.8049   1st Qu.:0.008308  
##  Median :-0.3544   Median :0.017208  
##  Mean   :-0.3544   Mean   :0.072378  
##  3rd Qu.: 3.0961   3rd Qu.:0.135808  
##  Max.   : 6.5467   Max.   :0.293022  
## 
## $tables$PC15$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.278
## 
##        x                y            
##  Min.   :-2.456   Min.   :0.0004397  
##  1st Qu.:-1.160   1st Qu.:0.0430802  
##  Median : 0.135   Median :0.1555928  
##  Mean   : 0.135   Mean   :0.1927938  
##  3rd Qu.: 1.430   3rd Qu.:0.2962430  
##  Max.   : 2.726   Max.   :0.5541658  
## 
## $tables$PC15$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.7077
## 
##        x                 y            
##  Min.   :-5.4720   Min.   :0.0003382  
##  1st Qu.:-2.9577   1st Qu.:0.0200073  
##  Median :-0.4434   Median :0.0877919  
##  Mean   :-0.4434   Mean   :0.0993212  
##  3rd Qu.: 2.0708   3rd Qu.:0.1731722  
##  Max.   : 4.5851   Max.   :0.2328668  
## 
## $tables$PC15$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.599
## 
##        x                 y            
##  Min.   :-6.3216   Min.   :0.0001835  
##  1st Qu.:-3.6139   1st Qu.:0.0174471  
##  Median :-0.9062   Median :0.0459828  
##  Mean   :-0.9062   Mean   :0.0922343  
##  3rd Qu.: 1.8014   3rd Qu.:0.1810170  
##  Max.   : 4.5091   Max.   :0.2613864  
## 
## 
## $tables$PC16
## $tables$PC16$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.5827
## 
##        x                 y            
##  Min.   :-4.5911   Min.   :0.0001499  
##  1st Qu.:-2.1335   1st Qu.:0.0169597  
##  Median : 0.3241   Median :0.0529605  
##  Mean   : 0.3241   Mean   :0.1016208  
##  3rd Qu.: 2.7816   3rd Qu.:0.2197450  
##  Max.   : 5.2392   Max.   :0.2425987  
## 
## $tables$PC16$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.631
## 
##        x                y            
##  Min.   :-4.074   Min.   :0.0001649  
##  1st Qu.:-1.387   1st Qu.:0.0189954  
##  Median : 1.301   Median :0.0426801  
##  Mean   : 1.301   Mean   :0.0929242  
##  3rd Qu.: 3.988   3rd Qu.:0.1820026  
##  Max.   : 6.676   Max.   :0.2676484  
## 
## $tables$PC16$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.5238
## 
##        x                 y            
##  Min.   :-3.9554   Min.   :0.0002313  
##  1st Qu.:-1.8817   1st Qu.:0.0185951  
##  Median : 0.1919   Median :0.0922475  
##  Mean   : 0.1919   Mean   :0.1204289  
##  3rd Qu.: 2.2656   3rd Qu.:0.2108783  
##  Max.   : 4.3393   Max.   :0.3281117  
## 
## $tables$PC16$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.5091
## 
##        x                 y            
##  Min.   :-5.9478   Min.   :0.0002521  
##  1st Qu.:-3.0308   1st Qu.:0.0193252  
##  Median :-0.1139   Median :0.0472262  
##  Mean   :-0.1139   Mean   :0.0856149  
##  3rd Qu.: 2.8031   3rd Qu.:0.1278660  
##  Max.   : 5.7200   Max.   :0.3267905  
## 
## $tables$PC16$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.7047
## 
##        x                 y            
##  Min.   :-5.0454   Min.   :0.0001524  
##  1st Qu.:-2.1506   1st Qu.:0.0149089  
##  Median : 0.7443   Median :0.0563422  
##  Mean   : 0.7443   Mean   :0.0862696  
##  3rd Qu.: 3.6392   3rd Qu.:0.1583477  
##  Max.   : 6.5340   Max.   :0.2320700  
## 
## 
## $tables$PC17
## $tables$PC17$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.7258
## 
##        x                 y            
##  Min.   :-8.0721   Min.   :0.0001659  
##  1st Qu.:-4.2201   1st Qu.:0.0078207  
##  Median :-0.3682   Median :0.0282085  
##  Mean   :-0.3682   Mean   :0.0648345  
##  3rd Qu.: 3.4838   3rd Qu.:0.1170452  
##  Max.   : 7.3357   Max.   :0.2232386  
## 
## $tables$PC17$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.5709
## 
##        x                y            
##  Min.   :-4.252   Min.   :0.0001638  
##  1st Qu.:-1.457   1st Qu.:0.0075380  
##  Median : 1.339   Median :0.0451308  
##  Mean   : 1.339   Mean   :0.0893286  
##  3rd Qu.: 4.135   3rd Qu.:0.1393491  
##  Max.   : 6.930   Max.   :0.3030325  
## 
## $tables$PC17$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4508
## 
##        x                 y            
##  Min.   :-3.0535   Min.   :0.0002705  
##  1st Qu.:-1.1679   1st Qu.:0.0223384  
##  Median : 0.7177   Median :0.0908433  
##  Mean   : 0.7177   Mean   :0.1324435  
##  3rd Qu.: 2.6034   3rd Qu.:0.2159647  
##  Max.   : 4.4890   Max.   :0.3885378  
## 
## $tables$PC17$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.4203
## 
##        x                 y           
##  Min.   :-3.2628   Min.   :0.000329  
##  1st Qu.:-1.3975   1st Qu.:0.033891  
##  Median : 0.4678   Median :0.097330  
##  Mean   : 0.4678   Mean   :0.133882  
##  3rd Qu.: 2.3331   3rd Qu.:0.200115  
##  Max.   : 4.1985   Max.   :0.401977  
## 
## $tables$PC17$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3827
## 
##        x                 y            
##  Min.   :-4.5365   Min.   :0.0002741  
##  1st Qu.:-2.8471   1st Qu.:0.0346438  
##  Median :-1.1576   Median :0.0983663  
##  Mean   :-1.1576   Mean   :0.1478187  
##  3rd Qu.: 0.5318   3rd Qu.:0.2793878  
##  Max.   : 2.2213   Max.   :0.4069048  
## 
## 
## $tables$PC18
## $tables$PC18$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4387
## 
##        x                y            
##  Min.   :-8.222   Min.   :0.0000046  
##  1st Qu.:-5.010   1st Qu.:0.0034212  
##  Median :-1.798   Median :0.0173338  
##  Mean   :-1.798   Mean   :0.0777513  
##  3rd Qu.: 1.414   3rd Qu.:0.1144880  
##  Max.   : 4.626   Max.   :0.3676340  
## 
## $tables$PC18$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.7178
## 
##        x                 y            
##  Min.   :-8.3729   Min.   :0.0001308  
##  1st Qu.:-3.9517   1st Qu.:0.0034764  
##  Median : 0.4696   Median :0.0107412  
##  Mean   : 0.4696   Mean   :0.0564872  
##  3rd Qu.: 4.8908   3rd Qu.:0.1075878  
##  Max.   : 9.3120   Max.   :0.2184835  
## 
## $tables$PC18$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4171
## 
##        x                 y           
##  Min.   :-2.9156   Min.   :0.000291  
##  1st Qu.:-1.0025   1st Qu.:0.008772  
##  Median : 0.9107   Median :0.037251  
##  Mean   : 0.9107   Mean   :0.130540  
##  3rd Qu.: 2.8238   3rd Qu.:0.275963  
##  Max.   : 4.7369   Max.   :0.429967  
## 
## $tables$PC18$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.4365
## 
##        x                  y            
##  Min.   :-3.95640   Min.   :0.0002943  
##  1st Qu.:-1.97107   1st Qu.:0.0220699  
##  Median : 0.01427   Median :0.0587911  
##  Mean   : 0.01427   Mean   :0.1257912  
##  3rd Qu.: 1.99960   3rd Qu.:0.2348520  
##  Max.   : 3.98493   Max.   :0.3933910  
## 
## $tables$PC18$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3636
## 
##        x                 y            
##  Min.   :-3.7588   Min.   :0.0002937  
##  1st Qu.:-1.8091   1st Qu.:0.0467648  
##  Median : 0.1407   Median :0.0791231  
##  Mean   : 0.1407   Mean   :0.1280896  
##  3rd Qu.: 2.0904   3rd Qu.:0.1780684  
##  Max.   : 4.0401   Max.   :0.4058116  
## 
## 
## $tables$PC19
## $tables$PC19$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.6435
## 
##        x                y            
##  Min.   :-6.424   Min.   :0.0001301  
##  1st Qu.:-3.475   1st Qu.:0.0147056  
##  Median :-0.527   Median :0.0343235  
##  Mean   :-0.527   Mean   :0.0847065  
##  3rd Qu.: 2.421   3rd Qu.:0.1844546  
##  Max.   : 5.370   Max.   :0.2365631  
## 
## $tables$PC19$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4399
## 
##        x                  y            
##  Min.   :-3.67176   Min.   :0.0002264  
##  1st Qu.:-1.84679   1st Qu.:0.0192997  
##  Median :-0.02183   Median :0.1030698  
##  Mean   :-0.02183   Mean   :0.1368474  
##  3rd Qu.: 1.80314   3rd Qu.:0.2529227  
##  Max.   : 3.62810   Max.   :0.3453398  
## 
## $tables$PC19$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4905
## 
##        x                 y           
##  Min.   :-3.1565   Min.   :0.000311  
##  1st Qu.:-1.2788   1st Qu.:0.023051  
##  Median : 0.5989   Median :0.096221  
##  Mean   : 0.5989   Mean   :0.133000  
##  3rd Qu.: 2.4766   3rd Qu.:0.232399  
##  Max.   : 4.3543   Max.   :0.367442  
## 
## $tables$PC19$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.5901
## 
##        x                 y            
##  Min.   :-4.3005   Min.   :0.0002175  
##  1st Qu.:-1.6966   1st Qu.:0.0159956  
##  Median : 0.9072   Median :0.0581699  
##  Mean   : 0.9072   Mean   :0.0959071  
##  3rd Qu.: 3.5111   3rd Qu.:0.1897102  
##  Max.   : 6.1150   Max.   :0.2602959  
## 
## $tables$PC19$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.6222
## 
##        x                 y            
##  Min.   :-5.6556   Min.   :0.0001723  
##  1st Qu.:-2.9026   1st Qu.:0.0160069  
##  Median :-0.1496   Median :0.0665507  
##  Mean   :-0.1496   Mean   :0.0907165  
##  3rd Qu.: 2.6033   3rd Qu.:0.1478796  
##  Max.   : 5.3563   Max.   :0.2725186  
## 
## 
## $tables$PC20
## $tables$PC20$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4146
## 
##        x                 y            
##  Min.   :-3.3847   Min.   :0.0002016  
##  1st Qu.:-1.5881   1st Qu.:0.0158796  
##  Median : 0.2085   Median :0.0994064  
##  Mean   : 0.2085   Mean   :0.1390114  
##  3rd Qu.: 2.0050   3rd Qu.:0.2545086  
##  Max.   : 3.8016   Max.   :0.3823682  
## 
## $tables$PC20$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4764
## 
##        x                  y            
##  Min.   :-3.72215   Min.   :0.0001966  
##  1st Qu.:-1.85152   1st Qu.:0.0184591  
##  Median : 0.01912   Median :0.1296633  
##  Mean   : 0.01912   Mean   :0.1335006  
##  3rd Qu.: 1.88976   3rd Qu.:0.2472074  
##  Max.   : 3.76040   Max.   :0.2914365  
## 
## $tables$PC20$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4094
## 
##        x                 y            
##  Min.   :-3.0206   Min.   :0.0002958  
##  1st Qu.:-1.3194   1st Qu.:0.0208530  
##  Median : 0.3817   Median :0.0835546  
##  Mean   : 0.3817   Mean   :0.1468066  
##  3rd Qu.: 2.0828   3rd Qu.:0.3084282  
##  Max.   : 3.7840   Max.   :0.3766389  
## 
## $tables$PC20$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.6631
## 
##        x                 y            
##  Min.   :-4.2534   Min.   :0.0002183  
##  1st Qu.:-1.8347   1st Qu.:0.0157516  
##  Median : 0.5839   Median :0.1005571  
##  Mean   : 0.5839   Mean   :0.1032546  
##  3rd Qu.: 3.0025   3rd Qu.:0.1898214  
##  Max.   : 5.4211   Max.   :0.2227499  
## 
## $tables$PC20$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.541
## 
##        x                y            
##  Min.   :-4.800   Min.   :0.0001936  
##  1st Qu.:-1.654   1st Qu.:0.0113630  
##  Median : 1.492   Median :0.0186911  
##  Mean   : 1.492   Mean   :0.0793858  
##  3rd Qu.: 4.638   3rd Qu.:0.1325689  
##  Max.   : 7.784   Max.   :0.2944957  
## 
## 
## $tables$PC21
## $tables$PC21$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3747
## 
##        x                y            
##  Min.   :-7.482   Min.   :0.0000696  
##  1st Qu.:-4.362   1st Qu.:0.0105724  
##  Median :-1.242   Median :0.0245773  
##  Mean   :-1.242   Mean   :0.0800521  
##  3rd Qu.: 1.877   3rd Qu.:0.1058266  
##  Max.   : 4.997   Max.   :0.3903434  
## 
## $tables$PC21$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3489
## 
##        x                 y            
##  Min.   :-3.5496   Min.   :0.0001856  
##  1st Qu.:-1.3710   1st Qu.:0.0076562  
##  Median : 0.8077   Median :0.0462345  
##  Mean   : 0.8077   Mean   :0.1146310  
##  3rd Qu.: 2.9864   3rd Qu.:0.1980177  
##  Max.   : 5.1650   Max.   :0.4440329  
## 
## $tables$PC21$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4123
## 
##        x                 y            
##  Min.   :-3.0594   Min.   :0.0002944  
##  1st Qu.:-1.4514   1st Qu.:0.0212794  
##  Median : 0.1566   Median :0.1279959  
##  Mean   : 0.1566   Mean   :0.1553073  
##  3rd Qu.: 1.7646   3rd Qu.:0.2676239  
##  Max.   : 3.3726   Max.   :0.3962882  
## 
## $tables$PC21$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.4087
## 
##        x                 y            
##  Min.   :-4.0748   Min.   :0.0003361  
##  1st Qu.:-2.3005   1st Qu.:0.0370209  
##  Median :-0.5263   Median :0.0924742  
##  Mean   :-0.5263   Mean   :0.1407546  
##  3rd Qu.: 1.2480   3rd Qu.:0.2458485  
##  Max.   : 3.0222   Max.   :0.3842227  
## 
## $tables$PC21$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.5742
## 
##        x                 y            
##  Min.   :-4.3200   Min.   :0.0001892  
##  1st Qu.:-1.7294   1st Qu.:0.0205262  
##  Median : 0.8613   Median :0.0515645  
##  Mean   : 0.8613   Mean   :0.0963997  
##  3rd Qu.: 3.4519   3rd Qu.:0.1689875  
##  Max.   : 6.0426   Max.   :0.3051189  
## 
## 
## $tables$PC22
## $tables$PC22$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4649
## 
##        x                y           
##  Min.   :-5.987   Min.   :0.000000  
##  1st Qu.:-2.409   1st Qu.:0.001455  
##  Median : 1.169   Median :0.013348  
##  Mean   : 1.169   Mean   :0.069802  
##  3rd Qu.: 4.747   3rd Qu.:0.100710  
##  Max.   : 8.325   Max.   :0.328626  
## 
## $tables$PC22$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4513
## 
##        x                y            
##  Min.   :-3.436   Min.   :0.0002068  
##  1st Qu.:-1.002   1st Qu.:0.0077952  
##  Median : 1.432   Median :0.0320024  
##  Mean   : 1.432   Mean   :0.1026077  
##  3rd Qu.: 3.866   3rd Qu.:0.2037061  
##  Max.   : 6.300   Max.   :0.3558547  
## 
## $tables$PC22$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4438
## 
##        x                 y            
##  Min.   :-3.1090   Min.   :0.0003255  
##  1st Qu.:-1.4220   1st Qu.:0.0235237  
##  Median : 0.2649   Median :0.1277396  
##  Mean   : 0.2649   Mean   :0.1480383  
##  3rd Qu.: 1.9519   3rd Qu.:0.2637109  
##  Max.   : 3.6389   Max.   :0.3525356  
## 
## $tables$PC22$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3586
## 
##        x                 y            
##  Min.   :-2.7661   Min.   :0.0003662  
##  1st Qu.:-1.4453   1st Qu.:0.0261486  
##  Median :-0.1245   Median :0.1834466  
##  Mean   :-0.1245   Mean   :0.1890787  
##  3rd Qu.: 1.1963   3rd Qu.:0.3593836  
##  Max.   : 2.5171   Max.   :0.4025294  
## 
## $tables$PC22$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.5293
## 
##        x                y            
##  Min.   :-3.838   Min.   :0.0001974  
##  1st Qu.:-1.130   1st Qu.:0.0064596  
##  Median : 1.579   Median :0.0372029  
##  Mean   : 1.579   Mean   :0.0921977  
##  3rd Qu.: 4.288   3rd Qu.:0.1748484  
##  Max.   : 6.997   Max.   :0.3071155  
## 
## 
## $tables$PC23
## $tables$PC23$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4716
## 
##        x                 y            
##  Min.   :-4.7565   Min.   :0.0001768  
##  1st Qu.:-2.0784   1st Qu.:0.0079228  
##  Median : 0.5997   Median :0.0286808  
##  Mean   : 0.5997   Mean   :0.0932543  
##  3rd Qu.: 3.2778   3rd Qu.:0.1669899  
##  Max.   : 5.9558   Max.   :0.3609530  
## 
## $tables$PC23$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4347
## 
##        x                 y            
##  Min.   :-3.0888   Min.   :0.0000014  
##  1st Qu.:-0.1862   1st Qu.:0.0023170  
##  Median : 2.7164   Median :0.0172638  
##  Mean   : 2.7164   Mean   :0.0860399  
##  3rd Qu.: 5.6190   3rd Qu.:0.1625756  
##  Max.   : 8.5216   Max.   :0.3789807  
## 
## $tables$PC23$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2721
## 
##        x                 y            
##  Min.   :-2.5507   Min.   :0.0004499  
##  1st Qu.:-1.4688   1st Qu.:0.0351359  
##  Median :-0.3869   Median :0.1574924  
##  Mean   :-0.3869   Mean   :0.2308308  
##  3rd Qu.: 0.6950   3rd Qu.:0.4675236  
##  Max.   : 1.7769   Max.   :0.5418532  
## 
## $tables$PC23$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.371
## 
##        x                 y            
##  Min.   :-3.1671   Min.   :0.0003795  
##  1st Qu.:-1.6986   1st Qu.:0.0353980  
##  Median :-0.2301   Median :0.0975453  
##  Mean   :-0.2301   Mean   :0.1700564  
##  3rd Qu.: 1.2384   3rd Qu.:0.3010342  
##  Max.   : 2.7069   Max.   :0.4573137  
## 
## $tables$PC23$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.5667
## 
##        x                 y            
##  Min.   :-4.8659   Min.   :0.0001923  
##  1st Qu.:-2.3732   1st Qu.:0.0174573  
##  Median : 0.1194   Median :0.0786440  
##  Mean   : 0.1194   Mean   :0.1001899  
##  3rd Qu.: 2.6121   3rd Qu.:0.1655783  
##  Max.   : 5.1048   Max.   :0.2718955  
## 
## 
## $tables$PC24
## $tables$PC24$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.372
## 
##        x                 y            
##  Min.   :-3.2266   Min.   :0.0002881  
##  1st Qu.:-1.5677   1st Qu.:0.0361872  
##  Median : 0.0911   Median :0.1134137  
##  Mean   : 0.0911   Mean   :0.1505486  
##  3rd Qu.: 1.7499   3rd Qu.:0.2455262  
##  Max.   : 3.4088   Max.   :0.4154736  
## 
## $tables$PC24$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4407
## 
##        x                 y            
##  Min.   :-3.4020   Min.   :0.0002342  
##  1st Qu.:-1.5881   1st Qu.:0.0229747  
##  Median : 0.2257   Median :0.1210754  
##  Mean   : 0.2257   Mean   :0.1376848  
##  3rd Qu.: 2.0396   3rd Qu.:0.2323374  
##  Max.   : 3.8534   Max.   :0.3472528  
## 
## $tables$PC24$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4516
## 
##        x                 y            
##  Min.   :-3.9082   Min.   :0.0002687  
##  1st Qu.:-2.1349   1st Qu.:0.0207108  
##  Median :-0.3616   Median :0.1082647  
##  Mean   :-0.3616   Mean   :0.1408319  
##  3rd Qu.: 1.4117   3rd Qu.:0.2512567  
##  Max.   : 3.1849   Max.   :0.3637956  
## 
## $tables$PC24$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.4547
## 
##        x                y            
##  Min.   :-3.876   Min.   :0.0002817  
##  1st Qu.:-2.000   1st Qu.:0.0211126  
##  Median :-0.123   Median :0.0781154  
##  Mean   :-0.123   Mean   :0.1330686  
##  3rd Qu.: 1.754   3rd Qu.:0.2496668  
##  Max.   : 3.630   Max.   :0.3798078  
## 
## $tables$PC24$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.592
## 
##        x                y            
##  Min.   :-4.725   Min.   :0.0001764  
##  1st Qu.:-1.655   1st Qu.:0.0063071  
##  Median : 1.414   Median :0.0432318  
##  Mean   : 1.414   Mean   :0.0813575  
##  3rd Qu.: 4.484   3rd Qu.:0.1426149  
##  Max.   : 7.553   Max.   :0.2479926  
## 
## 
## $tables$PC25
## $tables$PC25$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4154
## 
##        x                y            
##  Min.   :-3.486   Min.   :0.0001703  
##  1st Qu.:-1.148   1st Qu.:0.0054676  
##  Median : 1.190   Median :0.0447529  
##  Mean   : 1.190   Mean   :0.1068229  
##  3rd Qu.: 3.528   3rd Qu.:0.1829505  
##  Max.   : 5.865   Max.   :0.3531702  
## 
## $tables$PC25$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4677
## 
##        x                y            
##  Min.   :-3.187   Min.   :0.0001997  
##  1st Qu.:-1.178   1st Qu.:0.0194962  
##  Median : 0.831   Median :0.0654617  
##  Mean   : 0.831   Mean   :0.1242975  
##  3rd Qu.: 2.840   3rd Qu.:0.2522098  
##  Max.   : 4.849   Max.   :0.3346188  
## 
## $tables$PC25$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3819
## 
##        x                 y            
##  Min.   :-3.4521   Min.   :0.0003434  
##  1st Qu.:-1.8745   1st Qu.:0.0308125  
##  Median :-0.2969   Median :0.1261533  
##  Mean   :-0.2969   Mean   :0.1583026  
##  3rd Qu.: 1.2806   3rd Qu.:0.2694833  
##  Max.   : 2.8582   Max.   :0.4151071  
## 
## $tables$PC25$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3744
## 
##        x                 y            
##  Min.   :-4.0278   Min.   :0.0003668  
##  1st Qu.:-2.3135   1st Qu.:0.0361793  
##  Median :-0.5993   Median :0.0611527  
##  Mean   :-0.5993   Mean   :0.1456747  
##  3rd Qu.: 1.1150   3rd Qu.:0.2889551  
##  Max.   : 2.8292   Max.   :0.4129477  
## 
## $tables$PC25$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.54
## 
##        x                 y           
##  Min.   :-5.3685   Min.   :0.000193  
##  1st Qu.:-2.9563   1st Qu.:0.016283  
##  Median :-0.5442   Median :0.046221  
##  Mean   :-0.5442   Mean   :0.103535  
##  3rd Qu.: 1.8679   3rd Qu.:0.223272  
##  Max.   : 4.2801   Max.   :0.290415  
## 
## 
## $tables$PC26
## $tables$PC26$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4316
## 
##        x                 y            
##  Min.   :-4.2994   Min.   :0.0001923  
##  1st Qu.:-2.3698   1st Qu.:0.0185310  
##  Median :-0.4401   Median :0.0987663  
##  Mean   :-0.4401   Mean   :0.1294202  
##  3rd Qu.: 1.4896   3rd Qu.:0.2276098  
##  Max.   : 3.4193   Max.   :0.3438211  
## 
## $tables$PC26$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4277
## 
##        x                 y            
##  Min.   :-2.9006   Min.   :0.0002186  
##  1st Qu.:-0.9266   1st Qu.:0.0099452  
##  Median : 1.0474   Median :0.0540632  
##  Mean   : 1.0474   Mean   :0.1265162  
##  3rd Qu.: 3.0214   3rd Qu.:0.2728915  
##  Max.   : 4.9954   Max.   :0.3593559  
## 
## $tables$PC26$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3414
## 
##        x                  y            
##  Min.   :-3.00282   Min.   :0.0003545  
##  1st Qu.:-1.53618   1st Qu.:0.0244360  
##  Median :-0.06954   Median :0.0902818  
##  Mean   :-0.06954   Mean   :0.1702793  
##  3rd Qu.: 1.39710   3rd Qu.:0.3443617  
##  Max.   : 2.86374   Max.   :0.4523068  
## 
## $tables$PC26$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3782
## 
##        x                 y            
##  Min.   :-3.2566   Min.   :0.0003391  
##  1st Qu.:-1.5419   1st Qu.:0.0262904  
##  Median : 0.1728   Median :0.0957796  
##  Mean   : 0.1728   Mean   :0.1456445  
##  3rd Qu.: 1.8875   3rd Qu.:0.2539764  
##  Max.   : 3.6022   Max.   :0.4255996  
## 
## $tables$PC26$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.5102
## 
##        x                 y            
##  Min.   :-6.4599   Min.   :0.0002054  
##  1st Qu.:-3.3077   1st Qu.:0.0103717  
##  Median :-0.1556   Median :0.0187623  
##  Mean   :-0.1556   Mean   :0.0792292  
##  3rd Qu.: 2.9965   3rd Qu.:0.1332915  
##  Max.   : 6.1487   Max.   :0.3690152  
## 
## 
## $tables$PC27
## $tables$PC27$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3324
## 
##        x                 y            
##  Min.   :-3.9994   Min.   :0.0002515  
##  1st Qu.:-2.1395   1st Qu.:0.0364197  
##  Median :-0.2796   Median :0.0775675  
##  Mean   :-0.2796   Mean   :0.1342769  
##  3rd Qu.: 1.5802   3rd Qu.:0.1868555  
##  Max.   : 3.4401   Max.   :0.4470101  
## 
## $tables$PC27$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4372
## 
##        x                 y            
##  Min.   :-5.1064   Min.   :0.0002144  
##  1st Qu.:-2.7919   1st Qu.:0.0193638  
##  Median :-0.4773   Median :0.0531151  
##  Mean   :-0.4773   Mean   :0.1079017  
##  3rd Qu.: 1.8372   3rd Qu.:0.1819963  
##  Max.   : 4.1517   Max.   :0.3485958  
## 
## $tables$PC27$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2302
## 
##        x                 y            
##  Min.   :-2.1410   Min.   :0.0005273  
##  1st Qu.:-0.9563   1st Qu.:0.0483829  
##  Median : 0.2284   Median :0.1175649  
##  Mean   : 0.2284   Mean   :0.2108006  
##  3rd Qu.: 1.4131   3rd Qu.:0.3572428  
##  Max.   : 2.5978   Max.   :0.6644116  
## 
## $tables$PC27$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3332
## 
##        x                 y            
##  Min.   :-3.6891   Min.   :0.0003858  
##  1st Qu.:-2.1943   1st Qu.:0.0355780  
##  Median :-0.6995   Median :0.1218695  
##  Mean   :-0.6995   Mean   :0.1670694  
##  3rd Qu.: 0.7953   3rd Qu.:0.2938796  
##  Max.   : 2.2901   Max.   :0.4531308  
## 
## $tables$PC27$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.4113
## 
##        x                 y            
##  Min.   :-3.2554   Min.   :0.0002545  
##  1st Qu.:-0.9233   1st Qu.:0.0074550  
##  Median : 1.4087   Median :0.0430111  
##  Mean   : 1.4087   Mean   :0.1070893  
##  3rd Qu.: 3.7408   3rd Qu.:0.1825964  
##  Max.   : 6.0729   Max.   :0.3890901  
## 
## 
## $tables$PC28
## $tables$PC28$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2304
## 
##        x                 y            
##  Min.   :-3.9410   Min.   :0.0003615  
##  1st Qu.:-2.3749   1st Qu.:0.0239873  
##  Median :-0.8089   Median :0.0736820  
##  Mean   :-0.8089   Mean   :0.1594738  
##  3rd Qu.: 0.7572   3rd Qu.:0.2650858  
##  Max.   : 2.3232   Max.   :0.6203895  
## 
## $tables$PC28$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4063
## 
##        x                 y            
##  Min.   :-4.7727   Min.   :0.0002307  
##  1st Qu.:-2.2983   1st Qu.:0.0194838  
##  Median : 0.1761   Median :0.0498458  
##  Mean   : 0.1761   Mean   :0.1009309  
##  3rd Qu.: 2.6505   3rd Qu.:0.1443150  
##  Max.   : 5.1249   Max.   :0.3750931  
## 
## $tables$PC28$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3099
## 
##        x                  y            
##  Min.   :-2.41405   Min.   :0.0003937  
##  1st Qu.:-1.15969   1st Qu.:0.0296415  
##  Median : 0.09468   Median :0.1404407  
##  Mean   : 0.09468   Mean   :0.1990943  
##  3rd Qu.: 1.34904   3rd Qu.:0.3467915  
##  Max.   : 2.60341   Max.   :0.5718956  
## 
## $tables$PC28$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2586
## 
##        x                 y            
##  Min.   :-3.2620   Min.   :0.0004958  
##  1st Qu.:-1.8895   1st Qu.:0.0373587  
##  Median :-0.5169   Median :0.1221300  
##  Mean   :-0.5169   Mean   :0.1819483  
##  3rd Qu.: 0.8556   3rd Qu.:0.2779809  
##  Max.   : 2.2282   Max.   :0.5499589  
## 
## $tables$PC28$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.5261
## 
##        x                 y            
##  Min.   :-5.1635   Min.   :0.0001987  
##  1st Qu.:-3.0085   1st Qu.:0.0190888  
##  Median :-0.8536   Median :0.0639797  
##  Mean   :-0.8536   Mean   :0.1158864  
##  3rd Qu.: 1.3014   3rd Qu.:0.2354937  
##  Max.   : 3.4564   Max.   :0.2926946  
## 
## 
## $tables$PC29
## $tables$PC29$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3901
## 
##        x                 y            
##  Min.   :-3.7291   Min.   :0.0002127  
##  1st Qu.:-1.9815   1st Qu.:0.0160634  
##  Median :-0.2338   Median :0.0925643  
##  Mean   :-0.2338   Mean   :0.1429050  
##  3rd Qu.: 1.5138   3rd Qu.:0.2696054  
##  Max.   : 3.2614   Max.   :0.3910725  
## 
## $tables$PC29$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3857
## 
##        x                 y            
##  Min.   :-2.7773   Min.   :0.0002424  
##  1st Qu.:-1.0183   1st Qu.:0.0243164  
##  Median : 0.7407   Median :0.0865907  
##  Mean   : 0.7407   Mean   :0.1419783  
##  3rd Qu.: 2.4997   3rd Qu.:0.2732362  
##  Max.   : 4.2587   Max.   :0.3706880  
## 
## $tables$PC29$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.1844
## 
##        x                 y            
##  Min.   :-1.9260   Min.   :0.0000822  
##  1st Qu.:-0.6756   1st Qu.:0.0155918  
##  Median : 0.5749   Median :0.1397487  
##  Mean   : 0.5749   Mean   :0.1997178  
##  3rd Qu.: 1.8253   3rd Qu.:0.2528046  
##  Max.   : 3.0758   Max.   :0.8932048  
## 
## $tables$PC29$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.5407
## 
##        x                 y            
##  Min.   :-4.7351   Min.   :0.0002917  
##  1st Qu.:-2.6814   1st Qu.:0.0204682  
##  Median :-0.6277   Median :0.0887104  
##  Mean   :-0.6277   Mean   :0.1216016  
##  3rd Qu.: 1.4260   3rd Qu.:0.2295617  
##  Max.   : 3.4797   Max.   :0.2956780  
## 
## $tables$PC29$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.4605
## 
##        x                 y            
##  Min.   :-2.9540   Min.   :0.0002292  
##  1st Qu.:-1.0889   1st Qu.:0.0200894  
##  Median : 0.7761   Median :0.0764365  
##  Mean   : 0.7761   Mean   :0.1339017  
##  3rd Qu.: 2.6412   3rd Qu.:0.2655154  
##  Max.   : 4.5063   Max.   :0.3336613  
## 
## 
## $tables$PC30
## $tables$PC30$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4887
## 
##        x                 y          
##  Min.   :-5.3247   Min.   :0.00017  
##  1st Qu.:-3.0968   1st Qu.:0.01630  
##  Median :-0.8689   Median :0.07104  
##  Mean   :-0.8689   Mean   :0.11210  
##  3rd Qu.: 1.3590   3rd Qu.:0.20113  
##  Max.   : 3.5869   Max.   :0.33071  
## 
## $tables$PC30$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4564
## 
##        x                  y            
##  Min.   :-3.41040   Min.   :0.0003156  
##  1st Qu.:-1.72157   1st Qu.:0.0248083  
##  Median :-0.03273   Median :0.1633324  
##  Mean   :-0.03273   Mean   :0.1478742  
##  3rd Qu.: 1.65611   3rd Qu.:0.2209157  
##  Max.   : 3.34494   Max.   :0.3671845  
## 
## $tables$PC30$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2938
## 
##        x                 y            
##  Min.   :-2.9486   Min.   :0.0004132  
##  1st Qu.:-1.3113   1st Qu.:0.0321249  
##  Median : 0.3261   Median :0.0827284  
##  Mean   : 0.3261   Mean   :0.1525261  
##  3rd Qu.: 1.9634   3rd Qu.:0.2336990  
##  Max.   : 3.6008   Max.   :0.5606675  
## 
## $tables$PC30$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.23
## 
##        x                 y            
##  Min.   :-1.9886   Min.   :0.0005797  
##  1st Qu.:-0.9402   1st Qu.:0.0584064  
##  Median : 0.1081   Median :0.1683684  
##  Mean   : 0.1081   Mean   :0.2382220  
##  3rd Qu.: 1.1564   3rd Qu.:0.3967196  
##  Max.   : 2.2048   Max.   :0.6702072  
## 
## $tables$PC30$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.4201
## 
##        x                 y           
##  Min.   :-4.1211   Min.   :0.000264  
##  1st Qu.:-2.3879   1st Qu.:0.024272  
##  Median :-0.6546   Median :0.074721  
##  Mean   :-0.6546   Mean   :0.144087  
##  3rd Qu.: 1.0786   3rd Qu.:0.274435  
##  Max.   : 2.8119   Max.   :0.399436  
## 
## 
## $tables$PC31
## $tables$PC31$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3528
## 
##        x                 y            
##  Min.   :-3.3004   Min.   :0.0002456  
##  1st Qu.:-1.7180   1st Qu.:0.0328899  
##  Median :-0.1356   Median :0.1163266  
##  Mean   :-0.1356   Mean   :0.1578186  
##  3rd Qu.: 1.4468   3rd Qu.:0.2881942  
##  Max.   : 3.0292   Max.   :0.3975180  
## 
## $tables$PC31$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3869
## 
##        x                 y            
##  Min.   :-3.1836   Min.   :0.0002421  
##  1st Qu.:-1.5359   1st Qu.:0.0185548  
##  Median : 0.1117   Median :0.1100221  
##  Mean   : 0.1117   Mean   :0.1515762  
##  3rd Qu.: 1.7593   3rd Qu.:0.2544227  
##  Max.   : 3.4070   Max.   :0.4307679  
## 
## $tables$PC31$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3582
## 
##        x                 y           
##  Min.   :-2.4440   Min.   :0.000339  
##  1st Qu.:-0.9986   1st Qu.:0.026864  
##  Median : 0.4468   Median :0.103603  
##  Mean   : 0.4468   Mean   :0.172783  
##  3rd Qu.: 1.8921   3rd Qu.:0.333369  
##  Max.   : 3.3375   Max.   :0.446947  
## 
## $tables$PC31$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.381
## 
##        x                 y            
##  Min.   :-3.5972   Min.   :0.0003384  
##  1st Qu.:-1.9857   1st Qu.:0.0313721  
##  Median :-0.3741   Median :0.1003788  
##  Mean   :-0.3741   Mean   :0.1549631  
##  3rd Qu.: 1.2375   3rd Qu.:0.2726520  
##  Max.   : 2.8490   Max.   :0.4481048  
## 
## $tables$PC31$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.5145
## 
##        x                 y            
##  Min.   :-3.5254   Min.   :0.0002044  
##  1st Qu.:-1.4720   1st Qu.:0.0196060  
##  Median : 0.5815   Median :0.0913548  
##  Mean   : 0.5815   Mean   :0.1216181  
##  3rd Qu.: 2.6349   3rd Qu.:0.2316572  
##  Max.   : 4.6884   Max.   :0.2978902  
## 
## 
## $tables$PC32
## $tables$PC32$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3227
## 
##        x                y            
##  Min.   :-3.347   Min.   :0.0002581  
##  1st Qu.:-1.622   1st Qu.:0.0210878  
##  Median : 0.102   Median :0.0549429  
##  Mean   : 0.102   Mean   :0.1448391  
##  3rd Qu.: 1.826   3rd Qu.:0.2577693  
##  Max.   : 3.551   Max.   :0.4919757  
## 
## $tables$PC32$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4634
## 
##        x                  y            
##  Min.   :-3.52998   Min.   :0.0002762  
##  1st Qu.:-1.78213   1st Qu.:0.0229666  
##  Median :-0.03428   Median :0.1536946  
##  Mean   :-0.03428   Mean   :0.1428826  
##  3rd Qu.: 1.71357   3rd Qu.:0.2394299  
##  Max.   : 3.46143   Max.   :0.3066883  
## 
## $tables$PC32$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3153
## 
##        x                 y            
##  Min.   :-3.0882   Min.   :0.0007283  
##  1st Qu.:-1.6987   1st Qu.:0.0555632  
##  Median :-0.3091   Median :0.1278530  
##  Mean   :-0.3091   Mean   :0.1797114  
##  3rd Qu.: 1.0805   3rd Qu.:0.2957054  
##  Max.   : 2.4701   Max.   :0.4645175  
## 
## $tables$PC32$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3733
## 
##        x                 y            
##  Min.   :-3.6262   Min.   :0.0003431  
##  1st Qu.:-2.1553   1st Qu.:0.0259800  
##  Median :-0.6843   Median :0.1314569  
##  Mean   :-0.6843   Mean   :0.1697744  
##  3rd Qu.: 0.7866   3rd Qu.:0.2963796  
##  Max.   : 2.2576   Max.   :0.4579900  
## 
## $tables$PC32$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.333
## 
##        x                 y            
##  Min.   :-4.2154   Min.   :0.0003142  
##  1st Qu.:-2.3673   1st Qu.:0.0110793  
##  Median :-0.5192   Median :0.0538098  
##  Mean   :-0.5192   Mean   :0.1351353  
##  3rd Qu.: 1.3289   3rd Qu.:0.2549752  
##  Max.   : 3.1769   Max.   :0.4534647  
## 
## 
## $tables$PC33
## $tables$PC33$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2975
## 
##        x                 y           
##  Min.   :-2.6617   Min.   :0.000283  
##  1st Qu.:-1.2313   1st Qu.:0.028841  
##  Median : 0.1992   Median :0.097924  
##  Mean   : 0.1992   Mean   :0.174595  
##  3rd Qu.: 1.6296   3rd Qu.:0.316409  
##  Max.   : 3.0600   Max.   :0.453979  
## 
## $tables$PC33$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4422
## 
##        x                 y            
##  Min.   :-4.1546   Min.   :0.0002195  
##  1st Qu.:-2.2364   1st Qu.:0.0201662  
##  Median :-0.3183   Median :0.0741957  
##  Mean   :-0.3183   Mean   :0.1302012  
##  3rd Qu.: 1.5998   3rd Qu.:0.2436597  
##  Max.   : 3.5179   Max.   :0.3664393  
## 
## $tables$PC33$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3582
## 
##        x                 y            
##  Min.   :-3.4148   Min.   :0.0003384  
##  1st Qu.:-1.8256   1st Qu.:0.0271054  
##  Median :-0.2363   Median :0.0847193  
##  Mean   :-0.2363   Mean   :0.1571397  
##  3rd Qu.: 1.3530   3rd Qu.:0.2901437  
##  Max.   : 2.9422   Max.   :0.4723171  
## 
## $tables$PC33$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3416
## 
##        x                 y            
##  Min.   :-4.1337   Min.   :0.0003756  
##  1st Qu.:-2.5761   1st Qu.:0.0331357  
##  Median :-1.0185   Median :0.0615801  
##  Mean   :-1.0185   Mean   :0.1603339  
##  3rd Qu.: 0.5391   3rd Qu.:0.3193731  
##  Max.   : 2.0967   Max.   :0.5038831  
## 
## $tables$PC33$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3884
## 
##        x                 y            
##  Min.   :-2.3231   Min.   :0.0002818  
##  1st Qu.:-0.8366   1st Qu.:0.0271891  
##  Median : 0.6499   Median :0.1359094  
##  Mean   : 0.6499   Mean   :0.1680014  
##  3rd Qu.: 2.1364   3rd Qu.:0.3108160  
##  Max.   : 3.6229   Max.   :0.3799684  
## 
## 
## $tables$PC34
## $tables$PC34$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.367
## 
##        x                 y            
##  Min.   :-2.8980   Min.   :0.0003311  
##  1st Qu.:-1.3217   1st Qu.:0.0321450  
##  Median : 0.2547   Median :0.1039471  
##  Mean   : 0.2547   Mean   :0.1584314  
##  3rd Qu.: 1.8310   3rd Qu.:0.2937437  
##  Max.   : 3.4073   Max.   :0.4020532  
## 
## $tables$PC34$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.2979
## 
##        x                 y            
##  Min.   :-2.3880   Min.   :0.0003176  
##  1st Qu.:-1.1422   1st Qu.:0.0295054  
##  Median : 0.1037   Median :0.1342770  
##  Mean   : 0.1037   Mean   :0.2004557  
##  3rd Qu.: 1.3495   3rd Qu.:0.3247393  
##  Max.   : 2.5954   Max.   :0.6099827  
## 
## $tables$PC34$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3486
## 
##        x                  y            
##  Min.   :-2.56726   Min.   :0.0007337  
##  1st Qu.:-1.30716   1st Qu.:0.0480069  
##  Median :-0.04706   Median :0.1942632  
##  Mean   :-0.04706   Mean   :0.1981697  
##  3rd Qu.: 1.21304   3rd Qu.:0.3536422  
##  Max.   : 2.47314   Max.   :0.4191529  
## 
## $tables$PC34$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3833
## 
##        x                  y            
##  Min.   :-3.34185   Min.   :0.0003345  
##  1st Qu.:-1.71849   1st Qu.:0.0243797  
##  Median :-0.09514   Median :0.1053480  
##  Mean   :-0.09514   Mean   :0.1538400  
##  3rd Qu.: 1.52822   3rd Qu.:0.2841954  
##  Max.   : 3.15157   Max.   :0.4446783  
## 
## $tables$PC34$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3971
## 
##        x                 y            
##  Min.   :-3.8648   Min.   :0.0002628  
##  1st Qu.:-1.7162   1st Qu.:0.0116518  
##  Median : 0.4323   Median :0.0445868  
##  Mean   : 0.4323   Mean   :0.1162381  
##  3rd Qu.: 2.5808   3rd Qu.:0.2306475  
##  Max.   : 4.7294   Max.   :0.3572020  
## 
## 
## $tables$PC35
## $tables$PC35$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3184
## 
##        x                 y            
##  Min.   :-3.1839   Min.   :0.0002613  
##  1st Qu.:-1.6606   1st Qu.:0.0215315  
##  Median :-0.1373   Median :0.1004420  
##  Mean   :-0.1373   Mean   :0.1639528  
##  3rd Qu.: 1.3859   3rd Qu.:0.2927211  
##  Max.   : 2.9092   Max.   :0.4894221  
## 
## $tables$PC35$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3183
## 
##        x                 y            
##  Min.   :-2.3024   Min.   :0.0005812  
##  1st Qu.:-1.0714   1st Qu.:0.0512006  
##  Median : 0.1597   Median :0.1664123  
##  Mean   : 0.1597   Mean   :0.2028563  
##  3rd Qu.: 1.3907   3rd Qu.:0.3403001  
##  Max.   : 2.6218   Max.   :0.4932819  
## 
## $tables$PC35$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.382
## 
##        x                 y            
##  Min.   :-3.6158   Min.   :0.0003174  
##  1st Qu.:-1.8871   1st Qu.:0.0236867  
##  Median :-0.1585   Median :0.0687381  
##  Mean   :-0.1585   Mean   :0.1444702  
##  3rd Qu.: 1.5702   3rd Qu.:0.2594633  
##  Max.   : 3.2988   Max.   :0.4487296  
## 
## $tables$PC35$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.354
## 
##        x                 y           
##  Min.   :-2.2690   Min.   :0.000375  
##  1st Qu.:-0.8911   1st Qu.:0.028883  
##  Median : 0.4868   Median :0.117729  
##  Mean   : 0.4868   Mean   :0.181242  
##  3rd Qu.: 1.8647   3rd Qu.:0.321967  
##  Max.   : 3.2425   Max.   :0.491458  
## 
## $tables$PC35$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.4043
## 
##        x                 y            
##  Min.   :-3.7329   Min.   :0.0002591  
##  1st Qu.:-1.8143   1st Qu.:0.0224158  
##  Median : 0.1043   Median :0.0882488  
##  Mean   : 0.1043   Mean   :0.1301659  
##  3rd Qu.: 2.0230   3rd Qu.:0.2172968  
##  Max.   : 3.9416   Max.   :0.3863378  
## 
## 
## $tables$PC36
## $tables$PC36$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2665
## 
##        x                 y            
##  Min.   :-3.1334   Min.   :0.0003496  
##  1st Qu.:-1.8026   1st Qu.:0.0184875  
##  Median :-0.4718   Median :0.0682130  
##  Mean   :-0.4718   Mean   :0.1876603  
##  3rd Qu.: 0.8590   3rd Qu.:0.3864051  
##  Max.   : 2.1898   Max.   :0.5495498  
## 
## $tables$PC36$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3211
## 
##        x                 y            
##  Min.   :-2.3840   Min.   :0.0002992  
##  1st Qu.:-1.0118   1st Qu.:0.0348900  
##  Median : 0.3604   Median :0.1408627  
##  Mean   : 0.3604   Mean   :0.1819936  
##  3rd Qu.: 1.7327   3rd Qu.:0.3162975  
##  Max.   : 3.1049   Max.   :0.4712329  
## 
## $tables$PC36$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3796
## 
##        x                 y          
##  Min.   :-3.9795   Min.   :0.00032  
##  1st Qu.:-2.2320   1st Qu.:0.01865  
##  Median :-0.4845   Median :0.07714  
##  Mean   :-0.4845   Mean   :0.14291  
##  3rd Qu.: 1.2629   3rd Qu.:0.26373  
##  Max.   : 3.0104   Max.   :0.43469  
## 
## $tables$PC36$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3371
## 
##        x                 y            
##  Min.   :-3.2810   Min.   :0.0003805  
##  1st Qu.:-1.8155   1st Qu.:0.0258825  
##  Median :-0.3501   Median :0.0934744  
##  Mean   :-0.3501   Mean   :0.1704163  
##  3rd Qu.: 1.1154   3rd Qu.:0.2876202  
##  Max.   : 2.5808   Max.   :0.5474941  
## 
## $tables$PC36$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.4149
## 
##        x                  y            
##  Min.   :-4.20394   Min.   :0.0002517  
##  1st Qu.:-2.11584   1st Qu.:0.0158776  
##  Median :-0.02775   Median :0.0473145  
##  Mean   :-0.02775   Mean   :0.1196019  
##  3rd Qu.: 2.06035   3rd Qu.:0.2340138  
##  Max.   : 4.14845   Max.   :0.3811832  
## 
## 
## $tables$PC37
## $tables$PC37$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3706
## 
##        x                  y            
##  Min.   :-3.15419   Min.   :0.0002771  
##  1st Qu.:-1.56790   1st Qu.:0.0269925  
##  Median : 0.01839   Median :0.0980281  
##  Mean   : 0.01839   Mean   :0.1574354  
##  3rd Qu.: 1.60468   3rd Qu.:0.3082809  
##  Max.   : 3.19097   Max.   :0.4051372  
## 
## $tables$PC37$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3076
## 
##        x                 y            
##  Min.   :-2.6393   Min.   :0.0003045  
##  1st Qu.:-1.1478   1st Qu.:0.0183850  
##  Median : 0.3437   Median :0.0979149  
##  Mean   : 0.3437   Mean   :0.1674378  
##  3rd Qu.: 1.8352   3rd Qu.:0.3185252  
##  Max.   : 3.3268   Max.   :0.4713728  
## 
## $tables$PC37$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3365
## 
##        x                 y            
##  Min.   :-3.4672   Min.   :0.0003595  
##  1st Qu.:-1.8787   1st Qu.:0.0211283  
##  Median :-0.2902   Median :0.1053916  
##  Mean   :-0.2902   Mean   :0.1572195  
##  3rd Qu.: 1.2982   3rd Qu.:0.2833600  
##  Max.   : 2.8867   Max.   :0.4745564  
## 
## $tables$PC37$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2931
## 
##        x                 y            
##  Min.   :-2.5908   Min.   :0.0004414  
##  1st Qu.:-1.2002   1st Qu.:0.0434481  
##  Median : 0.1905   Median :0.0932578  
##  Mean   : 0.1905   Mean   :0.1795861  
##  3rd Qu.: 1.5811   3rd Qu.:0.3181929  
##  Max.   : 2.9717   Max.   :0.5247508  
## 
## $tables$PC37$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3763
## 
##        x                  y            
##  Min.   :-2.88309   Min.   :0.0003314  
##  1st Qu.:-1.39235   1st Qu.:0.0275781  
##  Median : 0.09839   Median :0.1397838  
##  Mean   : 0.09839   Mean   :0.1675258  
##  3rd Qu.: 1.58914   3rd Qu.:0.2734887  
##  Max.   : 3.07988   Max.   :0.4625233  
## 
## 
## $tables$PC38
## $tables$PC38$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2847
## 
##        x                 y            
##  Min.   :-2.5889   Min.   :0.0002929  
##  1st Qu.:-1.0422   1st Qu.:0.0257167  
##  Median : 0.5045   Median :0.0971432  
##  Mean   : 0.5045   Mean   :0.1614718  
##  3rd Qu.: 2.0511   3rd Qu.:0.2739791  
##  Max.   : 3.5978   Max.   :0.5663911  
## 
## $tables$PC38$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.2541
## 
##        x                y            
##  Min.   :-3.836   Min.   :0.0003687  
##  1st Qu.:-2.338   1st Qu.:0.0141007  
##  Median :-0.840   Median :0.1033128  
##  Mean   :-0.840   Mean   :0.1667106  
##  3rd Qu.: 0.658   3rd Qu.:0.2743645  
##  Max.   : 2.156   Max.   :0.5858621  
## 
## $tables$PC38$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3084
## 
##        x                 y            
##  Min.   :-2.6761   Min.   :0.0006322  
##  1st Qu.:-1.4672   1st Qu.:0.0476289  
##  Median :-0.2584   Median :0.1257733  
##  Mean   :-0.2584   Mean   :0.2065832  
##  3rd Qu.: 0.9505   3rd Qu.:0.3983943  
##  Max.   : 2.1593   Max.   :0.4939718  
## 
## $tables$PC38$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3801
## 
##        x                 y            
##  Min.   :-3.0572   Min.   :0.0003685  
##  1st Qu.:-1.5905   1st Qu.:0.0261578  
##  Median :-0.1238   Median :0.1304873  
##  Mean   :-0.1238   Mean   :0.1702689  
##  3rd Qu.: 1.3429   3rd Qu.:0.2869042  
##  Max.   : 2.8096   Max.   :0.4720783  
## 
## $tables$PC38$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.2553
## 
##        x                 y            
##  Min.   :-2.1429   Min.   :0.0004118  
##  1st Qu.:-0.8884   1st Qu.:0.0474870  
##  Median : 0.3662   Median :0.1018735  
##  Mean   : 0.3662   Mean   :0.1990708  
##  3rd Qu.: 1.6207   3rd Qu.:0.3223509  
##  Max.   : 2.8752   Max.   :0.5793865  
## 
## 
## $tables$PC39
## $tables$PC39$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2741
## 
##        x                  y            
##  Min.   :-2.81152   Min.   :0.0003028  
##  1st Qu.:-1.44347   1st Qu.:0.0267496  
##  Median :-0.07542   Median :0.0841598  
##  Mean   :-0.07542   Mean   :0.1825551  
##  3rd Qu.: 1.29263   3rd Qu.:0.3583722  
##  Max.   : 2.66067   Max.   :0.5790825  
## 
## $tables$PC39$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3475
## 
##        x                 y            
##  Min.   :-3.2569   Min.   :0.0002709  
##  1st Qu.:-1.7623   1st Qu.:0.0244397  
##  Median :-0.2677   Median :0.1051849  
##  Mean   :-0.2677   Mean   :0.1670967  
##  3rd Qu.: 1.2268   3rd Qu.:0.3019446  
##  Max.   : 2.7214   Max.   :0.4689801  
## 
## $tables$PC39$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.3224
## 
##        x                 y            
##  Min.   :-2.3982   Min.   :0.0003817  
##  1st Qu.:-1.1444   1st Qu.:0.0302204  
##  Median : 0.1094   Median :0.1706477  
##  Mean   : 0.1094   Mean   :0.1991784  
##  3rd Qu.: 1.3633   3rd Qu.:0.3682692  
##  Max.   : 2.6171   Max.   :0.4549409  
## 
## $tables$PC39$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.4069
## 
##        x                 y            
##  Min.   :-3.2843   Min.   :0.0003148  
##  1st Qu.:-1.5863   1st Qu.:0.0250752  
##  Median : 0.1118   Median :0.0910534  
##  Mean   : 0.1118   Mean   :0.1470677  
##  3rd Qu.: 1.8098   3rd Qu.:0.2546418  
##  Max.   : 3.5079   Max.   :0.4465223  
## 
## $tables$PC39$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.323
## 
##        x                  y            
##  Min.   :-2.48838   Min.   :0.0003316  
##  1st Qu.:-1.23072   1st Qu.:0.0332861  
##  Median : 0.02694   Median :0.1644678  
##  Mean   : 0.02694   Mean   :0.1985664  
##  3rd Qu.: 1.28460   3rd Qu.:0.3590177  
##  Max.   : 2.54226   Max.   :0.4920690  
## 
## 
## $tables$PC40
## $tables$PC40$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3086
## 
##        x                 y            
##  Min.   :-3.2269   Min.   :0.0002697  
##  1st Qu.:-1.7773   1st Qu.:0.0239660  
##  Median :-0.3276   Median :0.0969505  
##  Mean   :-0.3276   Mean   :0.1722802  
##  3rd Qu.: 1.1220   3rd Qu.:0.3347654  
##  Max.   : 2.5716   Max.   :0.5172826  
## 
## $tables$PC40$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3205
## 
##        x                  y            
##  Min.   :-2.76977   Min.   :0.0002917  
##  1st Qu.:-1.39500   1st Qu.:0.0200457  
##  Median :-0.02023   Median :0.1263219  
##  Mean   :-0.02023   Mean   :0.1816618  
##  3rd Qu.: 1.35453   3rd Qu.:0.3635560  
##  Max.   : 2.72930   Max.   :0.4325731  
## 
## $tables$PC40$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2122
## 
##        x                 y            
##  Min.   :-1.8009   Min.   :0.0005909  
##  1st Qu.:-0.8130   1st Qu.:0.0641775  
##  Median : 0.1748   Median :0.1675898  
##  Mean   : 0.1748   Mean   :0.2528005  
##  3rd Qu.: 1.1627   3rd Qu.:0.3872944  
##  Max.   : 2.1506   Max.   :0.7823085  
## 
## $tables$PC40$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3891
## 
##        x                  y            
##  Min.   :-3.38256   Min.   :0.0003295  
##  1st Qu.:-1.70201   1st Qu.:0.0209824  
##  Median :-0.02147   Median :0.0648317  
##  Mean   :-0.02147   Mean   :0.1486051  
##  3rd Qu.: 1.65907   3rd Qu.:0.3050296  
##  Max.   : 3.33962   Max.   :0.4082948  
## 
## $tables$PC40$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.387
## 
##        x                 y            
##  Min.   :-2.8507   Min.   :0.0002711  
##  1st Qu.:-1.3230   1st Qu.:0.0250947  
##  Median : 0.2047   Median :0.1421614  
##  Mean   : 0.2047   Mean   :0.1634726  
##  3rd Qu.: 1.7324   3rd Qu.:0.2763845  
##  Max.   : 3.2601   Max.   :0.4197634  
## 
## 
## $tables$PC41
## $tables$PC41$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3257
## 
##        x                  y            
##  Min.   :-3.27859   Min.   :0.0002551  
##  1st Qu.:-1.68549   1st Qu.:0.0179537  
##  Median :-0.09239   Median :0.0570961  
##  Mean   :-0.09239   Mean   :0.1567662  
##  3rd Qu.: 1.50071   3rd Qu.:0.3288108  
##  Max.   : 3.09381   Max.   :0.4708008  
## 
## $tables$PC41$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3362
## 
##        x                 y            
##  Min.   :-3.3660   Min.   :0.0002784  
##  1st Qu.:-1.9275   1st Qu.:0.0267205  
##  Median :-0.4890   Median :0.1377192  
##  Mean   :-0.4890   Mean   :0.1736023  
##  3rd Qu.: 0.9495   3rd Qu.:0.3307554  
##  Max.   : 2.3880   Max.   :0.4408093  
## 
## $tables$PC41$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2277
## 
##        x                    y            
##  Min.   :-1.8948627   Min.   :0.0005517  
##  1st Qu.:-0.9477716   1st Qu.:0.0585473  
##  Median :-0.0006805   Median :0.2508864  
##  Mean   :-0.0006805   Mean   :0.2636846  
##  3rd Qu.: 0.9464106   3rd Qu.:0.4070356  
##  Max.   : 1.8935017   Max.   :0.7004761  
## 
## $tables$PC41$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2578
## 
##        x                  y            
##  Min.   :-2.18606   Min.   :0.0005199  
##  1st Qu.:-1.05066   1st Qu.:0.0476983  
##  Median : 0.08474   Median :0.1442313  
##  Mean   : 0.08474   Mean   :0.2199544  
##  3rd Qu.: 1.22014   3rd Qu.:0.3739897  
##  Max.   : 2.35553   Max.   :0.6166495  
## 
## $tables$PC41$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3735
## 
##        x                y            
##  Min.   :-2.633   Min.   :0.0002865  
##  1st Qu.:-1.216   1st Qu.:0.0267866  
##  Median : 0.201   Median :0.1715286  
##  Mean   : 0.201   Mean   :0.1762566  
##  3rd Qu.: 1.618   3rd Qu.:0.3222836  
##  Max.   : 3.035   Max.   :0.3939759  
## 
## 
## $tables$PC42
## $tables$PC42$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3593
## 
##        x                 y            
##  Min.   :-3.2598   Min.   :0.0002796  
##  1st Qu.:-1.7723   1st Qu.:0.0265374  
##  Median :-0.2848   Median :0.1133517  
##  Mean   :-0.2848   Mean   :0.1678898  
##  3rd Qu.: 1.2028   3rd Qu.:0.3109830  
##  Max.   : 2.6903   Max.   :0.4362223  
## 
## $tables$PC42$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3016
## 
##        x                 y            
##  Min.   :-2.6028   Min.   :0.0003259  
##  1st Qu.:-1.3532   1st Qu.:0.0270378  
##  Median :-0.1036   Median :0.1405746  
##  Mean   :-0.1036   Mean   :0.1998563  
##  3rd Qu.: 1.1460   3rd Qu.:0.3831700  
##  Max.   : 2.3956   Max.   :0.5130313  
## 
## $tables$PC42$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2425
## 
##        x                 y            
##  Min.   :-2.2606   Min.   :0.0005519  
##  1st Qu.:-1.2350   1st Qu.:0.0516493  
##  Median :-0.2094   Median :0.1753252  
##  Mean   :-0.2094   Mean   :0.2434975  
##  3rd Qu.: 0.8162   3rd Qu.:0.4440968  
##  Max.   : 1.8418   Max.   :0.5949649  
## 
## $tables$PC42$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2575
## 
##        x                 y            
##  Min.   :-2.5843   Min.   :0.0005052  
##  1st Qu.:-1.4000   1st Qu.:0.0404141  
##  Median :-0.2156   Median :0.1188080  
##  Mean   :-0.2156   Mean   :0.2108566  
##  3rd Qu.: 0.9687   3rd Qu.:0.3648901  
##  Max.   : 2.1531   Max.   :0.6328193  
## 
## $tables$PC42$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.297
## 
##        x                 y            
##  Min.   :-2.4009   Min.   :0.0003519  
##  1st Qu.:-0.8809   1st Qu.:0.0151537  
##  Median : 0.6390   Median :0.0886691  
##  Mean   : 0.6390   Mean   :0.1643077  
##  3rd Qu.: 2.1590   3rd Qu.:0.2977821  
##  Max.   : 3.6790   Max.   :0.5029885  
## 
## 
## $tables$PC43
## $tables$PC43$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.3161
## 
##        x                  y            
##  Min.   :-2.75795   Min.   :0.0002637  
##  1st Qu.:-1.41268   1st Qu.:0.0248769  
##  Median :-0.06741   Median :0.1429528  
##  Mean   :-0.06741   Mean   :0.1856444  
##  3rd Qu.: 1.27786   3rd Qu.:0.3244688  
##  Max.   : 2.62313   Max.   :0.4952364  
## 
## $tables$PC43$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.2486
## 
##        x                 y            
##  Min.   :-2.5894   Min.   :0.0003759  
##  1st Qu.:-1.3755   1st Qu.:0.0414655  
##  Median :-0.1616   Median :0.1733906  
##  Mean   :-0.1616   Mean   :0.2057335  
##  3rd Qu.: 1.0523   3rd Qu.:0.3108065  
##  Max.   : 2.2662   Max.   :0.6341915  
## 
## $tables$PC43$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2377
## 
##        x                 y            
##  Min.   :-2.2821   Min.   :0.0005207  
##  1st Qu.:-1.3088   1st Qu.:0.0380480  
##  Median :-0.3355   Median :0.1315048  
##  Mean   :-0.3355   Mean   :0.2565821  
##  3rd Qu.: 0.6379   3rd Qu.:0.4886823  
##  Max.   : 1.6112   Max.   :0.7333261  
## 
## $tables$PC43$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2229
## 
##        x                  y            
##  Min.   :-1.91088   Min.   :0.0005897  
##  1st Qu.:-0.91076   1st Qu.:0.0568818  
##  Median : 0.08936   Median :0.1430394  
##  Mean   : 0.08936   Mean   :0.2496982  
##  3rd Qu.: 1.08948   3rd Qu.:0.4615757  
##  Max.   : 2.08960   Max.   :0.6945699  
## 
## $tables$PC43$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.4009
## 
##        x                 y            
##  Min.   :-3.2158   Min.   :0.0002605  
##  1st Qu.:-1.4477   1st Qu.:0.0191976  
##  Median : 0.3203   Median :0.0738674  
##  Mean   : 0.3203   Mean   :0.1412510  
##  3rd Qu.: 2.0884   3rd Qu.:0.2991451  
##  Max.   : 3.8565   Max.   :0.3806270  
## 
## 
## $tables$PC44
## $tables$PC44$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2599
## 
##        x                 y           
##  Min.   :-3.6460   Min.   :0.000321  
##  1st Qu.:-1.9677   1st Qu.:0.021069  
##  Median :-0.2894   Median :0.070138  
##  Mean   :-0.2894   Mean   :0.148810  
##  3rd Qu.: 1.3889   3rd Qu.:0.257914  
##  Max.   : 3.0672   Max.   :0.588458  
## 
## $tables$PC44$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.2963
## 
##        x                 y           
##  Min.   :-2.8140   Min.   :0.000322  
##  1st Qu.:-1.5466   1st Qu.:0.025548  
##  Median :-0.2791   Median :0.134527  
##  Mean   :-0.2791   Mean   :0.197048  
##  3rd Qu.: 0.9883   3rd Qu.:0.384337  
##  Max.   : 2.2557   Max.   :0.508945  
## 
## $tables$PC44$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2985
## 
##        x                 y            
##  Min.   :-1.9441   Min.   :0.0004861  
##  1st Qu.:-0.8323   1st Qu.:0.0393434  
##  Median : 0.2796   Median :0.1868963  
##  Mean   : 0.2796   Mean   :0.2246070  
##  3rd Qu.: 1.3915   3rd Qu.:0.4086367  
##  Max.   : 2.5033   Max.   :0.5188311  
## 
## $tables$PC44$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.3044
## 
##        x                 y            
##  Min.   :-2.6045   Min.   :0.0004229  
##  1st Qu.:-1.4778   1st Qu.:0.0358840  
##  Median :-0.3512   Median :0.2015695  
##  Mean   :-0.3512   Mean   :0.2216507  
##  3rd Qu.: 0.7755   3rd Qu.:0.3769172  
##  Max.   : 1.9021   Max.   :0.5180775  
## 
## $tables$PC44$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.2351
## 
##        x                  y            
##  Min.   :-1.74092   Min.   :0.0004811  
##  1st Qu.:-0.84436   1st Qu.:0.0449439  
##  Median : 0.05221   Median :0.2498392  
##  Mean   : 0.05221   Mean   :0.2785443  
##  3rd Qu.: 0.94877   3rd Qu.:0.5368318  
##  Max.   : 1.84533   Max.   :0.6034330  
## 
## 
## $tables$PC45
## $tables$PC45$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2642
## 
##        x                  y            
##  Min.   :-2.69267   Min.   :0.0003148  
##  1st Qu.:-1.32959   1st Qu.:0.0284936  
##  Median : 0.03349   Median :0.1072742  
##  Mean   : 0.03349   Mean   :0.1832205  
##  3rd Qu.: 1.39657   3rd Qu.:0.2971160  
##  Max.   : 2.75965   Max.   :0.6046782  
## 
## $tables$PC45$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3238
## 
##        x                 y           
##  Min.   :-2.3197   Min.   :0.000293  
##  1st Qu.:-0.9980   1st Qu.:0.028355  
##  Median : 0.3236   Median :0.145639  
##  Mean   : 0.3236   Mean   :0.188962  
##  3rd Qu.: 1.6452   3rd Qu.:0.341292  
##  Max.   : 2.9669   Max.   :0.500676  
## 
## $tables$PC45$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.1738
## 
##        x                 y            
##  Min.   :-1.9949   Min.   :0.0007169  
##  1st Qu.:-1.0853   1st Qu.:0.0970947  
##  Median :-0.1757   Median :0.1957790  
##  Mean   :-0.1757   Mean   :0.2745540  
##  3rd Qu.: 0.7339   3rd Qu.:0.3667848  
##  Max.   : 1.6436   Max.   :1.0025417  
## 
## $tables$PC45$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2995
## 
##        x                 y            
##  Min.   :-2.6751   Min.   :0.0004315  
##  1st Qu.:-1.3877   1st Qu.:0.0337108  
##  Median :-0.1003   Median :0.0925327  
##  Mean   :-0.1003   Mean   :0.1939860  
##  3rd Qu.: 1.1871   3rd Qu.:0.3879378  
##  Max.   : 2.4745   Max.   :0.5375880  
## 
## $tables$PC45$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3013
## 
##        x                  y            
##  Min.   :-2.29516   Min.   :0.0003615  
##  1st Qu.:-1.10267   1st Qu.:0.0280889  
##  Median : 0.08981   Median :0.1738358  
##  Mean   : 0.08981   Mean   :0.2094286  
##  3rd Qu.: 1.28229   3rd Qu.:0.4009811  
##  Max.   : 2.47478   Max.   :0.5057383  
## 
## 
## $tables$PC46
## $tables$PC46$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.274
## 
##        x                 y            
##  Min.   :-2.1912   Min.   :0.0003082  
##  1st Qu.:-0.9043   1st Qu.:0.0264117  
##  Median : 0.3825   Median :0.0758337  
##  Mean   : 0.3825   Mean   :0.1940777  
##  3rd Qu.: 1.6693   3rd Qu.:0.3883485  
##  Max.   : 2.9561   Max.   :0.5661528  
## 
## $tables$PC46$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.2631
## 
##        x                 y            
##  Min.   :-2.2548   Min.   :0.0003565  
##  1st Qu.:-1.0633   1st Qu.:0.0289951  
##  Median : 0.1283   Median :0.1250798  
##  Mean   : 0.1283   Mean   :0.2095889  
##  3rd Qu.: 1.3199   3rd Qu.:0.3696893  
##  Max.   : 2.5115   Max.   :0.6460218  
## 
## $tables$PC46$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.241
## 
##        x                 y            
##  Min.   :-2.2806   Min.   :0.0005028  
##  1st Qu.:-1.0654   1st Qu.:0.0356274  
##  Median : 0.1497   Median :0.0899683  
##  Mean   : 0.1497   Mean   :0.2055180  
##  3rd Qu.: 1.3649   3rd Qu.:0.3747441  
##  Max.   : 2.5801   Max.   :0.6566934  
## 
## $tables$PC46$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.1575
## 
##        x                 y            
##  Min.   :-1.6830   Min.   :0.0008116  
##  1st Qu.:-0.5367   1st Qu.:0.0202895  
##  Median : 0.6095   Median :0.0692078  
##  Mean   : 0.6095   Mean   :0.2178745  
##  3rd Qu.: 1.7558   3rd Qu.:0.2364947  
##  Max.   : 2.9020   Max.   :1.0258363  
## 
## $tables$PC46$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3172
## 
##        x                  y            
##  Min.   :-2.44370   Min.   :0.0003646  
##  1st Qu.:-1.24225   1st Qu.:0.0289369  
##  Median :-0.04081   Median :0.1989848  
##  Mean   :-0.04081   Mean   :0.2078652  
##  3rd Qu.: 1.16063   3rd Qu.:0.3662283  
##  Max.   : 2.36208   Max.   :0.4745637  
## 
## 
## $tables$PC47
## $tables$PC47$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2835
## 
##        x                  y            
##  Min.   :-2.75398   Min.   :0.0002945  
##  1st Qu.:-1.40343   1st Qu.:0.0280722  
##  Median :-0.05288   Median :0.0868758  
##  Mean   :-0.05288   Mean   :0.1849191  
##  3rd Qu.: 1.29768   3rd Qu.:0.3636554  
##  Max.   : 2.64823   Max.   :0.5081487  
## 
## $tables$PC47$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.3431
## 
##        x                 y           
##  Min.   :-3.2410   Min.   :0.000286  
##  1st Qu.:-1.7986   1st Qu.:0.024099  
##  Median :-0.3561   Median :0.108260  
##  Mean   :-0.3561   Mean   :0.173137  
##  3rd Qu.: 1.0863   3rd Qu.:0.351547  
##  Max.   : 2.5288   Max.   :0.457268  
## 
## $tables$PC47$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.1899
## 
##        x                 y            
##  Min.   :-1.5263   Min.   :0.0006389  
##  1st Qu.:-0.7047   1st Qu.:0.0410689  
##  Median : 0.1168   Median :0.1373380  
##  Mean   : 0.1168   Mean   :0.3039885  
##  3rd Qu.: 0.9384   3rd Qu.:0.5926276  
##  Max.   : 1.7599   Max.   :0.8862401  
## 
## $tables$PC47$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2717
## 
##        x                 y            
##  Min.   :-2.1043   Min.   :0.0004799  
##  1st Qu.:-1.0756   1st Qu.:0.0335850  
##  Median :-0.0469   Median :0.1955185  
##  Mean   :-0.0469   Mean   :0.2427706  
##  3rd Qu.: 0.9818   3rd Qu.:0.4348258  
##  Max.   : 2.0105   Max.   :0.6227226  
## 
## $tables$PC47$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.2615
## 
##        x                 y            
##  Min.   :-1.6039   Min.   :0.0006068  
##  1st Qu.:-0.6033   1st Qu.:0.0592235  
##  Median : 0.3973   Median :0.2024413  
##  Mean   : 0.3973   Mean   :0.2495607  
##  3rd Qu.: 1.3979   3rd Qu.:0.4765764  
##  Max.   : 2.3985   Max.   :0.5958888  
## 
## 
## $tables$PC48
## $tables$PC48$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.2604
## 
##        x                  y            
##  Min.   :-2.19308   Min.   :0.0003232  
##  1st Qu.:-1.10493   1st Qu.:0.0353912  
##  Median :-0.01678   Median :0.2039472  
##  Mean   :-0.01678   Mean   :0.2295088  
##  3rd Qu.: 1.07137   3rd Qu.:0.3909215  
##  Max.   : 2.15951   Max.   :0.5686532  
## 
## $tables$PC48$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.2099
## 
##        x                   y            
##  Min.   :-1.759743   Min.   :0.0004467  
##  1st Qu.:-0.882109   1st Qu.:0.0400130  
##  Median :-0.004475   Median :0.2252785  
##  Mean   :-0.004475   Mean   :0.2845568  
##  3rd Qu.: 0.873159   3rd Qu.:0.4875894  
##  Max.   : 1.750793   Max.   :0.7706856  
## 
## $tables$PC48$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2464
## 
##        x                 y            
##  Min.   :-1.6765   Min.   :0.0004923  
##  1st Qu.:-0.6466   1st Qu.:0.0368531  
##  Median : 0.3833   Median :0.1701567  
##  Mean   : 0.3833   Mean   :0.2424854  
##  3rd Qu.: 1.4132   3rd Qu.:0.4715644  
##  Max.   : 2.4431   Max.   :0.6372767  
## 
## $tables$PC48$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2299
## 
##        x                 y            
##  Min.   :-1.6393   Min.   :0.0005571  
##  1st Qu.:-0.5510   1st Qu.:0.0174225  
##  Median : 0.5374   Median :0.0717317  
##  Mean   : 0.5374   Mean   :0.2294626  
##  3rd Qu.: 1.6257   3rd Qu.:0.4561151  
##  Max.   : 2.7141   Max.   :0.7501261  
## 
## $tables$PC48$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.3682
## 
##        x                 y           
##  Min.   :-3.2603   Min.   :0.000284  
##  1st Qu.:-1.7454   1st Qu.:0.022363  
##  Median :-0.2306   Median :0.112146  
##  Mean   :-0.2306   Mean   :0.164861  
##  3rd Qu.: 1.2843   3rd Qu.:0.295443  
##  Max.   : 2.7991   Max.   :0.451926  
## 
## 
## $tables$PC49
## $tables$PC49$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.21
## 
##        x                 y            
##  Min.   :-1.9304   Min.   :0.0005172  
##  1st Qu.:-1.0708   1st Qu.:0.0545130  
##  Median :-0.2112   Median :0.2387843  
##  Mean   :-0.2112   Mean   :0.2905207  
##  3rd Qu.: 0.6484   3rd Qu.:0.4956137  
##  Max.   : 1.5080   Max.   :0.7592140  
## 
## $tables$PC49$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.2687
## 
##        x                 y            
##  Min.   :-2.8028   Min.   :0.0003479  
##  1st Qu.:-1.4826   1st Qu.:0.0308377  
##  Median :-0.1624   Median :0.0926299  
##  Mean   :-0.1624   Mean   :0.1891694  
##  3rd Qu.: 1.1578   3rd Qu.:0.3818065  
##  Max.   : 2.4780   Max.   :0.5697780  
## 
## $tables$PC49$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2182
## 
##        x                 y            
##  Min.   :-1.9188   Min.   :0.0005572  
##  1st Qu.:-0.8374   1st Qu.:0.0473424  
##  Median : 0.2439   Median :0.1470826  
##  Mean   : 0.2439   Mean   :0.2309577  
##  3rd Qu.: 1.3252   3rd Qu.:0.3730039  
##  Max.   : 2.4065   Max.   :0.7516285  
## 
## $tables$PC49$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2939
## 
##        x                 y            
##  Min.   :-2.0289   Min.   :0.0004599  
##  1st Qu.:-0.9654   1st Qu.:0.0347967  
##  Median : 0.0981   Median :0.2288500  
##  Mean   : 0.0981   Mean   :0.2348229  
##  3rd Qu.: 1.1616   3rd Qu.:0.4029916  
##  Max.   : 2.2251   Max.   :0.5393334  
## 
## $tables$PC49$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.2783
## 
##        x                 y            
##  Min.   :-2.2958   Min.   :0.0003924  
##  1st Qu.:-1.2061   1st Qu.:0.0345819  
##  Median :-0.1163   Median :0.1979231  
##  Mean   :-0.1163   Mean   :0.2291660  
##  3rd Qu.: 0.9735   3rd Qu.:0.3883615  
##  Max.   : 2.0632   Max.   :0.5769287  
## 
## 
## $tables$user_namecarlitos
## $tables$user_namecarlitos$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.1374
## 
##        x                  y           
##  Min.   :-0.41224   Min.   :0.002545  
##  1st Qu.: 0.04388   1st Qu.:0.041013  
##  Median : 0.50000   Median :0.207737  
##  Mean   : 0.50000   Mean   :0.546850  
##  3rd Qu.: 0.95612   3rd Qu.:0.629738  
##  Max.   : 1.41224   Max.   :2.526183  
## 
## $tables$user_namecarlitos$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.1906
## 
##        x                 y           
##  Min.   :-0.5718   Min.   :0.006853  
##  1st Qu.:-0.0359   1st Qu.:0.104760  
##  Median : 0.5000   Median :0.336593  
##  Mean   : 0.5000   Mean   :0.465433  
##  3rd Qu.: 1.0359   3rd Qu.:0.608839  
##  Max.   : 1.5718   Max.   :1.482187  
## 
## $tables$user_namecarlitos$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.1736
## 
##        x                  y           
##  Min.   :-0.52069   Min.   :0.004881  
##  1st Qu.:-0.01035   1st Qu.:0.075664  
##  Median : 0.50000   Median :0.269761  
##  Mean   : 0.50000   Mean   :0.488742  
##  3rd Qu.: 1.01035   3rd Qu.:0.627600  
##  Max.   : 1.52069   Max.   :1.863388  
## 
## $tables$user_namecarlitos$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.1569
## 
##        x                  y           
##  Min.   :-0.47078   Min.   :0.004082  
##  1st Qu.: 0.01461   1st Qu.:0.055957  
##  Median : 0.50000   Median :0.227215  
##  Mean   : 0.50000   Mean   :0.513868  
##  3rd Qu.: 0.98539   3rd Qu.:0.656655  
##  Max.   : 1.47078   Max.   :2.178200  
## 
## $tables$user_namecarlitos$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.1925
## 
##        x                  y           
##  Min.   :-0.57754   Min.   :0.006499  
##  1st Qu.:-0.03877   1st Qu.:0.104136  
##  Median : 0.50000   Median :0.329395  
##  Mean   : 0.50000   Mean   :0.462955  
##  3rd Qu.: 1.03877   3rd Qu.:0.578038  
##  Max.   : 1.57754   Max.   :1.493479  
## 
## 
## $tables$user_namecharles
## $tables$user_namecharles$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.1928
## 
##        x                  y           
##  Min.   :-0.57855   Min.   :0.007749  
##  1st Qu.:-0.03927   1st Qu.:0.111098  
##  Median : 0.50000   Median :0.356031  
##  Mean   : 0.50000   Mean   :0.462524  
##  3rd Qu.: 1.03927   3rd Qu.:0.673940  
##  Max.   : 1.57855   Max.   :1.378647  
## 
## $tables$user_namecharles$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.1977
## 
##        x                  y           
##  Min.   :-0.59304   Min.   :0.007556  
##  1st Qu.:-0.04652   1st Qu.:0.114874  
##  Median : 0.50000   Median :0.354296  
##  Mean   : 0.50000   Mean   :0.456392  
##  3rd Qu.: 1.04652   3rd Qu.:0.659407  
##  Max.   : 1.59304   Max.   :1.345096  
## 
## $tables$user_namecharles$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.2149
## 
##        x                  y           
##  Min.   :-0.64476   Min.   :0.007889  
##  1st Qu.:-0.07238   1st Qu.:0.137971  
##  Median : 0.50000   Median :0.365742  
##  Mean   : 0.50000   Mean   :0.435770  
##  3rd Qu.: 1.07238   3rd Qu.:0.672430  
##  Max.   : 1.64476   Max.   :1.153636  
## 
## $tables$user_namecharles$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2026
## 
##        x                  y          
##  Min.   :-0.60778   Min.   :0.00631  
##  1st Qu.:-0.05389   1st Qu.:0.11294  
##  Median : 0.50000   Median :0.33101  
##  Mean   : 0.50000   Mean   :0.45032  
##  3rd Qu.: 1.05389   3rd Qu.:0.56245  
##  Max.   : 1.60778   Max.   :1.40630  
## 
## $tables$user_namecharles$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.1971
## 
##        x                  y           
##  Min.   :-0.59135   Min.   :0.006872  
##  1st Qu.:-0.04567   1st Qu.:0.110783  
##  Median : 0.50000   Median :0.340865  
##  Mean   : 0.50000   Mean   :0.457098  
##  3rd Qu.: 1.04567   3rd Qu.:0.609166  
##  Max.   : 1.59135   Max.   :1.411560  
## 
## 
## $tables$user_nameeurico
## $tables$user_nameeurico$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.1832
## 
##        x                  y           
##  Min.   :-0.54970   Min.   :0.006785  
##  1st Qu.:-0.02485   1st Qu.:0.096638  
##  Median : 0.50000   Median :0.330499  
##  Mean   : 0.50000   Mean   :0.475234  
##  3rd Qu.: 1.02485   3rd Qu.:0.603589  
##  Max.   : 1.54970   Max.   :1.572220  
## 
## $tables$user_nameeurico$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.1637
## 
##        x                  y           
##  Min.   :-0.49102   Min.   :0.005133  
##  1st Qu.: 0.00449   1st Qu.:0.068524  
##  Median : 0.50000   Median :0.268763  
##  Mean   : 0.50000   Mean   :0.503376  
##  3rd Qu.: 0.99551   3rd Qu.:0.624052  
##  Max.   : 1.49102   Max.   :1.979989  
## 
## $tables$user_nameeurico$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.1736
## 
##        x                  y           
##  Min.   :-0.52069   Min.   :0.004881  
##  1st Qu.:-0.01035   1st Qu.:0.075664  
##  Median : 0.50000   Median :0.269761  
##  Mean   : 0.50000   Mean   :0.488742  
##  3rd Qu.: 1.01035   3rd Qu.:0.627600  
##  Max.   : 1.52069   Max.   :1.863388  
## 
## $tables$user_nameeurico$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.1649
## 
##        x                   y           
##  Min.   :-0.494784   Min.   :0.006984  
##  1st Qu.: 0.002608   1st Qu.:0.080137  
##  Median : 0.500000   Median :0.318280  
##  Mean   : 0.500000   Mean   :0.501471  
##  3rd Qu.: 0.997392   3rd Qu.:0.620593  
##  Max.   : 1.494784   Max.   :1.796353  
## 
## $tables$user_nameeurico$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.1584
## 
##        x                  y           
##  Min.   :-0.47535   Min.   :0.004603  
##  1st Qu.: 0.01233   1st Qu.:0.061034  
##  Median : 0.50000   Median :0.245896  
##  Mean   : 0.50000   Mean   :0.511465  
##  3rd Qu.: 0.98767   3rd Qu.:0.638116  
##  Max.   : 1.47535   Max.   :2.107319  
## 
## 
## $tables$user_namejeremy
## $tables$user_namejeremy$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.4053
## 
##        x                 y          
##  Min.   :-1.2159   Min.   :0.01104  
##  1st Qu.:-0.6079   1st Qu.:0.07850  
##  Median : 0.0000   Median :0.31894  
##  Mean   : 0.0000   Mean   :0.40973  
##  3rd Qu.: 0.6079   3rd Qu.:0.74092  
##  Max.   : 1.2159   Max.   :0.98416  
## 
## $tables$user_namejeremy$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.4149
## 
##        x                 y          
##  Min.   :-1.2448   Min.   :0.01078  
##  1st Qu.:-0.6224   1st Qu.:0.07667  
##  Median : 0.0000   Median :0.31151  
##  Mean   : 0.0000   Mean   :0.40019  
##  3rd Qu.: 0.6224   3rd Qu.:0.72367  
##  Max.   : 1.2448   Max.   :0.96125  
## 
## $tables$user_namejeremy$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.4371
## 
##        x                 y          
##  Min.   :-1.3114   Min.   :0.01024  
##  1st Qu.:-0.6557   1st Qu.:0.07278  
##  Median : 0.0000   Median :0.29571  
##  Mean   : 0.0000   Mean   :0.37989  
##  3rd Qu.: 0.6557   3rd Qu.:0.68697  
##  Max.   : 1.3114   Max.   :0.91249  
## 
## $tables$user_namejeremy$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.442
## 
##        x                y          
##  Min.   :-1.326   Min.   :0.01012  
##  1st Qu.:-0.663   1st Qu.:0.07198  
##  Median : 0.000   Median :0.29244  
##  Mean   : 0.000   Mean   :0.37569  
##  3rd Qu.: 0.663   3rd Qu.:0.67937  
##  Max.   : 1.326   Max.   :0.90240  
## 
## $tables$user_namejeremy$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.4242
## 
##        x                 y          
##  Min.   :-1.2725   Min.   :0.01055  
##  1st Qu.:-0.6363   1st Qu.:0.07500  
##  Median : 0.0000   Median :0.30473  
##  Mean   : 0.0000   Mean   :0.39148  
##  3rd Qu.: 0.6363   3rd Qu.:0.70793  
##  Max.   : 1.2725   Max.   :0.94033  
## 
## 
## $tables$user_namepedro
## $tables$user_namepedro$A
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (54 obs.);	Bandwidth 'bw' = 0.1793
## 
##        x                  y           
##  Min.   :-0.53783   Min.   :0.006483  
##  1st Qu.:-0.01891   1st Qu.:0.090733  
##  Median : 0.50000   Median :0.319779  
##  Mean   : 0.50000   Mean   :0.480671  
##  3rd Qu.: 1.01891   3rd Qu.:0.578887  
##  Max.   : 1.53783   Max.   :1.647901  
## 
## $tables$user_namepedro$B
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (48 obs.);	Bandwidth 'bw' = 0.1637
## 
##        x                  y           
##  Min.   :-0.49102   Min.   :0.005133  
##  1st Qu.: 0.00449   1st Qu.:0.068524  
##  Median : 0.50000   Median :0.268763  
##  Mean   : 0.50000   Mean   :0.503376  
##  3rd Qu.: 0.99551   3rd Qu.:0.624052  
##  Max.   : 1.49102   Max.   :1.979989  
## 
## $tables$user_namepedro$C
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (37 obs.);	Bandwidth 'bw' = 0.1901
## 
##        x                  y          
##  Min.   :-0.57039   Min.   :0.00573  
##  1st Qu.:-0.03519   1st Qu.:0.09723  
##  Median : 0.50000   Median :0.30831  
##  Mean   : 0.50000   Mean   :0.46605  
##  3rd Qu.: 1.03519   3rd Qu.:0.58745  
##  Max.   : 1.57039   Max.   :1.58738  
## 
## $tables$user_namepedro$D
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (35 obs.);	Bandwidth 'bw' = 0.2082
## 
##        x                  y           
##  Min.   :-0.62457   Min.   :0.006763  
##  1st Qu.:-0.06228   1st Qu.:0.123599  
##  Median : 0.50000   Median :0.343358  
##  Mean   : 0.50000   Mean   :0.443596  
##  3rd Qu.: 1.06228   3rd Qu.:0.599117  
##  Max.   : 1.62457   Max.   :1.313664  
## 
## $tables$user_namepedro$E
## 
## Call:
## 	density.default(x = xx)
## 
## Data: xx (43 obs.);	Bandwidth 'bw' = 0.1583
## 
##        x                  y           
##  Min.   :-0.47483   Min.   :0.007239  
##  1st Qu.: 0.01259   1st Qu.:0.074461  
##  Median : 0.50000   Median :0.315725  
##  Mean   : 0.50000   Mean   :0.511738  
##  3rd Qu.: 0.98741   3rd Qu.:0.642753  
##  Max.   : 1.47483   Max.   :1.875324  
## 
## 
## 
## $levels
## [1] "A" "B" "C" "D" "E"
## 
## $call
## NaiveBayes.default(x = x, grouping = y, usekernel = param$usekernel, 
##     fL = param$fL)
## 
## $x
##            PC1         PC2           PC3           PC4          PC5
## 1   -4.5861008 -6.57842740  -6.983832579  4.567782e+00   0.58118790
## 2    4.9679510  3.10281961  -6.161183799  6.351767e-01   4.19489504
## 3    5.4508083  2.63975730  -4.350489469  1.184100e+00   3.09620383
## 4    4.9102604  2.72818626  -1.301128181  8.252339e-01   3.49574651
## 5    5.1426308  4.19013322   0.869165118  1.464258e+00   0.49417239
## 6    4.5424628  4.66653242   3.727657559  1.522143e-01  -0.91255866
## 7    4.6371649  5.90148530   0.050036386 -3.098484e-01   0.34608866
## 8    5.0158104  5.67800661  -1.595479394 -6.747494e-03  -1.09058885
## 9    4.1987360  5.47538470  -2.126211481  3.707171e-01   2.11746332
## 10   5.1660914  4.36033839   1.446738771  8.965251e-01   0.09441246
## 11   5.2390363  7.15275311   1.417070381 -3.012502e-01  -1.61792077
## 12   4.8718631  5.50437637   2.807152190 -2.588001e-01  -1.88459787
## 13   5.9950547  3.37272911   0.278574351  1.434618e+00  -0.88501434
## 14   2.5381184 -5.19976772  -0.166184628 -7.707021e-04  -1.52907839
## 15  -4.6186018  3.06955180  -8.940558942  3.803073e-01   4.45915303
## 16  -5.9993975  0.70519505  -4.522442133  5.493591e-01   0.88688971
## 17  -6.5188079 -0.58666873   4.277459759  7.107717e-01  -2.91691995
## 18  -6.2453665 -1.16328938   1.035053425  1.259741e+00   1.06609820
## 19  -6.4743297 -0.08962471   1.907241925  1.201559e+00  -1.02254359
## 20  -6.1072712  1.00635267   0.001954487 -3.654398e-01   0.56521396
## 21  -5.8210243  0.91077050  -1.810765482  2.367947e+00   1.30015641
## 22  -5.4336960  0.89269047  -0.459413678  1.081942e+00   2.81655652
## 23  -7.1520008 -0.01391586   3.088704570  5.645512e-01   0.91221386
## 24  -5.1557623  1.32360829  -2.236139848  1.835677e+00   1.55195314
## 25  -6.8331609 -0.31731866   2.475624051  1.456911e-01   0.62679967
## 26  -7.1350790 -1.06732334   1.240846116 -4.732573e-01   0.29191864
## 27  -2.4699122 -2.38547120  -1.023117317  2.847665e+00  -2.93582393
## 28  -3.5763584  1.22797831   1.001018555  1.112540e+00  -5.54133658
## 29  -5.3159328  2.42266625  -0.487152371  4.012635e-01  -4.67817494
## 30  -3.7759538  2.61976793   5.449280090  7.642279e-01  -8.04823957
## 31  -3.0174980  0.02356260   1.411914922  1.865832e+00  -0.96534250
## 32  -5.5943287  1.43860557  -1.223076304  7.033826e-01  -2.16352650
## 33   3.1487425 -1.20791405  -5.001238008  8.689409e-01   1.68251141
## 34   2.1640510 -5.70441438  -1.824519990  1.922471e+00   0.01324338
## 35   3.6407681 -4.28085103  -4.413530107  2.334145e+00   1.52612571
## 36   2.0110929 -7.55191373  -3.551315547  3.256983e+00   1.20629279
## 37   3.2996664 -4.53391896  -5.301103287  1.003039e+00   0.25484798
## 38   2.2582426 -5.43642523  -0.210426666  8.764651e-01  -2.62468337
## 39   2.5886205 -4.80283402  -1.484625082  7.644963e-01  -1.99070045
## 40   4.0337007 -3.86205885  -2.183984536  1.725078e+00  -0.15781573
## 41   4.0345010 -2.55958995  -1.540605128  1.509914e+00  -1.86764562
## 42   3.4145956 -2.32249426  -0.522857193  1.436271e+00  -0.72405717
## 43   2.4585506 -4.49855223   0.268694190  9.919242e-01  -0.77651208
## 44   3.8475652 -3.82207963  -1.974305439  2.370849e+00  -1.18990437
## 45   3.7941066 -1.85009342  -0.118796235  1.457162e+00  -0.78376637
## 46   3.1835972 -4.70915500  -0.708867901  1.446398e+00  -1.86986977
## 47   3.5938385 -3.63628598  -3.482860634  7.790781e-01   1.81893135
## 48   3.0112816 -3.95489305   0.068495493  1.635238e+00  -0.81047380
## 49   3.1362361 -4.67787517  -3.152551894  1.677821e+00   0.86604891
## 50  -7.3279333 -1.44582735   3.219647135  3.973448e-02  -0.61387865
## 51  -7.1628344  4.25859613 -28.300411143 -2.181614e+01 -13.72046426
## 52  -4.2365533  1.27167998  -8.972570487  6.259891e-01   3.80909708
## 53   5.9382158  3.90134183   0.027662986  8.427221e-01  -0.51249837
## 54   4.5612231  4.88558935  -0.622284950  5.330193e-01   0.57909203
## 55   5.1208668  3.80450977   2.076105499 -7.800772e-01   0.71378462
## 56   4.4656057  5.81091723   4.465280167 -1.578847e+00  -0.15138000
## 57   4.0889006  6.02500634   4.414534380 -1.879379e+00  -1.18593262
## 58   3.5522104  3.59971060   4.470587427 -2.514687e+00  -2.16597444
## 59   3.3894341  3.48914259  -1.676892654 -2.797929e+00   1.60660562
## 60   3.7287335  3.82372479   1.404497528 -4.412027e+00   1.19877452
## 61   3.5655509  3.58174653   3.137406744 -1.228536e+00   0.08127464
## 62   3.2035400  2.41637667   2.239075935 -7.691956e-01  -1.11254166
## 63  -5.2239178  2.88714759  -1.762288475  1.240039e+00   3.06591845
## 64  -6.0789495  3.32070644  -2.375283567  2.522341e+00   3.11345813
## 65  -7.0474161 -0.06486067  -2.884059496  2.050195e+00   1.72918718
## 66  -4.6838910  3.08718519  -2.851703237  3.368385e+00   3.66570236
## 67  -7.5097200  1.94930431   0.566708916  1.297509e+00   1.87760232
## 68  -3.6897537  2.81124980   2.476444660  2.658128e+00  -2.96482136
## 69  -5.1133813  1.63415218   2.870336801 -5.469007e-01  -5.30528020
## 70  -4.4252118 -0.42102899  -0.458417761  1.920864e+00   1.71891814
## 71  -6.6488478  0.69902863  -3.238238946 -1.907520e+00  -3.61571793
## 72  -4.9881085  1.79765826   2.896927820 -1.185082e-01  -5.93235303
## 73  -4.7958777  0.84591149   1.704848209  5.338803e-01  -3.93739099
## 74  -4.0913914  3.08370734   0.640417365  2.582458e+00  -1.66279362
## 75  -3.1419075  4.10334129   1.108834128  2.037045e+00  -1.94747306
## 76  -2.5817244  1.12346478   0.676069371  3.320086e+00  -0.36999255
## 77  -2.7052254  4.14731183   0.878973438 -1.327348e-01  -1.58388461
## 78   3.4037235 -3.17411041  -1.393410538  8.062585e-01   1.05536320
## 79   3.5792451 -1.75560622   1.011341319  1.637153e+00  -2.42125195
## 80   3.0129148 -1.45717852  -0.434131587  3.221104e+00  -0.99393524
## 81   3.1524450 -3.37395482   0.119425790  2.303721e+00   0.17271137
## 82   3.1641758 -1.89460156  -2.193940468  2.097097e+00  -0.26086331
## 83   3.0236433 -1.75155416   0.412013937  1.159526e+00  -0.55578672
## 84   2.4548700 -1.86597166  -4.859007411  3.033642e+00   0.99881332
## 85   3.3282142 -1.86534506  -0.402468443  1.606374e+00   0.29692261
## 86   2.4793428 -0.99422902  -0.987830484  2.183720e+00  -1.63302179
## 87   2.5063641 -2.95349154   0.368930216 -1.547713e+00  -3.90189158
## 88   0.3008661 -6.53405081   1.337424629  6.981891e-01  -0.79146961
## 89   0.5467206 -4.54158737   0.455078778 -5.902853e-01  -2.58327649
## 90   2.3744953 -4.07341937   0.446947421 -1.209830e+00  -1.38468600
## 91   1.1251452 -3.63716316   2.510078283  4.751275e-01  -3.28764765
## 92   1.9286010 -4.38658880   0.426014828  1.678712e-01  -4.05486129
## 93   3.5556662 -2.79282651  -0.830202938  2.450314e+00  -0.64164762
## 94  -5.0029737  3.78801080   0.746202441  2.267102e+00   3.30482241
## 95  -5.7939482 -1.51896533   3.541204977 -7.270847e-01   0.10523294
## 96  -5.2806763 -1.13666368  -1.073119921  6.830486e-01   2.34422946
## 97  -6.2412870  1.98042454  -3.421317242  1.951447e+00   3.15043002
## 98   3.8676399  4.19601278   0.565607858 -3.449621e+00   0.48540661
## 99  -3.4806710  0.88855241  -0.411807049  3.586225e+00  -0.61804575
## 100 -2.7433730  2.09053865   3.790472574  1.546342e+00  -0.84941123
## 101 -4.1083514  3.25650192   0.245553466  1.279273e+00  -1.99651340
## 102 -4.3564331  2.40141742   4.761973843  1.801618e+00  -3.00241918
## 103  4.0254209  4.41957070   0.078323743 -2.625812e+00   0.07289094
## 104  4.0959976  3.84856260   0.390806960 -1.099920e+00  -1.47196979
## 105  5.1874498  4.22068288  -2.143460423  1.799616e+00   0.75252094
## 106  5.7924266  2.98536705  -0.872474638  1.722863e+00   0.67523626
## 107  4.3504304  5.56585569   0.494754057 -2.305687e-01  -0.14154846
## 108  3.8172472  5.25760659  -0.233366703  7.486221e-01   2.05638800
## 109  4.2671229  4.93496894  -0.503925155  1.985658e-01   1.14353347
## 110 -5.2079244  0.01494534  -1.556227254  1.910293e+00   2.44275264
## 111 -4.3665873  0.38078187  -1.751555097  5.298462e-01   2.95025457
## 112 -3.9692258 -0.56113336  -1.092887157  1.130896e+00   2.94980621
## 113 -6.5199668 -1.61976663   0.811719748  1.467734e+00   1.14358518
## 114 -7.0199619 -0.41993778  -0.660399942  6.091201e-01   0.63045842
## 115 -7.1207666 -1.58989638   1.062866044  4.701629e-01   0.67761468
## 116 -2.8921622  2.50433089   1.243303176  6.820568e-01  -0.95685009
## 117 -4.5123452  1.59267520  -3.068289100  2.593203e-02  -0.29515707
## 118 -3.5325546  1.90800212  -1.177018284  9.030783e-01  -1.23870188
## 119 -3.4399144  1.54162511  -1.943678031  4.741004e-01  -3.31438023
## 120 -3.0571813  1.84698798   0.994681528  1.373763e+00  -3.58810962
## 121 -2.9152614  2.12451881  -1.375683935  1.108733e+00  -1.61436024
## 122 -3.4392803  2.34720307   0.020558912  2.139040e-01  -3.02221814
## 123  2.5055745 -3.63011525  -0.021757819 -3.808297e-01  -1.02677760
## 124  3.3086261 -2.06519198  -3.894065713  4.332998e-01  -1.59653310
## 125  3.0524997 -2.76646619   1.140909507 -2.661845e-01  -1.62936180
## 126  3.7624221 -2.83367282  -0.839650022  7.320154e-01  -0.51010886
## 127  3.1300866 -2.51345961   1.392342834 -4.454579e-01  -2.41569518
## 128  3.4337069 -2.51343547  -1.526692837  8.690611e-01   0.26395127
## 129  4.0382283 -4.17610764  -3.415536724  1.886979e+00  -0.95354316
## 130  3.7846775 -3.30906031  -1.055728368  1.349244e+00  -0.63209817
## 131  3.7076098 -2.91901904  -1.626424805  1.342448e+00  -0.59677096
## 132  4.7116486 -3.80309798  -1.443912855  2.106045e+00   0.02053512
## 133  4.5477516 -4.18748981  -2.064050674  1.905035e+00   0.20746643
## 134  4.1268727 -3.84282732  -2.689165637  1.766038e+00  -0.66765672
## 135  3.3192549 -2.97403301   0.695404588  5.414821e-01  -0.25996400
## 136  3.8545082 -3.00988102  -1.130001828  7.543853e-01  -0.35558887
## 137 -5.0767622  0.31278903  -3.366159006  1.033210e+00   3.58778465
## 138  3.6964644  5.96705658   1.393317887 -1.197506e+00  -0.17667039
## 139  4.9485056  3.70196592  -3.598371011 -9.782217e-02   1.50052827
## 140 -2.8890364  1.31840620   4.221809624  2.283473e+00  -2.28579076
## 141  4.4493764  4.46725769  -0.866908993  4.322685e-01   1.50655121
## 142  4.2030738  4.78035640   0.087863509  6.048494e-01   0.65709981
## 143  4.8703523  4.20484513   0.174322051  1.058000e+00  -1.80691581
## 144  5.0481427  3.94523319   1.436585166  6.873130e-01  -1.74970230
## 145  4.7450085  3.77313370   0.703814509  4.936411e-01  -1.15467242
## 146  4.3704427  3.44445708   2.356618306  8.734446e-01  -0.98326852
## 147  5.3456706  3.70292645   0.060684058  5.985558e-01  -0.05556605
## 148  4.7390971  3.79634363   1.347341164  1.199701e+00   0.16340361
## 149  3.7706118 -4.37343155  -0.899446883  4.215505e+00   0.75587518
## 150  3.7204495 -3.67330617  -0.039335967  3.466257e+00   0.43724101
## 151  3.2101806 -2.54669180   0.044274677  2.701198e+00  -1.61462376
## 152 -5.9796708 -3.80454994   1.493970431 -7.966796e+00  10.02310972
## 153 -7.6824939 -0.33557974  -0.512808657 -2.781129e+00   4.48729512
## 154 -7.3793007 -3.60172819   3.674898187 -5.085613e+00   5.17202648
## 155 -6.1380782 -2.64203690   1.743750521 -3.993479e+00   3.88213648
## 156 -4.1031278 -0.68765066   2.427646616 -2.443868e+00   5.64534988
## 157 -8.2427548 -3.01916294   1.698723634 -6.101912e+00   4.65575682
## 158 -5.2907930  0.72775124   1.205828763  1.790764e+00  -4.12915504
## 159 -4.7412204  0.87186105  -0.327576996  2.511633e+00  -1.38132417
## 160 -3.7496881  1.40505817   0.326932000  2.727273e+00  -2.14164369
## 161  4.7464233 -4.25351774  -1.710487938  2.525832e+00   1.52310965
## 162  1.9781980 -6.40283390  -0.018900147  7.543832e-01  -1.93779706
## 163  3.3613852 -3.96862626  -1.253411446  2.631728e+00  -0.01516742
## 164  3.1305598 -4.88618555  -1.050784412  2.099055e+00  -0.90856209
## 165  2.8654329 -4.79812442  -0.040393284  2.438727e+00   0.20551638
## 166  3.3383827 -3.25213410  -2.234725639  2.093663e+00  -1.66898379
## 167  2.7953062 -4.10512276  -1.200159292  2.947113e+00  -0.62331548
## 168 -5.7358379 -1.12911350   2.777714164 -2.957024e+00   4.75499479
## 169 -7.8866505 -4.21853081   2.192883757 -5.878186e+00   4.61318069
## 170 -6.6492985 -0.23124726  -0.041132720 -1.374915e+00   3.53166110
## 171  5.2126429  4.21801810   0.488960616  9.520420e-01  -0.38170817
## 172  4.4508148  4.18800382  -0.617200355  1.578992e+00   0.70500034
## 173  4.4207882  4.85243350   0.300363317  1.017117e+00   1.77998674
## 174 -3.2679312  1.41826989   1.569873348  2.300229e+00  -2.01806426
## 175 -3.5000223  3.06858770  -5.823038561  5.265596e-02   2.55556762
## 176 -2.1388569  2.23347960   0.642516124  1.963618e+00  -1.70478029
## 177 -4.2042583  1.03643439   4.331095717 -3.252537e-02  -0.25838549
## 178 -6.1285329 -0.09983946   3.650557798 -2.774683e+00  -3.09109243
## 179 -6.3028242 -1.49991209   5.558994233 -9.921258e-01  -3.42042202
## 180 -5.8972163 -2.06369560   2.757823947 -1.681950e+00  -4.27895201
## 181 -5.1379924 -2.21501317   1.036557688 -4.374752e-02  -1.68709601
## 182 -3.5691298  1.71625640   3.066343166 -7.164882e-01  -0.72981449
## 183 -5.3632918  0.16434263   3.683157800 -1.492241e+00  -1.76459308
## 184 -4.6057467  1.81280270   3.356064188  4.392291e-02  -1.01563944
## 185 -3.8563082  0.19056229   3.154347522  5.670504e-01  -0.75553053
## 186  4.9754472  5.29879797   3.040815862 -5.332541e+00   1.63942036
## 187  5.4585198  4.91292479   2.736993530 -4.876026e+00   0.07966299
## 188  4.0746279 -4.29685002   2.491416215 -2.551312e+00   5.08374467
## 189  5.3308789 -7.04527750   2.623152766 -5.457928e+00   7.29847451
## 190  4.1448485 -3.97118059  -1.006296779 -2.426382e+00  -0.08744963
## 191  3.2718898 -6.45723806   1.871276938 -4.810553e+00   0.84423441
## 192  3.6200873 -5.78058046  -2.634200803 -5.461050e+00   1.70341934
## 193  2.6409598 -5.79715161   3.803857857 -3.709088e+00   0.14040356
## 194  2.9580712 -6.54847807   5.053464961 -7.086005e+00   2.66256920
## 195  2.0471854 -5.06667459   1.720579426 -3.631274e+00  -2.64348236
## 196  1.8177086 -7.04815744   4.439305118 -5.727018e+00   0.20177052
## 197  1.5803646 -6.06030215   2.435957366 -3.567580e+00  -2.09646560
## 198  1.1473437 -6.37676448   6.925534654 -7.480639e+00  -0.83393450
## 199  2.4396026 -5.19859909   2.789287654 -2.598545e+00  -3.35238215
## 200  3.3232113 -6.03443575   2.502246564 -5.120821e+00  -0.05131587
## 201 -5.4957069  1.92632249  -2.788615909  8.873770e-01   2.40360568
## 202 -5.3511493  1.56832502  -2.021380638  2.971079e+00   2.86783815
## 203 -5.8105123  2.11441308  -4.664897848  3.693514e+00   4.66030644
## 204 -4.7092953  0.95490883   0.530684669  1.998758e+00   2.76222648
## 205 -4.5795939  1.02107504   1.060987789  2.144539e+00   2.92745968
## 206 -4.4283111  2.27633090   0.496317395  1.833900e+00   3.60368522
## 207  4.2755772  4.27974619  -2.216501200 -3.505962e+00   3.52182243
## 208  3.1806823  3.36059629   2.078326664 -3.608823e+00   1.43595253
## 209  4.7660000  4.54862630   0.238282196 -2.492251e+00   2.73457727
## 210  4.2249852  3.33124774   1.940618142 -4.673566e+00   3.16769565
## 211  3.4447265  2.78561211  -0.475497290 -4.960442e+00   4.55170153
## 212  4.3149440  2.83909041   4.915752846 -5.987613e+00   1.91232149
## 213  5.3465279  3.47155008  -0.491393818 -2.094088e+00   0.51595454
## 214  4.9673027  0.39296434  -3.106972618  1.159469e+00   1.89794279
## 215  4.7317293  2.81949423   5.115727190 -6.053066e+00   1.91053812
## 216 -5.5757500 -0.70002627   2.565621678 -2.842036e+00  -1.08984654
## 217 -6.2823783 -0.71104132   1.597876986 -1.855552e+00  -5.06826009
##              PC6          PC7          PC8          PC9         PC10
## 1   -3.170826747  3.677458779 -4.011982310 -2.703241760 -3.020892915
## 2   -0.798985743  3.451578949 -3.124451779  1.752130145  1.570788335
## 3   -0.790515562  2.279442183 -0.937906834  0.216244268 -1.563990972
## 4    2.173063770  1.855186236 -2.677702980  1.790077989 -2.224688923
## 5    0.228508037 -2.838300905 -0.056676681  2.141317416 -1.014067899
## 6    1.887730324 -3.072245062 -0.563635963  1.170978160 -0.331265659
## 7   -1.141062038 -1.179731606  0.152642312  0.420617186  0.942968142
## 8   -1.672272986 -0.671780363 -1.169825935 -0.735491093  2.991506379
## 9   -2.507031436  1.683055361 -0.460329401 -0.108805211  0.005205392
## 10   0.059093872 -3.597915346  0.112274112  2.838337389  1.664471096
## 11  -1.362032920  1.167925029 -0.020516179 -1.621397381  1.414260685
## 12   0.184179168 -2.427981450 -0.438400070 -0.515540574  0.166270340
## 13  -1.883111568 -1.881523602 -1.253154472 -2.234650393  0.098680035
## 14  -0.054361720  2.658596736 -3.383653783  1.210315860 -2.289571427
## 15   0.327521795  2.374235267  3.229524914 -1.460690514  0.185067220
## 16  -3.503303282  0.363157805 -0.853708414 -1.086873736  3.615319439
## 17   1.126573307  1.769763063 -3.144476568 -2.641487986 -1.317608025
## 18   3.910885728  2.531834382 -4.125334947  2.934114838  4.701010657
## 19   0.324520475 -0.959162838 -0.544335555  0.663042330 -2.015658480
## 20   0.196317928  4.125912284  0.453086370  0.212055920 -0.480790250
## 21  -1.488856744 -1.295704817  2.196145004  1.465823632 -0.181909301
## 22   2.579248685  0.299942611 -1.387564686  2.694921694  3.063075466
## 23   3.033235537  1.249944317 -3.656053955  1.567000905 -1.517055407
## 24   0.879374900 -0.469914193  5.572444249 -1.070097127 -0.488086018
## 25   2.807176250  0.798443978 -1.845833695  1.657398639 -3.502594550
## 26   0.829829951  1.376350037  1.903916255  1.166336202 -4.146573147
## 27  -2.894855352  1.477021277 -0.973684953 -1.911260214 -1.593582730
## 28  -2.819008623  0.531364701  0.288667049 -2.303789643  3.403023919
## 29  -5.973574769  1.879962984  0.650534758 -0.096321505  4.147239059
## 30   0.398125427  0.419849555 -1.257647156 -4.039877250  0.124830680
## 31   2.035027565  1.596900826  0.145606429  1.103826087 -0.886117022
## 32  -2.288624283  5.006636757 -1.394773102  0.730278624  3.145003888
## 33   0.865801626  1.725983523  1.109671170  4.003980728 -0.176881367
## 34  -0.858180545  2.449215244 -2.331405290  3.571931375 -0.257620863
## 35   1.845692415  0.740804299  0.127280265  2.240232994 -0.591349601
## 36  -2.545681185  3.394046466 -1.439437431  3.363821985 -2.279971175
## 37   0.981359462  1.265286859 -1.900138067  1.826744534 -0.480436326
## 38  -3.689446639  0.005377245 -1.394074336  3.824663736  0.836879927
## 39  -0.498554610  1.113977650 -1.125199082  5.208699655  0.923049035
## 40   0.968843624 -0.750623781 -0.230734117  0.054248487  0.017200360
## 41  -1.350346124 -1.702467563  1.521371710 -0.507076181 -0.965114676
## 42   3.213150723 -0.614161521 -0.090981272  0.983147001 -1.224392552
## 43  -1.286366307  2.768213207 -3.485544917  0.821622670  0.095055148
## 44  -0.815249719 -0.901447710 -0.103286235 -1.803501396 -1.109996739
## 45   2.547784916 -1.352343196  0.020791195 -0.740361267 -3.158140956
## 46  -2.081814186  0.783774633 -0.812663500 -0.288103092 -0.689543311
## 47   4.687426264  1.779184319 -2.567383194  5.813118095  7.052199282
## 48   1.191505722  1.461264901 -2.540619033  0.070790153 -0.815122295
## 49   0.679870004  1.842466991 -0.521671204  1.324088521 -1.677434302
## 50   2.163024904  1.488891909 -2.866037505  0.249215086 -3.005072150
## 51   7.540546495 -8.200518382 -5.403894018  0.056666816 -0.533703279
## 52   2.308322736  1.666173573  2.313647294  0.728115358  2.703538033
## 53  -1.585341941 -0.985720751 -0.462571830 -1.324247696 -0.168680587
## 54  -0.140929773  1.240664915 -1.382496235  1.236608254  1.773173881
## 55   3.548187915  2.010356036 -3.191134112 -1.505724015 -3.548785177
## 56   1.764294239  1.756550023 -0.793617475 -0.908773881  1.222706327
## 57   2.117026853  3.043506227  0.598366438  2.619604869  2.031152855
## 58   1.620460803  1.526303574  2.483668135  2.808472080 -1.832742733
## 59   0.160886960  5.697718435  2.369723707 -0.307700489  0.464539974
## 60  -0.133089055  5.755235835  1.215921884 -2.531434927  1.034144261
## 61   0.229779436 -2.918290573  0.011081851  6.192331059 -0.324031443
## 62  -2.900732966 -2.288675254  1.133651234  3.355115724 -1.197823487
## 63  -0.006842239  0.330345747 -3.995347660 -0.894447769 -0.891491572
## 64  -1.494622659 -3.325380074 -2.100575376  0.040579431 -0.718763604
## 65  -4.940497638 -0.645443883  0.392832838  3.709350864 -0.856195042
## 66  -2.424187886 -5.225823114 -0.748549386  0.175977247  1.925296626
## 67  -2.319833247  0.286867761 -3.425845612  0.272708069 -3.406650532
## 68  -0.296499533 -2.405041987 -1.540555043 -3.827677334  4.767969484
## 69  -1.939916425  4.923751262  0.131448989 -1.396477021  2.829379624
## 70   3.742193558  3.407911885 -3.804281648 -0.819717043  0.027037524
## 71  -1.824467499  4.895854683 -3.012405619 -1.725664333 -4.027520568
## 72  -1.849192508  4.090661638  2.046147400 -2.109810526  1.528546908
## 73  -1.935928950  2.713869601  0.571310427 -1.619131701  2.039772238
## 74  -1.917848494 -2.088109875  0.425888147 -2.217588100  2.837353854
## 75   1.074293214 -1.386029248  0.691251181 -3.821648104  1.968683007
## 76   4.196699292 -3.621892256 -0.070695576 -1.316259850 -2.311675583
## 77   2.828043801  0.721529734 -0.110430005 -3.334513886  3.334116707
## 78   2.394679315  0.730558682 -2.144699583  1.148590195  1.349848582
## 79   0.168834002 -1.803458210  0.727155806  0.410748642 -1.101699106
## 80  -2.535826148 -4.220467826 -2.539444321 -2.110979535  0.407579399
## 81   1.670521464 -1.178378559 -2.197385521  0.948315886 -1.115726500
## 82   1.116766745  1.004804811 -1.660826282 -0.610433708 -0.753525493
## 83   0.976293669 -0.823835943 -1.171749772  1.517174539  2.373152693
## 84  -3.640348268 -1.863641534 -2.600546479 -3.541672785  0.411008144
## 85   2.453708523  0.343910682 -2.702652306 -1.677404580  2.869253597
## 86  -2.932245099 -2.320990329 -1.661615917 -3.154894685  0.893879666
## 87   2.461353773 -0.050197721  3.026206919 -0.619081090 -1.083701997
## 88  -1.088867527  2.860899826 -4.472480753  3.992730589 -0.104465823
## 89  -2.373424312  3.402402873  0.110349353 -0.263856901 -2.870918188
## 90   4.072977778 -0.011196505  1.886241354  2.724656423 -0.661567527
## 91  -2.205813052  1.946173863 -4.339890332 -0.601201004  0.614953829
## 92  -3.977187667  0.794040690  0.801146443 -0.653557172 -0.333890395
## 93  -1.657520585 -1.491493328 -2.512976581 -3.133913584 -0.294437584
## 94   0.066737460 -3.374226270 -1.427603047 -1.698709196  1.320691675
## 95   3.485213437  3.971556864 -1.151257910  0.343310096 -2.281152711
## 96   1.914909901  2.138791205  0.383694272  1.604210825 -1.103118501
## 97  -2.289611464 -1.754568670  0.727753726  1.088743527 -1.239055887
## 98  -0.203679875  6.762564863  1.024756043  0.168975027  1.975730595
## 99   1.399331290 -2.673412281  1.778838975 -0.820114036 -1.225089813
## 100  5.472954082 -1.717116758 -0.265441624 -0.196585806 -1.400068555
## 101  0.438110650  2.698627059 -1.982081791 -2.587040121  4.302074898
## 102  2.251760319 -2.733366658 -2.611459030 -2.277904398  2.353577105
## 103 -0.397740028  3.614134111  0.147005093  0.454718826 -0.392343045
## 104 -2.602601647  1.373864189  3.011923210 -0.529177862  0.376420837
## 105 -2.033686369  0.848660815 -0.554702249 -1.578433772 -2.513185325
## 106 -0.700713686  1.435996723 -1.721410648 -2.758598804 -2.278667765
## 107 -0.776896176  0.551777113 -0.524467651 -0.656035476  1.253800823
## 108  0.238355946  1.111073048 -1.462891344  0.880773463 -0.684888139
## 109 -0.321953590  2.041294306 -1.194174373  0.986570911  2.891258575
## 110  0.372024032 -0.931842834  3.808132872  1.433398488  0.465493168
## 111  4.767495836  2.229760649  4.550581931 -3.133897616 -0.120867690
## 112  5.157769790  0.982924861  4.313657342 -2.201514705 -0.501116725
## 113  1.758537386 -0.352624162  2.456328257 -0.096800856 -1.875698606
## 114 -1.852441996  0.546242505  1.839606466 -0.618458700 -0.594619741
## 115  1.294342033  0.536282086 -0.598045732  3.116187552  1.123823980
## 116  2.811488692  1.994698190  2.195848534 -0.956672732  1.240282764
## 117  0.149957754  2.415066728  2.015277104  0.229037502 -2.381809162
## 118  0.751185662  4.092181916  1.595730647 -1.246542180 -0.080117792
## 119 -2.447972750  0.348330251  3.796050804  2.876222884  2.558746919
## 120 -0.948386868 -1.059550238  3.511114410  0.615783494  0.975704537
## 121  0.195435410  4.028117769  3.114253092 -2.994643560  0.774143467
## 122 -0.294362834  1.096482113  3.658029837  1.030290409  2.523751359
## 123  3.998751479  0.558101061  1.585867808  2.521873485 -0.390287276
## 124 -0.021948089 -0.833419632  3.507508784 -2.886344701  0.363735621
## 125  5.585196651 -0.507843888  3.075295296  0.883456230 -0.627189989
## 126  3.166860023  2.217987434  2.219110536 -2.970454764  1.582912177
## 127  4.711600592  0.008887174  2.568508523  1.043469248  1.266907670
## 128  4.243065039  2.721168814  2.544305524 -1.395085456  1.771160674
## 129 -0.495171740 -1.205178060  3.869633755 -1.964655193 -0.918936605
## 130  3.128219013  0.920289373  2.776612929 -2.294822030 -0.764828138
## 131  1.873874418  0.464338929  3.443337671 -1.479284788 -0.708364908
## 132  1.941715162 -0.218847971  4.077448487 -2.453166649 -0.999573242
## 133  1.479113179  0.188573638  3.584181557  0.194029605  3.086078008
## 134  1.891658571  0.186631695  3.401883621 -2.763755653 -1.911720491
## 135  5.428503950  0.226984366  2.470975967 -0.069824737 -1.982259440
## 136  2.514259631  1.763423580  2.244725227 -2.166933078  1.380641852
## 137  0.447650228 -0.192023940  6.185397600 -1.477717939 -0.046428996
## 138  2.030635162  0.800382131 -1.816566177  0.652297070 -0.139706360
## 139 -1.483343391  2.465496375 -1.887743724 -1.480812228 -1.296524424
## 140  3.439701629 -2.751866789 -0.097222584  0.132972910 -0.510454317
## 141  0.623975014  1.311532325 -2.151936266 -0.053388356 -0.846206716
## 142 -1.165751530 -3.992672461 -0.205774401  1.078368171 -0.269209119
## 143 -1.139113176 -3.983781153  1.899607097  3.093325242  0.221054144
## 144  0.216553560 -3.368371750 -0.320700841 -0.674482443 -3.486590122
## 145 -1.367361421 -1.973011032  1.667718598  1.308674083 -1.986224797
## 146  1.042913337 -3.499434887 -0.007631621  1.544276025 -3.003203238
## 147 -1.820889256 -2.060390593  1.233206290  1.292957755 -1.655252498
## 148  0.517555241 -2.080007847  0.248602646  5.036818275  0.666666617
## 149  0.371780842 -2.916726506 -1.061927980 -2.311391676  2.200846341
## 150 -0.852113336 -3.049780547 -1.374026281 -1.630760059  0.194368136
## 151 -1.190728725 -2.106589496 -1.533436896 -1.470828406  0.756302385
## 152 -1.477735362 -2.352124257  5.907199899 -5.543656881  0.655039159
## 153 -7.660840522 -0.895758495  0.060394979  2.596917092 -0.994495133
## 154  2.810196469 -0.012662245 -2.193159558  2.279253243  2.274391360
## 155  1.922462049  1.146027746 -1.294651705 -0.004739475  2.421525406
## 156  4.227383457 -2.185859609 -2.438204977 -2.458045300  2.205342997
## 157 -6.778465863  0.854850479  1.933905851  0.052747494 -3.196495447
## 158 -4.351111986 -1.267006784  1.072918809  0.823137820  0.937691572
## 159 -3.793207818 -0.760720522  1.117092647  3.741801711  2.476630825
## 160 -1.079332745 -4.020954034  1.419432222  1.789691410  0.710320234
## 161  2.107671405 -0.139288879  3.395578584 -0.020682060  1.398504023
## 162 -2.250317269  2.038814292  0.407675956 -0.191414704 -2.437514249
## 163 -0.154836555 -1.486267895 -0.973278876 -2.169830821 -0.567878514
## 164 -1.003791548 -1.765542767 -1.033907087 -1.496697586  1.452435050
## 165 -0.009046479 -1.457752817 -2.675173353 -0.268116993  0.738488250
## 166 -0.563514794 -2.916883361 -1.386312462 -3.375217306  0.577633098
## 167 -1.322359538 -0.857780774 -0.798352548 -1.723991819 -1.204066377
## 168  4.623489513 -1.719486724 -1.353100136 -0.335375441  2.465900014
## 169 -1.804170734  0.255438533  1.126275621  1.510635452  2.775065900
## 170 -5.828211214 -0.652082087  0.251745621  0.807698533 -0.010393731
## 171 -1.732955831 -2.620058697  1.651313890  0.341091179 -2.241159055
## 172 -2.456838402 -5.003685183  0.886019930  2.657144148 -0.316871999
## 173  0.709764404 -1.640512686 -0.217684555  4.565303337  2.281929871
## 174  0.086366057 -1.976910376  0.335042695  1.807707808 -0.068467227
## 175 -4.072478530 -0.393240840 -2.053331784  0.089294963 -1.717548694
## 176  0.537608387 -1.997969291  4.498906453  1.823824662  0.478109007
## 177  1.407356750 -1.934040641 -2.046154860 -1.609532619  0.071955679
## 178 -3.261272308 -1.270035665 -0.070667879 -0.790591709 -2.513013244
## 179  0.145746804 -1.767966252  0.468166543  0.813785017 -4.016763414
## 180 -1.314186488 -2.052746334  3.081041276  2.173848897 -3.442017662
## 181 -2.403269991 -0.697336956  1.749373246  2.036340113 -2.784718251
## 182  1.453937243  0.224537264 -2.873295465 -2.542065820  2.198769678
## 183  0.826188476 -1.515060336  1.407299189  2.005930415 -2.654376877
## 184  0.017051982 -0.716019852 -1.376390579 -1.408588422  2.448112070
## 185  3.649616009  3.892632618  0.697003103  0.095101979 -2.369452815
## 186 -0.243972032  3.209034438 -1.708677734 -5.225993108 -2.929269724
## 187 -1.881137025  2.101764204 -2.015882289 -5.591469076 -1.764378070
## 188 -2.254985766 -6.136286272  0.264654417 -3.258704084  0.874495274
## 189 -0.799095469 -6.076179243 -1.229906133 -4.450883314  4.509063986
## 190  1.700924082  0.455902627  3.568665093 -1.143954515 -1.026623064
## 191 -2.633779243  0.956489150 -1.596488704 -0.093469539  1.594905217
## 192 -3.291335106 -2.608666759  3.631871157 -1.570840205 -1.674310795
## 193 -0.106439372 -1.163604760 -2.174494963  0.391724508 -0.123299160
## 194 -3.770073806 -0.787972584 -1.173506047 -2.750709614 -2.458389358
## 195  0.056273463  1.500390483  2.509205034  3.185996916  4.605152182
## 196 -0.707583312 -2.680812573 -3.380820147 -0.837850277  1.485778725
## 197 -1.858903221 -0.356582442  1.014174886  2.028172018  2.417537721
## 198 -2.826495841 -1.818621179 -3.833712821  0.314767137  2.147278047
## 199 -1.377193595  1.660818147  1.752956615  2.836148766  4.284201672
## 200 -0.710811945 -2.034092613  4.038646849 -1.237046570  3.594188283
## 201 -0.632915739 -1.616913349  0.865957380  0.342538759 -1.331395279
## 202 -0.204786635 -3.348572961 -1.164754322 -1.415288386  0.073508737
## 203 -1.906514967 -2.845612377 -0.971900202  2.494637133  1.899159720
## 204  4.772859938 -3.677131596 -1.837566098 -0.643460607 -1.721373881
## 205  5.767047772 -1.180408770 -3.267846863 -1.276197945 -2.556599967
## 206  3.502551900 -1.814314501 -1.235822834 -1.397318137  0.137583839
## 207 -2.623373108  3.208510793  0.719444994  0.225335242 -2.233434648
## 208 -1.121983914  1.918407405  0.014842156  2.418174866  0.668852028
## 209 -3.894197423 -1.670506012  0.838648195  1.064179102  0.909961162
## 210 -0.642940314  1.789105008  1.372783517  0.610308098  1.369145335
## 211 -3.152787554  4.024941788  0.226299396 -1.111211212 -0.707911428
## 212  2.704867509 -1.534054210 -1.258580203  1.451000547 -4.033558900
## 213 -2.856731466  2.522458021  2.163131205 -0.128914643 -1.381663687
## 214 -3.164164976  4.215420321 -1.827341395  1.484457160 -2.416008723
## 215  2.916729250 -1.798864634  0.332385434  2.723291509 -2.788291497
## 216 -0.259147891 -2.566521721  1.882126870  3.505719708 -2.524071630
## 217 -3.455040066 -1.472893650  0.534275355  0.135758010 -2.770924256
##             PC11         PC12         PC13        PC14        PC15
## 1    3.895732211  0.037292818 -6.805952612 -7.55152919 -6.70599115
## 2   -5.120803635 -4.955117185  0.040614380 -5.09117280 -5.02231628
## 3   -0.416122768 -0.303363275  0.388669622 -0.18188613 -0.22267097
## 4   -2.293912265 -0.355434223 -0.724961144 -0.56835170 -2.55938219
## 5   -0.564526047  1.683023813 -1.040794425  1.35529054  1.06228829
## 6    0.839522896 -1.652623708  0.012794540  1.15323864  1.18944757
## 7    0.555160607  0.002054240 -2.236599722 -0.22148017  1.36838439
## 8   -4.039361543 -1.746556849 -0.305310461 -0.61024411  2.07422458
## 9   -2.302621837 -0.544973121  0.388886115  1.26321686  1.99183255
## 10   3.563704989 -0.717685353  1.623464223  1.82007210  0.14351398
## 11   0.682390273 -1.956538163  1.182365090 -0.43940502 -1.83549365
## 12   3.928683992 -1.647540244  1.093338175 -0.12154203 -0.34945766
## 13   0.563673322  0.059055041  0.277697182 -1.13222898  0.55916791
## 14  -0.308221733 -0.001430683  0.105704927 -0.27124927  2.25528916
## 15   4.430351632 -4.316423257 -1.314894695  0.37029628 -1.58183213
## 16   2.058641552 -1.366440782 -1.534290388  2.53949778 -1.14145090
## 17   0.256752673 -2.882371669  2.476845850  0.32524923 -1.51439208
## 18   1.357555650  1.083877692  5.034951204  2.40884888 -2.40440534
## 19  -0.007130463 -2.965008599  3.210332550 -0.25209780 -1.25911239
## 20   1.840181578  2.077818508  0.911602216  0.75279013  1.00022337
## 21  -0.300265915 -2.483675521  1.735631726 -1.94383994 -0.49116526
## 22   1.618525842  2.513445969  0.868730065  2.07587447 -0.07265097
## 23   1.999460106 -0.923828904  1.667443398 -0.71625429 -0.16467949
## 24  -0.084751664 -0.864120334  1.377563232 -1.76124304 -0.07984182
## 25   1.396323230 -1.129559509  1.285938730  1.48413701  0.78154613
## 26  -1.586310127 -0.356458037  1.212301068  0.30571497  0.12389417
## 27   1.240597744  2.863071866 -0.420771688  3.46772707 -2.76196269
## 28   1.637868756 -0.622474105 -1.385743774  0.97319321 -0.42421268
## 29   1.100940538 -3.754819464 -2.569403655 -0.02047360  0.36031960
## 30  -0.797635666 -1.947708876 -0.514519570  1.22624128 -4.58082556
## 31  -2.293629877  1.675223437 -2.974923637  2.18080512  0.98845869
## 32  -1.999052975  1.085023668  2.369110089  3.11254222 -2.44645140
## 33  -2.130370038 -0.534396642 -0.405995215 -1.16557847  0.30461969
## 34  -2.178369776  0.546277868 -1.622187170 -1.88339719  0.41895837
## 35  -2.128796500  2.601040558 -1.072448945 -1.60976328 -1.24071383
## 36  -2.006872182  1.139164789 -3.582585573 -3.38831198 -2.28735811
## 37  -2.532818247  1.092971545 -0.573497927 -2.08291780 -1.06425660
## 38   1.088420779  1.011602335 -1.473390419 -0.88579916 -0.04107044
## 39  -0.028190959  1.365183172 -0.461242925 -2.21127341  0.02430959
## 40   0.577940579  0.323047579  0.408465374 -0.74862165  1.35005821
## 41   2.860621884  0.882403212  0.019709131 -0.74584753 -0.25633403
## 42  -0.670879266 -0.387256506  1.080960513  0.11447930 -1.14953603
## 43  -1.362737281  0.054814064 -0.012664944  0.19209535  2.23852338
## 44   1.835309431  1.096539544  1.062301957 -0.07523126  1.08087865
## 45  -1.070362004 -0.013451862  0.179378972  0.16808014  0.12410950
## 46   2.034934211  2.070866774  1.644228540  0.36690890  1.57551964
## 47  -2.974447281  0.133876767  4.665443744  1.25558710 -4.59579188
## 48  -0.898094902 -0.340757280  0.265412221  0.08770461  1.66626275
## 49  -1.725502571  1.866504785 -1.157174841 -1.68802579 -0.26459362
## 50   1.147537228 -2.453060701  2.907926863 -0.45827449 -0.57974969
## 51  -0.635625730  1.433680446  0.537864958  0.63398012  2.27027987
## 52   4.320917996 -4.097279995  1.961327909  1.97272454 -2.91451500
## 53   3.265792521  1.146749779 -1.300295114 -0.29656480  0.24482762
## 54  -4.918135760 -1.362275689  0.645198380  1.61648859  2.12570743
## 55  -1.573675257 -0.084299972  0.046194261  0.49308211 -0.72243928
## 56  -1.895978110 -0.176993163  5.241663141 -5.88408764  4.97820703
## 57   1.293565019  1.354821673  2.374368617  1.05311843 -5.68697872
## 58   2.771893322  0.302090176 -1.386813362  0.90906000  0.85084815
## 59   0.107263477 -0.606666672 -0.406967695  0.65224330  1.05553475
## 60  -0.253163392 -0.896834519 -1.192318814  0.67980890  3.10147252
## 61   0.557894502 -1.535216499  0.924593441  1.54720835  0.51085444
## 62  -0.145164754 -3.234824887 -0.949312611 -0.48214448  2.87840903
## 63   2.027533903  1.848059337 -0.571822682  1.66266591  2.60874305
## 64   1.400762155 -0.582978789 -0.316370042 -0.98951054  1.32265088
## 65   1.829046186 -1.150192620  0.228868662  1.78445205  1.91573084
## 66   0.247983722 -1.360500337 -0.376104297 -0.99386017  0.72621040
## 67   3.042502910 -0.391938858  0.808952637 -1.42876131  2.36485942
## 68  -3.251615157 -2.146336423  0.136802489  1.69181789 -1.09394494
## 69  -1.356269412  0.117309571 -2.263511094  0.35389229 -0.08680673
## 70  -0.819779234 -0.192300978 -2.373124478  1.39117504 -1.10353699
## 71  -1.650252559  0.679270534  0.210411992  1.03356003  0.57100986
## 72  -1.045049736  0.779375774 -1.541959293 -0.25100847 -0.67095226
## 73  -1.909987418  1.368920710  0.517478532 -1.81836218  0.64026460
## 74  -1.213976648  0.206265693  0.594745283 -2.33161493 -0.16927828
## 75  -0.763853915  0.364064609  1.280958224 -3.01779793  0.40487625
## 76  -2.869606141 -1.487346922  0.433619433  1.15330865 -0.03814395
## 77  -0.877621229  1.755778969 -1.120250279 -2.29884435  0.55098496
## 78  -2.404646419  0.043232886 -0.134725144  0.35929130 -0.27395829
## 79   1.215538917  0.496123724 -0.388641555  0.44233338  0.01933279
## 80   3.159621912 -0.439207431 -0.172268736 -0.01069835 -0.48930503
## 81  -0.916240853 -0.997833047 -1.933201333  0.63869670  1.16026796
## 82   0.805442914  1.360939640  1.950674804  0.24816900 -0.16363975
## 83  -0.151204792 -0.829409555 -1.418154968 -1.46466902  0.46775291
## 84   2.786361796 -1.725001974 -0.485394726  0.75802723 -3.12791741
## 85   0.069442931 -1.254063846  0.111111378 -1.36866604  1.44285051
## 86   3.494327304 -1.460269011  0.220790271 -0.72414842 -1.12737199
## 87   0.332714069 -1.931145621  0.316752119  0.61494369 -1.83901878
## 88   0.377236813 -2.558986824 -0.180074696 -0.66680673  1.35071692
## 89   1.879503273 -1.783423283  1.670717267  2.31282200 -0.38484558
## 90  -0.762746474 -2.788267975 -2.255899042 -1.05610350 -0.17071598
## 91   3.002170857  1.121263856  2.047564550 -0.56087324  2.25648986
## 92   4.378187951  0.471657103 -0.389349229 -1.61987006  0.12747176
## 93   1.958499558  0.303336948  0.991149277 -0.14481885  1.61847551
## 94   1.661031777  1.202503353  1.052713234 -2.31025779  2.44741042
## 95   2.609957090  0.572055725 -1.434092765  0.05481327  2.58822734
## 96   0.787424126 -0.296384017  0.815382841  3.29781157  1.54685388
## 97  -1.141575907  0.862005195  0.300436908  0.15101602 -0.20136843
## 98   1.375609996 -0.344253472 -0.458491109  0.71117195  2.18686888
## 99  -3.311630251 -1.156802417  1.275571160 -1.06516356  0.56127348
## 100  0.533392681  0.453319996 -3.126655082  1.33835843 -0.09806827
## 101 -1.038663345  2.260551009 -0.316930661 -0.36961188  0.04797892
## 102 -1.113400881 -1.552746728 -0.853877806  0.27995294 -1.85248689
## 103  2.425337238  0.408824724 -1.174848145 -0.66451717  1.43037475
## 104  2.611823284 -2.553991243 -0.172986938 -1.59013742  1.70612771
## 105  0.332510608  0.244885740  0.629917357  2.09496050 -1.24745244
## 106 -0.415918469  2.013137299 -0.823063979  0.92781241 -0.52129114
## 107 -1.579538689 -1.104752678 -1.149042533 -0.56510704  1.89168374
## 108 -2.332568747 -1.864674719 -0.992724973  1.27849222  0.39515342
## 109 -0.383499366 -0.096397446  1.315540312  2.43415487  0.38653840
## 110 -1.107250288 -0.758317280  3.166883073 -3.38896019 -0.26223219
## 111  2.248251302  0.695140858  0.028469076  1.57313673  1.26482844
## 112  2.362487555 -0.252647464 -0.464508509  1.19873981 -0.36353656
## 113  0.026278291 -2.237848749  0.308568474 -0.40371135  0.41196936
## 114 -0.056100426  0.396976326  1.478757338  2.29887273  0.84483038
## 115  1.013972545  0.879437973  3.553256872  0.89623700 -0.14512612
## 116  0.407182749  1.942324144 -3.415123641 -1.15853204  0.83406949
## 117 -3.351648698  0.352445458 -0.904636963 -0.10728862 -0.14702384
## 118 -2.290710147  1.270117220  0.103190338  0.52235626  0.61574293
## 119 -0.923123644  1.376880748 -3.059713184  0.36915119  0.60915144
## 120 -1.706022766 -0.305455133 -2.052852096  0.12913207 -0.17242311
## 121 -1.683624886  1.464106527 -1.302273195  2.19488317 -0.17251746
## 122 -0.188050579  2.971450270 -0.844265274  0.09420686 -0.85180851
## 123 -0.444221483 -2.926366092 -1.678056038 -0.82003149  0.21611736
## 124  1.805728926 -1.699871259 -0.167204431 -0.90143544 -1.62171649
## 125  0.866971264 -2.930524499 -1.855346973 -1.34390018  0.29027885
## 126  0.366829051  0.242646599  1.277134003 -1.55591583  1.51948187
## 127  1.046483031 -3.178355243 -1.482145075  0.46610577 -0.67858923
## 128  0.248635234 -0.387924064  1.891538573 -0.30456990  0.24124882
## 129  1.614876212  0.751423657  0.544628361 -0.53497747  0.27572093
## 130 -0.317204308 -0.189341790  1.215146414 -0.26605159  0.25614585
## 131  0.302285698 -0.416614430  1.115315002 -0.15189597 -0.19090152
## 132  0.755237352  1.230340112  0.915385600  0.77966873  0.67971602
## 133 -0.761224406  0.262283313  2.009231646  1.17315443 -0.05130532
## 134  0.850138511  1.696130530  1.494995043  0.55798218  1.33954298
## 135 -0.309361835 -2.086324496  0.123959893  0.12175269  0.15910724
## 136  0.709527801  0.726592122  1.590643164 -1.26043922  1.00302971
## 137 -0.103371977  0.093565945 -0.160452772 -0.04993167  0.31684843
## 138 -1.970907705 -1.670123032 -0.995870172 -0.67922674  0.08791772
## 139  0.942508294  1.416114309 -0.230900599  1.43893342 -0.85093730
## 140 -2.481946974  0.935511522 -0.953684476 -0.51081887  0.07381548
## 141 -1.917970994 -0.379666517 -0.870522112  1.66427989 -0.17552794
## 142  1.296288773  0.673768887 -0.007875464 -1.98586152  1.54617110
## 143  2.851822102  2.736974584  3.312720102  1.31111897 -3.30426569
## 144  0.701707287  0.762592120  0.702023516  0.52267029 -2.50171713
## 145  1.045740057  3.923786484 -0.303021838  0.87425587 -1.57888362
## 146  0.974068555  1.107123280  0.761388347 -0.28761615 -2.61062003
## 147  0.035627910  3.467350996 -1.256606296  0.05449081 -1.49754635
## 148  0.651049302  4.505409197  1.021615664  1.81714920 -3.34879381
## 149 -0.982569058 -1.207823618  0.179288192  1.77559159  0.57726323
## 150 -2.244065694  1.492665289  0.762343950  2.00641519  1.50522199
## 151 -1.515052626  2.088920900  0.300669643  1.81900042  1.87376537
## 152 -1.807906589  2.075908689  0.505424370  0.52505725 -2.34880339
## 153 -0.358597752 -1.923879657 -0.646517180  3.29755964  0.06471165
## 154  2.278601097  4.119517534  1.216064716 -1.90069065 -0.42973972
## 155  1.418525798  4.614983293 -3.156561753 -1.61019713  1.05547405
## 156  0.943440982  1.574260772 -3.913953423 -1.21351779  0.28596210
## 157 -1.592089368 -1.253822290  1.255923836  2.22090937  1.52646523
## 158 -1.762936626 -0.114521785 -0.246218976 -1.09616843  0.80836928
## 159 -0.986587694  1.584760667 -1.276240284  1.07043267  0.02410863
## 160 -2.610866190 -0.109467953 -0.366865148 -0.83706500  0.34142323
## 161 -1.839979921 -0.403420554 -0.032190041  0.39994790  0.70943048
## 162  1.738625087  1.818144709  1.239631992  1.14294264  2.30602414
## 163 -0.171393107  2.542540683  0.703314390  1.69633684  1.69313180
## 164 -1.022961171  0.478480058 -0.215516360  1.49335560  2.21710842
## 165 -1.169974645 -0.559813886 -0.538531292  1.90319637  2.46191245
## 166 -0.414506488  1.306582613  0.833620251  1.34119016  1.05845595
## 167  1.149258019  1.992075316  0.877793314  1.72574233  1.57813212
## 168  1.692996213  3.299016695 -0.385766753 -3.17679897 -0.69341109
## 169  0.936845754  4.723198329  1.466571260 -2.57481915  0.13651792
## 170 -0.999813016 -0.889137784 -1.265287665  3.68680293 -0.33892763
## 171  1.504152989  2.856615460 -1.366860181  0.93323008 -1.34457774
## 172  1.362963840  0.617598826 -0.311143397 -1.10384667  1.07037467
## 173  0.564983395  2.256864451 -0.870290017  0.49137617 -1.26791344
## 174 -3.467484645  2.316076882 -1.017531870  0.73251824 -0.51374318
## 175  3.567269337 -1.338383870 -2.932929137  3.59944224 -2.07222675
## 176 -0.881747969  3.514865243 -3.289200474 -0.69716462  1.54642169
## 177 -2.405292490  0.509879972 -4.116325062 -0.21889935 -2.27729214
## 178 -0.835801124  0.077034909  2.607167686 -3.09498648 -0.24870853
## 179 -0.943876707 -2.792596929  2.741216674 -1.23182108 -1.49393577
## 180 -2.904748633 -0.314226453  0.917184881 -0.18592715 -0.11735190
## 181 -4.017705875  0.517416854  0.478414392 -0.47502613  0.89770970
## 182  0.659606893  0.706644016 -1.783779796 -1.49601869  0.51849080
## 183 -1.526760381  1.231839502 -2.163893979 -0.46083294 -0.93066086
## 184  1.784421717  0.486430427  0.543558460 -3.39018102  1.05067792
## 185 -1.180484939 -0.348322283 -1.966556526  3.20326659  0.42050615
## 186 -0.930192586  2.149625750  2.965795968  0.62486226 -4.01085307
## 187 -1.805042581  2.733051305  2.319041915 -1.20039437 -4.52463841
## 188 -2.679797294  2.180216369  1.243577660  0.81697474 -0.84164108
## 189 -4.625566165 -1.879959825 -4.255061681  3.89993360 -0.36431429
## 190 -2.143319798 -1.080850783 -0.214997984  0.57701676 -0.47960859
## 191 -1.307907120  0.667082178  0.572103354  1.09743415  1.62320886
## 192 -0.336301617  1.513593806  1.751480650 -2.83937985 -2.99308228
## 193 -0.796275323 -0.992033495  2.436548322  1.59040823 -0.57611981
## 194 -2.122813529 -0.671114625  1.732291051 -0.17342622 -0.92265609
## 195  3.792232135 -0.800482808  1.752406073  0.87444388 -1.47841801
## 196 -0.421844568 -1.486921841 -0.459672550  0.34434563 -0.01639625
## 197  1.862363218 -2.664750834 -2.074908676  0.19938852 -1.14413063
## 198  1.680626047 -2.060101113 -1.551098376 -0.16884880 -0.30795553
## 199  2.844462483 -0.989526197  1.726728262  0.33144917 -0.27019558
## 200  0.377367415 -4.198296984 -2.919797487  0.79913563 -0.78769246
## 201 -0.675946353 -0.606975800  0.224856028  1.42969999  0.96426628
## 202 -0.450229438 -2.897636142  2.020661074 -1.40043293  0.73098497
## 203 -0.507603475 -0.784811969  1.760435476 -1.61089728 -0.89126806
## 204  2.305943726 -1.733186369 -0.414006181  0.12527665  1.24567652
## 205  2.149443186 -2.144762541 -1.152652459  0.15632945  1.19925967
## 206  0.800046635  0.175086149  1.319581045 -1.28317009  1.20123431
## 207 -3.631512785 -0.682926839  1.485336166 -0.56096474 -0.10822946
## 208  2.766983943  1.463563899 -0.564866362 -2.33777759  2.26651292
## 209 -1.797658027 -2.656742147  1.861243582 -4.53778583  0.96678628
## 210  3.039031787  0.755550227 -1.233501497 -2.10419557  1.84083023
## 211 -1.039210560 -0.978671700  0.112721604 -0.16068766  2.71219633
## 212  2.883936741 -0.753924067 -3.155419410  0.09955020 -1.69044337
## 213  0.734542253 -0.519337179  0.854953928 -1.38669512 -0.11827946
## 214 -0.294989468 -0.198071662 -0.025806385 -0.02022211  1.10285869
## 215  2.191950326 -2.007801915 -2.819670131  1.16328143 -0.73659741
## 216 -3.335989750  2.110948067 -1.818981246 -0.37229941 -1.80512324
## 217 -0.615120904 -0.135581120  2.854303963 -2.39077842  0.14978358
##             PC16        PC17         PC18          PC19        PC20
## 1    0.168139071  4.13309127  2.206675060  0.6479573896  0.06821046
## 2    3.047035279  5.15820730  2.089589603 -4.4823273797 -0.04998773
## 3   -1.463218695 -0.64044006  0.866020065  0.2458825042 -1.46032311
## 4    1.457599166 -0.40935785 -0.309529306  2.3549384425 -1.07011804
## 5   -0.534724655  2.17438420 -0.778015068  1.5563444043  0.05532582
## 6    0.085094411  0.56472155  0.394765139  0.6088461717  1.01976203
## 7    0.069214097  0.83645116 -1.249735792  1.1513524834 -0.36621235
## 8   -1.118684972  1.25143916  0.864978183  0.9165376467  0.18913679
## 9   -2.842988395 -0.01318985 -0.386129638 -0.7391989501  0.56484102
## 10   0.465496647  1.77609785  1.083873077 -0.3008096070  1.52486241
## 11   3.491119036  4.96957486 -6.906196413 -4.4929431363  1.33425201
## 12   0.986949243  3.32276029 -1.980585167 -0.9789085863  2.55776593
## 13  -1.126890283  1.35145412  2.331801605  0.0964168727  0.96205106
## 14  -0.976630415 -0.55035122 -0.377244147 -0.7236052061  1.85673754
## 15  -1.669834028 -5.76318556 -1.622951159 -0.9443012014 -0.37710543
## 16  -2.275082624  1.90247412 -1.732980069 -0.3860209622  0.17692355
## 17  -2.103622259  0.69208000 -1.559118125  3.4390320625  1.23865605
## 18  -1.011609819  1.27546414  1.237353462 -0.3685892446 -0.27718998
## 19  -1.169564943 -0.45632023 -0.070670446  2.0266383196  0.93571746
## 20   0.032921274  0.01898049  0.744274101  0.0105074291  0.54074379
## 21  -1.388956916  0.03086156 -0.323728286  0.7738062563  1.27393152
## 22   0.926654263 -0.38797364  0.710271901 -0.5550948383 -1.22767108
## 23  -2.020619449  0.50926048 -0.744363312  0.6463117695  0.54904196
## 24  -0.728673295  1.58856517 -0.423365374  1.8321644179  0.37892740
## 25   2.496647478 -0.79078133 -0.664774569 -0.4935265536 -0.39299628
## 26  -0.865813748  0.81369603 -0.170871109 -0.0004995842  0.95564512
## 27   0.208296002  0.21970562  3.309963011 -0.8261081404  0.54330582
## 28  -1.364116121 -1.25667231 -0.593448123 -0.6229265568  1.38107205
## 29  -0.765392027  0.55253705 -1.287380194 -3.4204316171  1.41583895
## 30  -1.744601900 -2.19535223 -1.360317172  1.9769168164  1.64119800
## 31   0.696572810  0.10255011  0.387095465  0.8756253278 -0.54068830
## 32  -1.547540660 -1.97405388  0.876964446 -1.6444636655  0.16797909
## 33  -1.013400359 -1.44828846 -3.198335265  1.2247651029  1.53571632
## 34   1.403908856 -1.23842510 -1.744242602  1.9362295120  0.42758031
## 35   1.389059145 -0.95940649  0.158000605  1.5760837306  0.34637457
## 36   3.425159903  3.15891811  1.598772077  2.5356907101  1.18486214
## 37   1.112075565 -0.76322980  0.643525043  0.9914243889  1.45309397
## 38   0.605099043 -1.73649562  0.383766071  1.7085519686 -2.14081164
## 39   0.737344949 -1.72306047 -1.030556815 -0.6380760683 -1.50436379
## 40   1.286801283 -1.81977415 -0.172781107 -1.2680121952  0.81825859
## 41   0.427949458 -1.74962601  1.308193832 -0.9170794109  0.58504269
## 42  -1.361685325 -1.53123459 -0.489602465 -1.4231046649  0.74792831
## 43   0.188455991 -1.03588365 -1.074891850  0.1228796961  1.57666880
## 44   0.420736398 -1.79345113 -0.146318034 -1.7250142950  1.94288265
## 45  -0.691722312 -1.52556033 -0.761068734 -1.2188512099  1.37903625
## 46   0.381595010 -1.50158354 -0.547970060 -2.1573965928  1.57267464
## 47   1.418065187 -0.50523163  2.622909484 -2.2152888920  0.64888563
## 48  -0.392329587 -1.14222356 -0.445587554 -0.7695608636  1.37520820
## 49   1.153322201 -1.18154941 -0.637899907  1.5833612690  1.01993454
## 50  -1.851109275  0.68100829 -0.436917247  1.6624350320  0.90638147
## 51   0.867522479  2.34246171  0.071606910  1.3718895187 -0.09755010
## 52  -0.380709522 -5.89457301  0.336354876 -0.8816799543 -0.90664931
## 53   0.592084050  1.45786547 -0.170699867 -1.2668810151 -0.71563365
## 54  -1.411914597 -0.12239654 -0.652730232  1.2034871654 -1.63475385
## 55  -0.406215858  0.70412744  0.745458076 -0.2841451343 -0.33233551
## 56   2.170149632  1.43702925  7.158682922  0.6374022332  0.01523222
## 57   4.782643069  5.21777168 -6.219551004 -0.5209097440  0.89889243
## 58   0.992585929  1.06813591 -0.716365105 -0.0519880223  0.57547907
## 59  -1.029966877  1.01153081  0.264061073  1.1599016724 -0.15315300
## 60  -0.478024215  0.56850436  1.367939010 -0.2241156177  1.21011361
## 61  -1.092759966  1.11472501  3.544642764 -1.9053032508  0.23029686
## 62  -0.698460079  0.46863510 -0.623999274 -0.4417273005  0.25068477
## 63   1.406919388  0.46839091  0.198265672  0.7969881034 -2.23236674
## 64  -0.208993562  1.62332115 -1.319919064  0.2071261558 -2.07651827
## 65   3.032231160 -0.29855316 -1.492543099 -0.0677571102 -0.62301916
## 66  -0.872442127  0.65014783  0.239553764 -0.9649203069 -1.65842828
## 67   1.511818079  2.05558944 -0.372266956  0.8710355384 -1.64113814
## 68  -1.949758885  0.64103348  0.379864481  1.8364229210 -1.20145950
## 69   0.606858790  0.71779845 -1.677997716  1.2362001495 -1.18753079
## 70  -1.323954624 -0.14023630  0.495633780  0.8866724796 -1.74168251
## 71   1.317871621 -2.53983256 -1.306616971  0.0027850027  0.44369683
## 72   1.139554500  0.35956949 -2.540130290  1.2032916284 -1.27062914
## 73  -0.143315562  1.18488563  2.248056421  0.7558143741  0.96699642
## 74  -0.945765188 -1.11476290  0.495571976 -0.0941022704  1.21484336
## 75   1.534161384 -1.97939950 -0.299069777  0.0821028667  1.14651727
## 76   0.575080062 -0.24049174  1.226004688 -0.4623559889  1.57932760
## 77   1.425292398 -1.50082980  1.214520669 -0.8114251040  0.42548947
## 78   0.201816287 -1.72937855 -0.877403477  0.2631350231  1.19021988
## 79   0.073429639 -1.68435046  1.338965292 -1.1002635412  0.27500115
## 80   0.238158707 -0.14426664  2.225439559  0.4363931082 -2.29298353
## 81  -0.055371054 -0.34079652 -1.900175269 -1.0539801646 -0.28804469
## 82   0.436358226 -2.42102206 -1.359547135 -1.3155284258  0.85485028
## 83   0.359600587 -2.44625367 -1.808076959 -0.6022044551 -1.10964039
## 84  -0.433892356  0.39042370  1.980560197  0.1509382592 -2.24329896
## 85   1.722129907 -1.61728165 -1.044489119 -1.3513162802  1.07227948
## 86   1.050993900 -1.63324821  1.825102072  0.2259086505 -1.90595685
## 87  -0.909605029  0.78734421  1.987370733  0.5050127690 -0.49782941
## 88  -2.117725806  1.95140380 -2.768655508  2.3083543755  0.10826836
## 89  -2.010893604  0.20399630  2.440793960  1.8976495408 -0.54771115
## 90   0.375814456 -0.31077864 -1.610752263 -1.1575500767 -0.29315570
## 91  -0.643616598  1.08806932 -0.256995077 -1.3568692053 -1.00719208
## 92  -0.314351858 -0.68695659  0.909016581  0.7108176013 -0.51747309
## 93   1.644455537 -1.98787947 -0.977373626 -1.9340834783 -0.26916370
## 94  -0.916926624  0.33352355  1.601644483 -0.5132388209 -0.14863391
## 95   0.458063290  1.01876563  0.978900223 -0.7166852758  0.69571968
## 96   4.368615500 -1.02057994  1.318718283  0.7212387160 -0.42216195
## 97  -2.181351922  2.10307044  1.730199641 -0.1244333616  1.25428664
## 98  -0.823554447  1.12616712  1.282681158  0.1472590012  0.91482721
## 99  -0.423703589  0.13199156  0.756724789 -0.9002772865  2.33122668
## 100  1.662559783 -0.10344029  0.013619410 -2.3520093795 -0.71726312
## 101 -0.870177758  2.70108304  0.935847972 -1.5835579957 -1.28159479
## 102 -1.915062590  1.20105834 -0.519893243  1.6494536690 -1.06024563
## 103 -0.426905706  0.06262879 -0.338515607  0.8710119525  0.71066241
## 104 -1.222900752 -0.31582434  0.496793023  0.5830493606  1.14469471
## 105 -2.345769025 -0.52105131  1.146642347 -0.3809153759 -0.65079582
## 106 -0.298029165 -1.70122605  1.036422097  1.7104540564 -1.79252416
## 107  0.662384347 -1.08867756 -0.337747658  1.0613183422 -0.88522379
## 108 -1.438097611 -0.53111121 -0.606638481  0.8856475741 -1.45176868
## 109 -0.292125724 -1.14329033  0.993853773  0.8077150882 -0.90645351
## 110 -2.384108357  1.20113190  0.712350704  0.6986440512  2.55590652
## 111  2.767958461  2.46873463  1.077924700  1.4381477197 -1.14357798
## 112  1.716546499  2.27738313  1.104499534  2.6742753078 -1.17547194
## 113 -1.650153240  3.13671690 -1.343437927  2.8827735970 -1.39114046
## 114  1.672422041  0.76992495 -0.363885639 -0.3105109405 -0.79226108
## 115 -0.894212401  1.05969679 -1.664174539  2.0138403481 -1.30409150
## 116  1.613683236 -0.04702152  0.206491258  0.4082134820 -0.39505647
## 117 -1.740637668 -0.58025239  1.300284579 -0.8833996069  0.32623513
## 118 -0.068385022 -0.61680574  1.134489640 -0.5704723591  0.54717160
## 119 -1.219285960 -0.04284112  0.229201048 -0.7185541565 -0.17631438
## 120 -0.911234307 -0.49394346  0.199453924 -0.5070626575 -0.17962788
## 121 -0.095229103 -1.25314591 -0.821453518 -0.1015691477 -0.18921180
## 122 -0.155470100 -0.78093653 -0.291867933 -0.4868660107  0.09884167
## 123 -0.600066725 -0.11885199 -1.131442267 -0.6658071572 -0.27437057
## 124 -0.123270642  0.73817144  3.485479995  1.8790146425 -1.16563370
## 125 -0.408192659  0.23161183 -0.885818166 -1.2209267757 -0.73464946
## 126  0.718973736 -0.02492284 -0.566368389 -0.6959276058  1.17110518
## 127  0.951352712 -0.16037777  0.893797334 -0.5008356516 -0.75025042
## 128  1.128989223  0.06484292 -0.316049471 -0.8055807405  0.59893493
## 129 -0.295102923  0.43464342  0.077428472 -1.0738492953  0.47300415
## 130 -2.152147448  1.58560734 -0.507590440 -1.3594517644  0.32867049
## 131 -1.301973111  0.85067115 -0.015999633 -0.9138879960  0.09779651
## 132  0.927370264 -0.49033862 -0.841345589 -0.4174857390  0.27777731
## 133  0.124099789 -0.42150007  0.351191211 -0.0343068184 -0.72138364
## 134  0.216980544  1.75586665  0.488178896 -1.3976358182  1.10057795
## 135 -0.138702770  1.41245658 -0.409777322 -1.6849862702  0.74698466
## 136  0.515054049 -0.96979406 -0.281231217 -0.1479588091  1.09213273
## 137  0.446503789  1.18067261 -0.851239879 -0.0585150687 -0.60397070
## 138 -0.351028907 -0.01556955 -0.533826011  0.7078081989 -0.68843696
## 139 -1.405151089 -1.41573405 -0.652257562  0.3116193460  0.36429266
## 140  0.747748663 -2.00196661  0.414410984 -1.1600464306  1.22811316
## 141 -1.338688395 -1.05093146 -0.062410202  1.4328006626 -0.22196605
## 142  0.225689809 -0.27909957  0.261575101  2.5542116539  1.06762397
## 143  0.295982690 -0.18642341  1.109881639  1.2751003483  2.43844169
## 144 -1.821074860 -0.51702084  0.022814474  1.7561507068  2.57041031
## 145 -0.328843452 -0.02084118 -0.686410661  1.5681851804  0.71653472
## 146  0.071067011 -0.82034935 -0.945243849  4.3447170276  1.68673015
## 147 -1.032006555  0.30123656  0.329370730  1.4113990853  0.02797976
## 148  1.425249420 -0.98304912  0.012206883  2.4279670609 -0.90903375
## 149 -0.574692761  1.08703980 -1.293590578  0.6558608558 -1.66879762
## 150 -0.980936844  1.65008255 -0.180963386 -0.0218676328 -1.38797305
## 151  0.109220154  0.46076994 -1.403196760 -1.0064936652 -2.26417184
## 152  1.406875777 -0.25972630 -2.646954565  0.7818604365 -1.91259710
## 153  4.192666858 -0.39975445  1.994023391 -1.0456737020  1.49830393
## 154 -4.246809296  0.12894572 -1.358856707 -0.1507670718  0.63448780
## 155 -2.356865148  0.28514360  0.569636225 -0.5525956752  2.99424655
## 156 -0.926718448 -1.48735417  0.260544574 -0.0265590486  3.43195101
## 157  2.585425051  1.51857759  1.282749512 -0.0397316955  0.74989068
## 158  1.143723471 -0.12139710  0.547322609 -0.4962929837  1.68318779
## 159  2.024802225 -0.02287437  0.154515458 -2.4966177226 -0.39291754
## 160  0.099455266 -1.12531289  0.368709433 -0.7157943241  0.88187418
## 161  0.431566875 -0.41783119 -0.515271026  0.7396852395 -0.61543303
## 162 -0.587584817  1.95087669 -0.187955746  0.4699110280  1.55110601
## 163 -0.083884703  1.21849092 -1.235696301 -0.4574140750 -0.14179521
## 164 -0.306518857  0.86950369 -1.625765108  0.7104979634 -1.73100816
## 165 -0.307984430  1.69756238 -1.003176672  1.1906063501 -0.23104261
## 166 -1.663777788  2.93764176 -0.181987273 -1.9023234117 -1.02544615
## 167 -0.439818079  2.62063377 -0.712940183 -2.5302599925 -0.34933526
## 168 -3.756341988 -0.50564137 -0.451005889 -0.5595240610  2.16704863
## 169 -4.420386882  0.11961906 -1.992823163  0.8276805229  0.91598519
## 170  1.980095949  0.75524603  2.675485891 -0.5096972769  2.45601396
## 171 -0.506707252  0.10852563 -1.041651490  1.0054074912  1.10973812
## 172 -0.464184615  0.27416047 -0.384544390  0.8617388287  0.83260564
## 173  1.690573499  0.39358010  0.933636642  1.5667528605 -1.28382886
## 174 -0.052039089 -1.94008714  0.031499669 -1.6829132632 -0.84016654
## 175 -1.418970476 -0.62154879 -1.251313619 -0.9198474117  6.16061283
## 176  0.262645552 -0.32117327  0.272110583  0.1061901771 -0.70421694
## 177 -2.045427069 -0.21466901 -0.418200563 -1.4918370621 -1.33710993
## 178  4.072208062 -3.38841866 -2.098473403  0.0801988727 -1.60395524
## 179  1.545241526 -0.30986608  0.443788511 -0.1975354036 -0.96884154
## 180 -0.365215500  0.32412043  0.214938580 -0.1135865665 -0.61492816
## 181 -1.001442352  0.95511532  0.614171088 -0.4108858451  0.40332508
## 182  3.436773423 -2.29555824  1.986845723  0.8364243202 -0.69894838
## 183 -1.624810416  0.90196307  1.019303284 -2.5549148039 -1.80523529
## 184  3.749673069 -2.13922118  0.761353430  1.1379161805  0.26988188
## 185  0.559235439  1.07320145  2.766304708 -1.6269409769  0.47194479
## 186  1.074360646 -1.26840601 -0.502266685 -0.3160794019  0.40454605
## 187  1.839031204 -0.63848156  0.598409785 -0.5325140517 -1.37375193
## 188 -0.834198336 -0.06760342 -0.473809721 -0.3497543220 -2.97747715
## 189  4.420001930  0.65648868  0.015464751  3.4896292986  4.80793923
## 190 -2.931385833  0.59855579 -1.452556650  1.3407947793 -3.17702204
## 191  1.078200748  0.50130283  0.707118387  0.8262579653  2.61486966
## 192 -0.841413198  0.93731767 -0.762380490 -2.9983329790 -2.12663208
## 193 -0.005802983  0.69491064  1.214081066 -1.1121782766 -0.33547270
## 194 -0.146882851  0.29869376  0.416517953  0.1300973131  0.18202362
## 195  0.364902130 -0.40160256  2.681393058  1.6798316391 -1.99649418
## 196 -0.664440742 -0.52967640 -0.334715343 -2.5142477802 -0.03492975
## 197 -1.022518587 -0.32276105  0.387420397  2.6324072382 -1.85324249
## 198 -1.990930712 -0.86859788  0.229806748 -1.3204118259 -0.78667742
## 199 -0.184634254  0.91246013  0.969394778  1.6412940187 -1.39002998
## 200  2.085689007 -0.80758590  0.560591985  2.9225996534  1.35857858
## 201  0.379847245 -0.95627117  0.005554457 -0.9707145739 -0.34581507
## 202 -0.263195979 -0.94172156 -0.460689646 -0.1723958763 -0.45073393
## 203  0.587477209  0.41760809 -2.254469683 -1.8969298505 -1.04266395
## 204  1.482701092  0.53636375 -0.525147687  1.4900775208 -0.58827676
## 205  1.403143127 -0.38463940 -0.313076746 -0.1446160540 -1.77215972
## 206  1.353420223 -0.78187479  0.156297159 -0.2163624912 -0.21525520
## 207 -1.019163252 -2.76566701 -1.208274685  0.3087205951 -1.04406165
## 208  0.507771541  0.16896959 -1.843218174 -0.4514114190 -1.91799762
## 209 -1.465101101 -2.58636283  0.262147357  1.2096324201  0.75287375
## 210  0.506202838 -0.29796980 -1.694460555 -0.2180341466 -2.02441349
## 211 -1.179643514 -0.39263788 -0.385663565 -0.6217064390 -1.41891298
## 212 -0.802500428 -2.11504446  0.800489761 -3.7888957141 -1.78698786
## 213 -1.603786010 -0.06237609  0.386032163 -1.1564528640 -0.08169296
## 214 -0.794963120 -0.12140961 -0.030937595  0.8745962944 -1.76242934
## 215 -0.056479808 -0.79014977  2.949265181 -2.9554222986 -0.16928332
## 216 -2.007486909  0.27619892  1.639255922 -2.9749707823 -2.84094594
## 217  3.742696918 -2.76018876 -2.667924078  0.6807900984 -1.13337421
##             PC21        PC22        PC23         PC24          PC25
## 1    3.872902789 -4.59215091 -0.71219640  2.292682825  4.6191031944
## 2   -6.357652076  6.93032588 -3.34185044 -1.533319142  1.5353152517
## 3   -1.462368737 -1.39207973  1.04533392  0.675977905  0.7528085970
## 4   -0.465628092  0.46520469  0.28786467 -0.667329915 -0.5989483495
## 5    0.780178309  1.94556708 -2.28843842  2.067452263 -1.7596954722
## 6    0.661779222 -0.07941119 -1.04281254 -0.793228475  1.0035647538
## 7    0.350259483 -1.06816410 -0.66874684  0.790397203 -1.1661231350
## 8   -0.323432724  0.89515662 -1.01634430 -0.889355762 -0.8386200651
## 9    0.200808918  0.06327214 -0.71764304  0.192706102  0.9570792481
## 10   1.122100044 -1.84828777 -1.38301281 -0.099374006  1.6968571756
## 11  -2.652713588 -2.31908007  4.54116002 -1.772256497 -0.3194586320
## 12  -0.742086163 -1.69257929  0.55397531 -1.324420854  0.8174056914
## 13  -1.110371971 -1.96062146 -0.21433200 -2.110483238 -1.0066226628
## 14  -1.333376890 -1.54609688  1.58476483 -1.259637846 -0.8266554531
## 15   0.252815498 -0.44039606  1.93016028 -0.263001861  0.4899015448
## 16   1.304794138 -1.04275057  0.12703849 -1.172034337 -1.9720870668
## 17  -0.305203700  0.72080087 -1.30284017  1.179425623 -2.0343843270
## 18   0.295756818  0.16664265 -1.48958885  2.259994946 -2.0774186375
## 19  -0.690400320  0.56783101  0.51064864 -0.369797565  0.2333798522
## 20  -0.083372252  1.93712658  1.50571174 -0.087660515 -1.1558157159
## 21  -1.304445072  0.30548797 -0.03744853  0.498114563 -0.8787977912
## 22  -0.049215823 -0.42185732  0.75473573 -0.373770152 -0.4381618005
## 23   0.045120806  0.46253127  0.16185047 -0.011774037 -0.0459012415
## 24  -0.290876994  0.85631492  0.13764554  0.822388479  0.1602692691
## 25   0.973376257  0.20403643 -0.51265612 -1.906949605  0.1433211650
## 26  -0.372657939 -0.68305024 -0.75475056 -1.825424153 -0.3910501538
## 27  -3.684167723 -2.86134146 -0.72314348 -0.385820060 -0.4776052517
## 28  -1.524822946 -0.77574273 -0.92964230  1.192269246 -0.7478200829
## 29   0.532541381 -2.76637624  0.05385627 -0.070489527 -0.5099267709
## 30  -0.376050891  0.64051937 -0.53447285 -0.640491269 -1.4212199293
## 31  -2.134247390 -0.81943921  0.23691978  0.431648945 -0.0271476780
## 32   1.349380981 -0.74748045 -0.01152221  0.563344638  0.4128032256
## 33   3.741066020  0.62290189 -0.54543599  1.070038157  0.2784993810
## 34   1.459660705 -0.09853389  1.65606400  0.126350121 -2.2396901486
## 35   2.118270827 -0.31753822  0.97669632  0.245280331 -2.0188589348
## 36   3.048737931 -1.11637727 -0.07504791 -1.264584382 -0.5557914310
## 37   0.162435348 -0.71242322  0.23143889 -0.041554799 -0.7705154065
## 38  -0.744155819  1.27766423  2.11411187 -1.325896081 -1.8403350293
## 39  -0.932758091  0.66857509  1.21186888  0.600765948 -2.1684158010
## 40  -0.616484658  0.28773245  0.42030221  0.571480405  0.1354200271
## 41  -1.167725517  1.06362036  1.06078019  0.052524163 -0.1859530622
## 42  -0.404400225 -1.50178070  1.34884828  1.294625337 -1.0779052702
## 43  -1.278086545 -0.43537615  0.25935750 -0.461917332  0.4358823984
## 44  -0.263856549  0.09513026  0.13917208  0.193319594  0.5269797414
## 45  -1.156082488 -1.07383317  1.35075525  0.868769075 -0.5924270345
## 46  -0.069515372  0.31681558 -0.20236153 -0.405384844  0.8850962968
## 47  -0.685908663  0.26225394 -2.29110335  0.260191352  1.8677441110
## 48  -0.811203827 -0.08782894  0.30255682 -1.142620904 -0.1973005866
## 49  -0.272329778 -0.51417773  1.29035421  0.142306373 -1.7954418092
## 50   0.598946315  0.60000102 -0.23262225 -0.016522076 -0.3403868901
## 51   0.031536543 -0.81832065  0.31202622 -0.554222578  0.3571472706
## 52  -0.168688216 -1.90504851  2.28069212  1.130152721  1.1112826752
## 53  -1.820321299 -1.58762804  0.41217732  0.068716054 -0.4296487164
## 54   0.203278080  0.33798023 -0.33433831 -0.088414872  0.1498798070
## 55   0.113161611 -0.88105092  1.78503566 -1.756994872 -0.5357547421
## 56   4.118288161 -1.29501026  7.21760697 -1.435976700 -0.2108093730
## 57  -0.608954048 -1.86347939  3.27812240 -0.602260185 -0.5656879407
## 58  -0.069331505  0.19281782 -0.93360387  0.912813693  1.9964895803
## 59   0.828897302 -0.65976389 -1.53148286  1.056495829  1.7298178341
## 60   0.668789285 -0.28522740 -0.87157822  0.668572041  0.9427533921
## 61   1.674730719 -0.21822514  1.95371942  2.040207638  0.8762804654
## 62  -1.193333733 -0.76603865  0.84855101  1.222543060  1.0682358951
## 63  -0.127045914  1.18248272  0.17584060  0.650607807 -1.4731522240
## 64  -0.121593840  1.07771395 -0.03950007  2.205830679 -0.9721125058
## 65   1.497558321 -0.32493853 -0.55451093 -1.318604829 -0.8141252063
## 66   0.352391749  1.35283217  0.23259817  1.149073729 -0.7353774571
## 67   0.007467732  1.50203014  1.40240973  2.531469091  0.0929450114
## 68   1.473520043 -2.08164615 -0.42147056  0.128154450  0.0925123617
## 69   0.460753815 -1.22916068  0.02049664  1.303933610 -1.1063639116
## 70  -0.171114891 -0.13257068  0.66537461 -0.392739067  2.6067396743
## 71  -0.030769953  2.01440554 -0.18452479 -0.616945184  3.4463246844
## 72   1.053154501  0.39590145  0.61154779 -0.213596051  0.7021006271
## 73   1.521692746  0.72987667  0.78230205 -0.923829646  1.2952865925
## 74   1.477662486  0.31602159 -0.33803522 -0.025221672 -0.2816671511
## 75   0.601730029  0.51520205  0.96462627 -0.401796995 -0.0868906827
## 76  -0.505680364 -0.23569806  1.05155523  1.787858261  0.5385434603
## 77   0.881028952  0.20198048  0.26126771  0.474178916 -1.7843719179
## 78  -0.211188822 -1.38311760  0.31008107 -0.228692635  0.0597099109
## 79  -2.028335896 -0.22162928  1.31518712  0.118643459 -0.3651901049
## 80  -0.063526447  0.49252451  0.95268945 -0.176772014  0.9900804300
## 81  -0.201441816 -1.21143525  0.49811001  0.144297807  0.0061507574
## 82   0.288377491 -0.72173950  0.52605305  1.705125109  0.6506109145
## 83  -1.419408179 -0.98933515  0.64487849  0.907866000 -1.0094569206
## 84  -1.376437387  0.95698088  0.09745380 -0.739226617 -0.4299126642
## 85  -0.905733188 -0.02591712 -1.29567448 -0.901203839  0.7755177257
## 86  -0.891535718  0.49825432  0.79930948 -0.322950384 -0.2836463505
## 87  -0.063166355  0.75150607  1.25408969  0.119737054 -0.4476851184
## 88   1.721956318  0.62460195 -0.00949152  0.523870028 -0.3295583505
## 89  -2.502913823  0.69568475  1.32090232 -0.469160511  0.7847907436
## 90  -0.047276333  0.87457796 -1.02305859  0.130815575 -1.5189557064
## 91   0.527292685  4.94606799 -0.64031495  1.657559864 -1.2994853809
## 92  -0.623092971  1.57841424  0.50570838 -1.545297885  0.3499978288
## 93   0.919343933  0.33447419 -0.52774859  0.405979739  0.7946025395
## 94   0.307637554  1.12816506  0.84259696  0.209873491 -1.4059609998
## 95   0.056366885  0.06362195 -1.29111775 -2.080005787 -0.8416785342
## 96  -1.614808200 -1.01921566 -1.78479714 -0.958131670 -1.3823500540
## 97  -0.049380180  0.41840371  1.41772314 -0.489677811 -1.7331688498
## 98   0.515961424 -1.46722236 -1.54417827  0.796796401  1.1534722703
## 99  -1.453273712  0.67382250 -1.16782685  2.274798605 -0.4535802043
## 100  0.207830151  0.89252935  1.22743595  0.016318836  0.7957988080
## 101  1.211432781  2.71960878  0.40295463  1.684286661 -0.6205097371
## 102  1.164143125 -0.55572633  0.74359602  0.483349908  0.3054319965
## 103  0.132762299 -0.48271993 -1.37628258  0.779327313 -0.4510620502
## 104 -0.949072754  0.01025415 -0.82945160  0.913512060  1.4545471620
## 105 -1.492998083 -1.17068384 -0.12239806  0.795606546 -0.0825492579
## 106 -1.732774128 -0.39751810  0.38608514  0.853116067 -0.6019830127
## 107 -0.802341979  1.26722881  0.70193147  0.124958003 -0.4230559129
## 108  0.040786902 -0.64888445  0.31344786  0.143889622  0.0007969009
## 109  0.572688660 -0.28606220 -0.44504217  1.359042128  0.8252189929
## 110  0.457818037 -1.23740078 -0.56326981 -1.038320803 -0.0242569382
## 111 -0.622844096  0.68870896 -0.23395542 -0.589812372 -0.2579313266
## 112 -1.307481089 -1.41454421 -1.73449734 -0.210770206 -0.1806005575
## 113 -0.223787614  0.22008645  0.21897029 -0.746692239  1.7124658338
## 114  1.174576169  0.97533134  0.23914653 -1.280997421 -0.1201251495
## 115  0.147901047 -1.48404475  0.17979304 -0.992182103  1.5606313808
## 116 -1.147816136  0.04686536 -0.05221648 -0.158678348 -1.1512866077
## 117 -0.619369503 -0.53644532 -0.74393626  0.921299658 -0.8426018218
## 118  0.573768585  2.10403161  0.17722757  0.391373662  0.4136151693
## 119  1.328911448  0.67386676 -0.84647510  1.651004154 -2.3063495455
## 120 -0.699396671 -0.27162613  0.05713397 -0.080728344 -0.3320674264
## 121  1.148745449  0.81235621 -1.16318755 -0.429516331  0.6974257180
## 122  0.892702994  0.18314050 -0.78041991  0.195921291 -0.6014849583
## 123 -0.112609447  0.08384160 -0.65841989 -0.427532150 -1.5280512188
## 124 -1.822334813  1.48259548  0.96066071 -1.793219956 -1.3298981695
## 125  0.156657268  0.88648069 -0.53901826 -0.743296033 -1.3908274934
## 126  0.771659994  1.12984634 -0.85188654 -0.810091714 -0.0626342713
## 127  0.064510365  0.45786259  0.90256032 -1.635336491 -0.6914512265
## 128  0.423772592 -1.01883743 -0.66484090  1.035183254  0.2417525089
## 129  0.668139615 -0.84096465 -0.06385134  0.080741987 -0.4797247141
## 130  0.256830545  0.12980177 -1.16122307  1.496437626 -2.0213602140
## 131 -0.102666140 -1.77770616 -0.35824273  1.490938721 -1.1797141522
## 132  1.006482586  1.42727042 -0.19313664  0.332381198  0.9467286426
## 133  0.884733712 -0.74750424 -0.39147299 -0.675275749  0.8568613032
## 134  0.833252595  2.30759734 -0.78220945  1.830156789  0.2100743262
## 135  1.107924065  0.27577495 -0.34954262 -0.525581377 -0.5285367093
## 136  0.910150613  0.31668476  0.09651962 -1.145720160  0.4793177996
## 137  2.135578664  1.60667342 -0.01042497 -2.553421065  0.5151163630
## 138 -0.075936036  0.94880460  0.31305922 -0.443852437 -0.9772073612
## 139 -1.096611219  0.28249326 -0.98138586 -0.453395695  1.0742943944
## 140  0.193597813 -0.32156577  0.16740605 -2.412356304  0.4406010528
## 141  0.328890082  0.06316974 -1.14547942 -0.711741614 -0.0949433497
## 142 -0.135237242 -0.32275313 -1.01539226 -1.799558739  0.2924043547
## 143  0.681325583 -0.68289371 -1.78463958  1.223756446  1.3719109493
## 144  0.785558634  0.46232804 -2.05416708 -2.512369148 -0.5835320876
## 145 -0.550570064  0.80280234 -0.75621814 -0.200435445 -0.0912671209
## 146 -0.123939284  0.77286564 -1.49539346 -1.315010239  0.0336325943
## 147 -0.460788634  0.25265153 -0.32432303 -1.428350887 -0.9029950175
## 148  0.935950040  0.72227723  0.21385091  0.789617748  0.2829259040
## 149  0.954723360  0.98024579 -0.16076112 -0.744830957  1.7060933926
## 150  0.263545033 -0.57275493  0.16195928 -1.605494342  0.9121215114
## 151  0.611721465 -1.13516194  0.00369772  0.109920762 -0.3067398364
## 152  1.796013484  1.01132576  0.41726493 -1.674216474  1.4394797092
## 153  1.193409035 -1.13287652 -0.98178749 -0.350527174 -2.9046580150
## 154 -2.208476233 -0.13575622  1.59403275 -0.253703794  1.0514394913
## 155 -2.698627514 -0.09501210 -0.70266205 -0.676367582  0.3436808785
## 156 -1.496553883 -1.10352512 -1.54083906 -0.109368117 -0.0501495795
## 157 -2.848604990 -1.69014727  0.26002957 -0.364137438 -1.9992040762
## 158 -0.355920157 -0.37120068 -1.27927119 -0.473374908  0.3080584308
## 159  1.357343210  0.44421737 -0.29436922 -0.819599897  1.6874796218
## 160 -0.904651765 -0.48151753 -0.09494740  0.151692621  1.2040637821
## 161  0.316195943 -0.52250853 -0.39329272 -1.479836241  0.7769306094
## 162  0.119913690  1.32475834  0.24265559  0.426218879  0.5409909429
## 163  0.578282614  0.89260218  0.22210920  0.392992914  1.5979074869
## 164 -0.013495233 -0.44215110  0.36295337 -1.907128873  1.3484553065
## 165 -0.078166433  0.71611528  0.03201192 -0.180020066  0.7612599498
## 166  1.424539233 -1.11360284 -1.45495719  0.678828101 -1.0551073598
## 167  1.094566911  1.44120546 -0.61904997  2.266368150 -0.0487524508
## 168 -1.552194777  0.20805308  0.39234193  0.441860318  0.0193293075
## 169 -2.425087824  0.50303258  1.32641956 -0.435586688  0.5848506460
## 170 -0.201324958 -1.27155582 -0.36453124 -1.170172932 -2.6126942670
## 171 -0.373411422  0.74089367 -0.52551035 -0.193769831 -0.2114603343
## 172 -0.037696456 -0.18653663 -0.83150397 -0.598464048  1.3722902113
## 173  0.918146676 -0.89559017 -0.44872203  0.379589784 -0.2318224324
## 174 -0.899656318 -0.06633125  1.58263284 -1.198202767  0.8089142408
## 175  4.319910599  5.40860296  3.40475468 -2.109852026  0.9551125651
## 176 -2.400809682  2.52193881  1.13676051 -0.173480949  1.0597266477
## 177  1.263052303  0.04074506  1.36585673 -1.159352127  0.7967048931
## 178 -0.172681509 -0.65089995 -1.44220525  0.964954428  1.2206088459
## 179  0.584804789  0.40563155  0.17669898  1.103080426  2.1774408164
## 180 -1.783370353 -0.70730886 -1.01614526  1.801816443  1.0885939260
## 181 -2.055844635 -1.43005955  0.12222606  2.258622925  1.5810941494
## 182  0.287314965 -1.65426266 -3.16587969 -0.395626372 -0.1070958966
## 183 -0.034431903  0.15146464  1.73944629  0.402781650  1.5127839521
## 184  0.221180814 -1.15246967 -2.14809049 -0.363586146  0.6560089823
## 185  1.189199581 -0.78320599 -0.98975440 -2.143136244  0.9709512829
## 186  0.826688545  0.84750531  0.96761981  2.127231516 -1.2567933037
## 187 -0.543368860  1.71709197  0.45209149 -0.002914794 -2.4425739937
## 188 -0.286749080 -1.33071038  0.39047204 -2.948994911  1.7421899754
## 189 -1.125087479  1.12485271  2.95004294  5.777451741  2.6601910097
## 190 -0.301186192 -1.46058510  0.33598206 -0.589378457 -1.0720486250
## 191 -0.985014636 -0.27652837 -0.29370425  0.287186188 -0.4724648572
## 192  1.226089351 -2.05419163 -2.59838530  1.256665380 -3.7486125803
## 193  3.131459414 -0.53427998 -1.87502299 -2.386000168  0.9055618080
## 194  2.578390187 -1.27513172 -0.25064902  1.907006808 -1.1001914252
## 195 -1.592749815  1.31656025  0.27347153 -0.510160965  1.7485391825
## 196  1.018619986 -0.32665426 -2.48397953 -2.877398219  0.2614776341
## 197 -0.744221351  2.34794286  0.19021841 -1.608624601 -0.2182160260
## 198 -0.675970961  0.86403868 -0.74315977 -1.438469629 -0.5940075622
## 199  1.247191137  0.09222431  1.93551678  1.782501955 -0.6368975176
## 200 -0.844831662  1.59690225  1.52870041  2.535033696 -0.8322796721
## 201 -0.675922391 -0.33108393  2.29859022 -1.442901551 -0.5082640431
## 202  0.010574403 -1.51857216 -0.41592548  0.796926591 -1.1467952751
## 203  0.918586607  1.56702988 -1.23943917  0.505754060  0.1641697908
## 204 -1.365154362 -0.15469917 -0.44858593  1.655668128 -0.4027939610
## 205 -0.208329878 -0.28543884 -0.91898324  1.102212470 -0.2988255349
## 206 -0.358411735 -1.76994876  0.02215462  0.283428984 -0.8095799504
## 207  1.594535487  1.84826080 -0.23716998 -0.312437317  0.2749346348
## 208 -0.423625452 -0.18171306 -0.91406362  0.678645727  0.6229993839
## 209  1.209232937 -0.72844838 -1.14443846 -2.038703836 -0.4107461311
## 210 -0.643791288 -0.84947214 -1.76959122 -0.304908423  0.5770809331
## 211 -0.308614763 -0.71394241 -0.32456076  0.440652860  1.3129322184
## 212  3.783110676  1.19425773 -1.34491029  0.425557388 -1.3260704202
## 213 -1.090776516 -0.69903658  1.98050195  0.882418049  0.8480228412
## 214 -2.597344537 -2.25045762 -0.09913191  0.296242047  1.6447460780
## 215  2.171694637  0.72130736  0.17152812  1.981295649  0.7149738294
## 216  0.112833057 -0.31566106  1.50170582  1.384093497  1.2285000853
## 217 -0.370779152  0.84057194 -1.82928516  1.816849156  2.4129787076
##             PC26         PC27          PC28         PC29         PC30
## 1   -0.711715675 -2.081445391  0.5018733517 -1.432983098 -2.086166475
## 2   -2.098236823  1.606752180 -0.1288998149 -1.774500595  0.912611113
## 3    0.025969636  0.808222218 -0.1028318759 -1.216604929  0.931269828
## 4    1.681869028 -0.686206127  1.2491886106  0.455032088  1.076552156
## 5    0.032901233 -1.076812504 -3.2498364881  0.220000988 -1.494204296
## 6   -0.380529108  0.051907192 -1.0364568129 -0.957967896 -0.032664013
## 7   -1.024270570  0.903360207 -1.4537986611 -0.543505937  0.070826158
## 8   -1.908419896  0.936934309 -0.5304716336 -2.558671895 -3.858611406
## 9    0.395157349  0.319215641 -0.0149566647 -0.168120322 -0.249264693
## 10  -0.126322393  0.895669424 -1.4407081148 -0.011142223 -0.219194228
## 11   1.599464416 -0.338894948  1.6321178950  2.090993573 -2.694676088
## 12   0.559062602  0.557297111 -0.8253073438  0.754699099 -1.399450895
## 13  -1.405302963  0.284563757 -0.9882755174  0.108953348 -0.802348408
## 14  -1.129221486 -0.404860197 -0.7448485281 -0.470140065 -0.283415192
## 15  -0.750737875 -1.658923609 -0.5846082679 -1.190006184 -2.252902044
## 16  -1.405048639  2.395186118 -1.5090361505 -1.367669127  1.685174419
## 17   0.105662936  0.590006118 -0.6655670826 -0.922983398  0.210163213
## 18   0.047136677 -3.002264139 -2.3511647649  0.502070388 -0.452258983
## 19  -0.343363743  0.398086357  0.0168362486  0.023093855  1.336145190
## 20  -0.256673544  0.679922687  1.1592127498 -1.180252683 -1.249335743
## 21   0.924669101 -0.328908992 -0.0216976777  0.909745657  1.308971175
## 22  -0.445646411  0.057390812  0.2808847944  0.418460945  0.050637041
## 23  -0.875707958 -0.134941402 -0.2704779060  0.838717035  0.192886267
## 24   1.289298667  0.765209105  0.7983962732 -0.938921198  1.934759171
## 25  -3.004741590  0.225216753 -0.5427467068  1.183651829 -0.431682800
## 26  -1.239551729 -1.547847107  0.5782858982  1.346545077  0.784191457
## 27  -1.191530657  2.204316840  0.2050402991  0.282325480  1.802933742
## 28  -1.107021101  0.703454744 -0.8282354558 -0.385764513 -1.234659339
## 29  -0.962951731 -0.393437583 -0.7471024069 -1.409877042  1.707944146
## 30  -0.862261097  0.280141674 -0.5547891114 -1.401270027  0.562846840
## 31   0.348337847  1.926424741 -0.2300227520  0.141488038 -0.885902687
## 32   0.138789955 -2.091821649 -0.2871390113  1.104685483  0.648899940
## 33   2.124561554  0.155411383 -1.7503562193  1.258861912 -0.655786750
## 34   1.042551897  0.553873160 -0.5478881788 -0.447192812  0.102126308
## 35   1.856721899  0.501262442 -0.7068631906  0.333400367 -0.364634777
## 36   0.940600551 -0.185693138 -0.7782973492 -0.160944772  1.737310713
## 37   0.406469142  1.018221309 -0.4123480488  0.319348656 -0.706551402
## 38   0.952999138 -1.513199673 -0.4053599541  0.471999478  2.120789840
## 39  -0.573242891 -0.732457557 -0.7546992337 -1.575420099  2.014676645
## 40   2.087098862 -0.065879585  0.3092734831 -0.438632021 -0.416498532
## 41   0.214414370 -1.024133593 -0.6823167071  0.687053923 -0.016802879
## 42  -0.264325816  0.739597838  0.4751858927 -1.785252130  0.925000384
## 43  -0.609261795  0.609538883  0.3964685515 -0.561636395 -0.790900803
## 44   1.373647406  0.199160609 -0.0681987812 -0.507577872 -0.251419110
## 45  -0.423199216  0.162163825  0.5389725866 -0.950904935  0.388363793
## 46   0.326059533 -0.447229842 -0.0303449168 -1.143489236  0.538710297
## 47  -1.446513729  2.442970448 -0.2908258972  0.166963730  0.097203484
## 48  -0.691213930 -1.558756246 -0.7269101213 -0.313751468 -0.646058617
## 49   0.685943150  0.604099973 -0.3990704256  0.049514266 -0.751831446
## 50  -0.916049100 -0.268678925 -0.5497540861 -0.078576146  0.104870192
## 51  -0.132596805 -0.980005448  0.4881284076 -0.372044192  1.004971189
## 52  -0.239591609 -2.511416301 -1.3383045406 -2.034266758 -2.359210500
## 53  -1.935805611  0.476181275 -0.6004749173 -0.097852000  1.403948939
## 54   1.228755669 -0.208298559 -1.5499128075  0.444388754 -0.353294050
## 55   0.991027585 -0.242760691 -0.3544964872  0.971477490  0.577798784
## 56   0.446026664  1.263814760 -3.5539300590 -1.017459089  1.626676267
## 57   3.712187747 -1.566010386  1.7883376416  2.292280998 -1.171682898
## 58  -0.840956028 -0.060519106 -1.0123838505 -0.972632632  0.454449842
## 59   1.317959207 -1.317675734 -0.0127404298  0.932410595  0.812842931
## 60   0.488495537 -1.002824281 -0.1634753735  2.235600226  1.975823229
## 61  -0.443920744  2.423009446  0.3408108825 -0.709886022  1.035850419
## 62  -1.143488106 -0.465262597  0.3332733145 -0.503827059  1.112631257
## 63  -0.640254767  1.384679755  3.9060617054 -0.441101138 -1.409176156
## 64   0.496800885  1.795449860  2.5033543572 -0.445142243  0.511861727
## 65  -1.617397118  0.797144224 -0.4823302197  0.496541336 -1.203793826
## 66   1.504438424  0.909318133  2.9821683937  0.218356518  0.684475649
## 67  -0.600473692  1.930281500  1.5682018183 -0.908065479 -1.257035776
## 68   0.502303743  0.441470047  0.6985366349 -0.060965338  0.548832218
## 69   0.238046368  2.839978200 -0.7490853137 -0.752259205 -0.034054959
## 70   0.770716594 -1.777502697 -0.1217451877  0.597233527 -0.199153402
## 71   0.031406851 -0.900685308  0.8349342065  1.300157611  1.080621743
## 72  -0.647702178  1.896008583 -0.5757516323 -1.620262706  0.727049110
## 73  -0.635604633  0.394357953  1.9046284816 -0.497983949 -1.378127327
## 74   0.870885577 -0.621941880  0.7954774718  0.276769616  0.533393438
## 75   0.187055479 -0.236103440 -0.4049806729 -0.088611399  0.895337294
## 76   1.435517287  0.289856396 -1.1692757949 -1.490969490 -1.250999147
## 77  -0.473583407  0.720958094  0.1858009473  0.286688220 -1.765999705
## 78  -0.702592712  1.531852812  0.5267773435 -0.435329412  0.307224610
## 79  -0.662228026 -0.970696235  0.1891249268  0.418393004 -0.186591242
## 80   0.042710433 -1.352624320  1.1020436866  1.850007364 -1.116478077
## 81  -0.653356625  0.186621163  0.6952360064 -1.106060819 -0.115700169
## 82  -0.670397969  1.299227499  0.3588528403 -1.013071119  0.161419063
## 83  -1.161065854 -0.492716657  0.0086013650 -0.140442854  1.870712958
## 84   1.880399307  1.186177181 -2.4548061779  3.101668244  1.535422406
## 85  -0.024807770  0.352508961  0.3465405497  0.170864094 -0.436100069
## 86   1.163346239 -0.124030640 -0.6011386422  1.086349677  0.565260290
## 87  -0.232759177 -0.598238357  0.3890120679 -0.157894595 -1.177992499
## 88  -1.174009526 -0.663695082  0.8125954666 -0.087333596 -1.715436911
## 89   0.664151884  0.420789567 -0.0040038271  2.190638005 -1.012860116
## 90  -0.800793626 -0.449926511 -0.0718684078  0.544612615  0.652792929
## 91   0.759007305 -3.110516722 -0.7542942658  0.153645441 -0.168888043
## 92   0.071576889 -0.252596614 -0.2056587526  0.686973827  0.846879188
## 93   0.737193347  0.389538834  0.7079776778 -0.794149969  0.732291891
## 94   0.582775007 -0.293932030  3.1498817926  0.699580203  0.787023475
## 95  -1.390628635 -0.731851965 -0.4207295419  0.659583767  0.003656129
## 96  -0.320203755  0.594354678 -1.1728508278  1.354046598 -1.849774576
## 97   0.370761899  0.003418206  0.9732410309  0.391024395 -2.041280343
## 98   0.929433382 -0.696962439 -0.7547107638  0.947048577  1.475298826
## 99   1.865804984 -1.812257511 -1.5847576199  0.425174787 -0.652339863
## 100 -0.497751868  0.028385079 -2.1009792506 -0.099290084  0.514489550
## 101 -0.812916946 -3.794665096 -1.0414626299 -1.402339693 -1.605178527
## 102  1.311494901 -0.093329574 -0.0069016357 -1.184373004  1.230735300
## 103 -0.690524300  0.643374066 -0.1399464068 -0.187749244 -0.360152450
## 104  1.160473964 -0.440364210  0.9486872140 -0.080975867 -0.317787363
## 105 -0.005163166  0.886570851  1.1478584689  0.075591467  0.036364224
## 106  0.280651545  1.366088762  0.0256804847 -0.266619275  1.105593798
## 107  0.088516837  0.211779572  0.7780418020 -0.846022718 -1.176211438
## 108  1.120049044  0.337176544 -0.0014105001 -0.062421904  0.084627050
## 109  1.255285208  1.907204081  0.1808644149  0.144397997  0.305495707
## 110 -0.131641600  0.122249007  1.6736110300  1.105138492  1.897674760
## 111  0.610465709 -0.103825890 -0.6485643943 -1.089953161 -0.965539208
## 112  0.770388716  0.527197704 -0.7119994451 -0.150051942 -0.036218002
## 113 -0.810854538  0.095293011  0.7777213335  0.007127254  1.435045552
## 114  0.035529189 -0.142278239  0.7261680653 -0.420911123  0.656135371
## 115 -1.978605031 -1.222181739  0.3834977683  0.490986819  2.719403637
## 116 -0.352525106  0.139640260 -1.0546699421  0.750727996  0.090643297
## 117  1.839530492 -0.280007866  1.5338760050  2.522555500 -0.651443203
## 118  1.469888270  0.356441870  0.3667089803  0.917811125  0.232413841
## 119  0.350552819  0.411458399 -0.3759974719  0.669450543 -2.067272276
## 120 -0.468731740  0.445986320  0.6968516188  1.034565789  0.085085795
## 121  0.793734957  0.832086965  0.4780778961 -0.020257562  0.913348809
## 122 -0.619843332  0.695554165  1.0035223003  0.015973419  0.386268451
## 123 -0.383465989  0.677428247  0.8088491960 -0.288775802  0.736222541
## 124  0.320213716 -0.290595988 -0.2923818312 -0.054436564 -1.853243886
## 125 -0.492874897 -0.349682202  0.4679428206 -0.115918743  0.853138619
## 126  0.227244463 -0.582738740 -0.0006861918 -0.283635295 -0.206780345
## 127 -0.757009085  0.031027449  0.8298213188 -0.426066241 -0.074640328
## 128 -0.794964712  0.327171848  1.1102584512 -0.003247195 -0.197430472
## 129  0.740438899 -0.074126644 -0.0769289268  0.089610789  0.337021579
## 130 -1.031757513 -1.450360143 -1.4842545641  0.234411792  0.091839396
## 131 -0.061129354  1.198951841  0.6772194865  0.143731390  0.471110083
## 132  0.625683070  0.534302437  1.1718961495 -0.977590594 -0.501157403
## 133  0.486984539  0.182725611  1.0447565919 -0.734211568 -0.034294933
## 134  0.515709191 -0.923168402 -0.8960683537 -0.147222862 -0.210668697
## 135 -0.326420065  0.221003243  0.9541919600 -1.247727369  0.776254229
## 136  0.544789995  0.069351346  1.0537207346 -0.916108209 -0.045727630
## 137 -0.443653297 -0.171424107  0.3818767275  0.238216756  0.915545756
## 138  0.051888040 -0.831898952  0.4729857670 -1.372808733 -1.691493210
## 139 -0.349451722  0.682217017  0.4717887252 -0.478586208  0.689930917
## 140 -2.121946316 -2.181159686 -0.3553086350  0.646632748 -0.244769703
## 141  0.758132330 -0.059328382 -0.4177783519  0.832903542 -0.539168798
## 142 -0.011027130 -0.943793863  0.4473103982  0.413150329  0.074788504
## 143  0.289960209  0.685134764  1.4091807529 -1.261896799  0.146906199
## 144 -1.591250825 -1.292458112  0.5283435604 -1.152329709 -1.298663712
## 145 -0.399170155 -1.261086285  0.9728393774 -0.531800250  0.330857796
## 146 -0.188797177 -0.928586454  0.4926934216 -1.160982445  0.483651989
## 147 -0.468974777 -0.876739009 -0.2473655878  0.712461751 -0.188612613
## 148 -0.183110725 -1.061840856 -0.2998331791 -0.319259451  0.402591654
## 149  1.198640159 -0.312444908 -0.2098956510 -0.972145488  0.135077859
## 150 -1.066378490 -0.528745666 -0.6470974995  0.968362460 -0.807057637
## 151 -1.220595491 -0.338473781  0.2441852568  1.596302267 -0.148324405
## 152 -1.548634138 -0.160303022 -1.7171868682 -2.185742753 -0.076633265
## 153  1.039108565 -1.516992158  1.4522144210 -2.343059366  0.071061989
## 154 -0.914068673  0.436428071 -1.2279946526 -0.606730584 -1.136132931
## 155  0.501133973  1.008299867  0.9617119505  0.503041534  0.143461290
## 156  1.805257727 -0.170837858 -0.1168505609  0.242671267  1.253521277
## 157  1.391495921 -1.633800768 -2.4860557927 -2.929458974  0.498796983
## 158  0.237030319 -0.184473833 -0.1900787533  0.270045449  0.912791891
## 159 -0.445113657 -0.685329136  0.3746357694  0.206125019  1.258010033
## 160  0.760109354 -0.769405467  0.0342445168  1.209928566  0.373414417
## 161 -0.167799195  0.335956870  1.3114469006 -0.113138940 -0.547102448
## 162 -0.467004270  0.100139225  1.0454889690 -0.624081245 -0.742361505
## 163  0.121763307  0.377798538  0.3052114514 -1.181127685 -0.321427372
## 164 -0.260360824  0.070292262  0.9253338939 -1.168762331  1.514840906
## 165  0.842469785 -0.008627146 -0.0295895199 -0.289108659 -0.529085868
## 166  0.066691108 -0.206193805  0.0289404006  1.857671110  0.425593687
## 167 -0.424777412 -1.014763267 -1.2161718819 -0.677267584  1.171990256
## 168  0.773628735  1.111248116  0.0455839679  1.093827280 -0.569761387
## 169  0.599219164  1.290517303 -0.1934422116 -1.928860733 -0.129452052
## 170  2.467469894 -2.689542724  1.2246555974 -3.113128609  0.457245294
## 171 -0.579303282 -0.821826075  0.4588917208 -0.751080376  0.286962513
## 172  0.278842505 -0.456113473 -1.3188114662 -0.040644789 -0.177610981
## 173 -0.273180995 -0.247548084  0.2103828678 -0.262244312 -0.065863990
## 174 -1.349853341 -1.865803921  0.2678015395  0.860843349  0.133525285
## 175  0.019764495  4.839007378 -3.5852598795  3.124765006 -0.161950408
## 176 -0.955290741  0.072471105  0.0280277847  0.333112142  0.506569112
## 177  0.459971422 -1.756684842  0.3069754655 -1.572473009  0.646247013
## 178  0.677495085  1.427624112 -0.7905147968  0.391412342  0.344144546
## 179  0.943435114  0.215895523 -1.3473746125 -0.778986087 -0.385641847
## 180  1.749772166 -0.111651362 -0.9551629334  0.530773612 -1.085371688
## 181  1.323477302  0.974648108 -0.2859440865 -0.296279255 -2.860662869
## 182 -0.957790219  0.847606911  0.5547606502  0.984599048 -2.723628816
## 183 -0.086008975  0.186005898 -0.6554310976  0.452428680 -0.558275969
## 184  0.248662473  1.019644038 -0.4809469363  1.365972227 -0.698544210
## 185 -0.127370740  0.588547357  0.3691984570  0.568948690  0.699629835
## 186  0.622168163  0.003076939  1.2442932420 -0.482070420  1.551420790
## 187 -2.011827307 -0.393494710 -0.4137221122 -0.485711452  1.002389487
## 188  0.237506812  0.083185740 -2.7263036637 -0.941326889 -1.642716853
## 189 -2.394963202 -1.033813727 -0.1452301069  1.322106819  0.922618088
## 190  0.562976857  2.277922049 -0.9217172092  0.317072004 -0.103206419
## 191 -0.048713503 -0.036127907  1.5798774576  0.009298139 -0.547773068
## 192 -0.500065781  2.279676629 -2.2993504528  2.235993256 -0.406675795
## 193  0.682084369  1.138648203  1.8781596480  0.009938959 -1.439970013
## 194 -4.929354761 -0.127470746  1.8003927368  3.022207651 -0.900718089
## 195  1.468405996  0.456094916  0.9522276638  0.667538071 -1.121553414
## 196  3.411711493  0.482667208  1.1336180224 -0.696668844  0.657836387
## 197  0.780323276 -0.507956334  0.3890092838  1.552641110 -1.375673637
## 198  4.618156054  0.310774185 -0.4787307504 -0.694781797  0.067616056
## 199 -2.194890608  1.142117390  1.4787892036 -0.724759769 -0.150231643
## 200 -2.162756725 -1.225365666  1.0785882685  2.562453344  1.156378041
## 201 -1.468120750  0.887396974  0.1616903660  1.182648572 -0.283414248
## 202 -0.335539505 -1.451725955  0.1165116752  1.486306086  0.432275154
## 203 -1.513115743 -0.735350424 -0.8296002331  1.146372163  1.023445485
## 204  0.861057932  0.764443386 -1.1641940183 -0.267192808  0.328315749
## 205  0.229872240  0.670568222 -0.3359807110  0.436146936  0.381538085
## 206  0.376681314 -0.478595982  1.2779913622  1.356233965  0.831779575
## 207 -0.027680108 -0.053596718  1.5158773985  0.408973106 -0.987481371
## 208  0.344059277 -0.621879735 -1.1647769722  0.377179694  0.584620131
## 209 -1.619214417 -0.610493237  0.8429658440  0.881443835 -1.122269694
## 210  0.034255348 -0.840356334 -1.4895575460 -0.444008778  0.653174156
## 211 -0.109492510 -2.021517281 -0.4197150080  1.706532107  0.349162892
## 212  0.616352852  0.560668280  1.7147419015 -0.779966490  0.067309375
## 213 -1.437403902 -0.806930415  0.9582001020 -0.596599498  0.002310168
## 214 -0.919519307 -0.402534860 -0.4219613186 -0.137817675  0.734175466
## 215  0.671008949  0.412281335  0.6386504021 -0.543883253  0.313593636
## 216 -0.155209221  0.344581014  0.3286809430  1.028118703 -1.313499541
## 217  0.824402139  1.731246884 -1.2450310015 -0.160552627 -0.596717742
##            PC31         PC32         PC33        PC34         PC35
## 1    0.46052294  1.141057294 -1.413077406  0.22824827  1.954120476
## 2    1.09887084 -0.157749081 -1.769058821 -0.08921589 -0.947107351
## 3    1.22368572 -0.398533605  0.754434648 -0.56577599  0.507389710
## 4    1.19088265 -0.536031943 -0.484421749  0.74190707  0.407436014
## 5    0.70290409  0.659242447 -0.359963761  0.51360722  1.076424779
## 6   -0.76642194  1.248986220 -1.114901892  0.68981362  0.201901983
## 7    0.99330254  0.354663078  0.485157189 -0.28009247 -0.254988859
## 8   -1.83330054  0.319475125  1.701555007 -0.08818242 -0.472129598
## 9    0.86665362  1.174109895  0.416388739  0.90137069  0.763630823
## 10  -0.27797520  0.558026229  0.668005607  0.25937684 -0.295585085
## 11   0.59125783  0.149672006  1.449676141  1.72994756  0.891550222
## 12   0.50280068 -0.293669830 -1.025063753  0.20472030 -0.648665316
## 13  -0.29095472  0.131529947  0.748290091 -0.24119147  0.689184024
## 14   0.19010430 -0.133642720  0.591499124 -0.36461029 -0.077307855
## 15  -0.16907699  0.182943493 -0.778595785 -0.89560666 -0.543514507
## 16   1.01035069  2.582509653  0.420253359  0.83712941  0.997677240
## 17   0.77150110  1.352920571 -0.303946662 -0.60154635  0.881395022
## 18   1.95659252  0.295334760  0.134431837  0.02896733  0.212005980
## 19  -0.17441285  0.100424851  0.752366750 -1.25875199  0.009768370
## 20   0.53500678  0.624379912 -1.028040549 -0.32244078 -0.369570630
## 21   0.18298128  0.333727646  0.846017504 -0.81473978 -0.235966533
## 22   0.62420417  1.224281763 -0.374035943 -1.65871951  0.553142223
## 23  -0.15281120  0.798228004 -0.205962691 -0.49336625 -0.188531675
## 24  -0.04008211 -1.057349636  0.969057307  0.68401646 -0.361244594
## 25  -1.20187850  1.746277502 -0.231280759 -0.85469077  0.091518828
## 26  -1.22603416  0.912872136 -0.940934638 -1.79698480 -0.669030697
## 27   0.99254732 -1.118667245  0.947076780  2.17593532  0.371239862
## 28   1.89739667 -2.228342062 -1.236917802  2.08679579 -1.026548006
## 29   0.64194010  0.027892022  0.246909911  1.43451809 -0.875706869
## 30  -1.87831845  0.962218263 -1.177732925 -0.49341182 -1.644551669
## 31   1.97084629 -0.826141428 -0.168365354 -0.05109862 -0.938969547
## 32  -1.37652346 -0.285297999 -0.444217228  2.30630737 -0.066955160
## 33  -1.02021384 -2.378520728 -0.420394958  0.84689798 -1.062171220
## 34  -0.40298193  0.669989282 -0.313487118  0.23290439 -0.330248773
## 35  -0.65751725  0.336118296  0.113680797  0.36901886 -0.574172937
## 36  -2.24203345  0.927196342 -1.167742512  1.24015336 -1.767197969
## 37  -0.53659977 -0.268356504  0.515108880 -0.78528139  0.770344096
## 38   1.02380738  0.049745275  2.167378136  0.07616325 -1.505035904
## 39   0.22369197 -0.385646914  1.796189555  0.59937371  0.631269731
## 40   0.50671998  1.305670291 -0.589338859 -0.80268624 -1.091738175
## 41  -1.18732270  1.219894412  0.179099442  0.48535437  0.187334322
## 42  -0.41515355 -0.240615488  0.543728715 -0.45361039 -0.173792284
## 43   1.04855953  0.229546792  0.166284959 -1.10388794  0.099880503
## 44  -0.33387722  1.731266374 -0.162142890 -0.03903488 -1.337014939
## 45  -0.06905288  0.596998301 -0.432550736 -0.50487647  0.173481812
## 46  -0.03586700  1.806397681  0.538617306  0.50268347 -2.228810598
## 47  -0.76068046 -0.146462961  0.569565703 -1.58952918  0.149755154
## 48   0.01858083  0.637542749  0.047842044 -0.99443217 -0.023492510
## 49  -0.09139212  0.384726578 -0.063669730  0.45311395  0.492414066
## 50   1.06514436  0.870942821  0.496972839  1.19729476  0.467042351
## 51  -0.05669547  0.459814213  0.316446847 -0.03559782 -0.006233306
## 52   0.26327404  0.543871117 -0.999511587  0.02965476 -0.019569727
## 53   0.28517336  0.393148045  0.193092068 -0.87295857  1.240674556
## 54  -1.57924060  1.897768196 -0.222004557  1.21666665  1.275780239
## 55  -0.43894549  1.195292241 -0.201683738  1.70159285  1.666759878
## 56   2.24622848 -1.610478391 -2.827904131 -0.60554128 -0.859791928
## 57  -0.79885993 -0.452604306  1.644127376 -0.97096698 -0.604291474
## 58   0.76366535 -0.663753260 -1.611408782 -0.72510086 -0.786422719
## 59  -0.50112976  0.064535731  0.008061205 -1.49424823  0.110233200
## 60  -0.63635719  0.852211482 -0.478477723 -0.45569387  0.449212000
## 61  -0.53693183  0.148734584  0.955623120  0.58441144  1.632568666
## 62  -2.02281141 -0.113584629 -0.225651806 -1.46425306  0.169248408
## 63  -1.37240000 -0.687522522 -1.269149642  0.39539761 -1.155896638
## 64  -0.91872789 -1.432714426 -1.425615003 -0.18735860 -1.347392310
## 65  -1.24701209  1.135652753 -0.764682441 -0.45529555 -0.977186759
## 66  -1.09805172  0.151940724  0.148441873 -0.50297863 -0.507203060
## 67  -1.20445991 -0.325076737 -0.817870547  0.47597517  0.267585802
## 68  -1.66746821  0.587615637 -0.367300152 -0.20753990 -0.250756722
## 69   1.38977454  1.884126163 -1.064576813 -0.53000293 -0.420598673
## 70   0.88334205 -1.961476890  1.717895145 -0.96235247  0.180126840
## 71   0.13054132 -1.330713613  0.682903267  1.30363997  1.071956599
## 72   0.98238845  2.071321956 -0.446030095 -0.02444594  0.064483790
## 73  -0.46573907 -0.608097434  2.191249575  1.13786016  0.111992170
## 74  -0.13172570 -0.592433892  0.132887148 -0.23652727  0.648996816
## 75  -1.02590217 -0.186374352 -0.395810125  0.23609379 -0.264522755
## 76  -0.27655824  1.085477429  0.634042991 -0.31431421 -0.739287874
## 77  -0.59303530  0.306758443  0.570073429 -0.25315663 -0.704908307
## 78   1.01863525 -0.608028278  0.716640429 -0.57322569  0.207635244
## 79  -0.41261233 -0.770920884 -0.399261307 -0.78355316  0.887029041
## 80  -1.09803048 -1.482726163 -0.241072518 -0.70197816  0.413971591
## 81   0.26999856 -0.687916769 -0.900377349 -0.61009107 -0.400648191
## 82  -0.60571117 -1.627972783 -0.865938374  1.24962112  0.214495469
## 83   0.13516376 -0.914913197  0.365210593 -0.49844856  1.448869087
## 84  -0.89846973 -0.011034916 -0.740527231  0.69449531 -1.294051190
## 85   0.40581739  1.080080284 -1.142874209  0.02469351  0.496812322
## 86  -0.62896244 -0.216710312  0.377374191  0.76844287 -0.178609406
## 87  -1.24486677 -2.139879779 -0.836034552 -1.08926936 -0.043193169
## 88  -0.18073396 -1.679463045  1.532546258 -0.73483174 -0.039707979
## 89   0.76128285  0.159912646 -0.241675507  0.34631108  0.198856116
## 90  -1.30215595 -0.976690125 -1.160726168  0.81390512  1.477624915
## 91   0.12283238 -0.770202804 -0.723553435 -0.05478988  1.180271574
## 92   0.94994346  1.501289364  0.926670942  0.52331760 -0.279099438
## 93  -0.52891105  0.707201027  0.309347805 -0.60917053 -1.242009139
## 94  -1.45229320  1.696898943  1.181865633  0.13542909  1.279570368
## 95  -1.27777088  1.414346441  1.303495172  0.43788321 -1.336312243
## 96   0.58010235  0.840777046 -1.323791907 -0.74751735  0.264792351
## 97   1.62432009  1.485858590 -2.566888821 -0.28971225 -0.134643161
## 98  -0.64334969  0.327956263 -0.208373161 -0.25826941  0.610089311
## 99   1.25747003  0.981762615  0.998698458 -1.01487717 -0.365570770
## 100 -0.47766200  1.315574218  0.025177449 -0.26935435  0.192878840
## 101 -0.18237527 -1.696536928  1.003319622 -0.40888137 -0.607964816
## 102  0.01371570 -0.032213181 -0.604882477 -0.64986401 -0.817041661
## 103  2.26278535 -0.773410697  0.871057044  0.81699868 -1.295372184
## 104 -0.24394681 -0.930808691  0.919447238 -0.55433980 -1.425489014
## 105 -0.25993012 -1.008886653 -0.040268501 -0.52132909  0.346214997
## 106  0.18880009  0.150302061  0.599025942 -0.29987066  0.373891729
## 107 -0.56472535 -0.568164725 -0.290554133  0.84325348  0.081286308
## 108  0.36015813  0.070153648 -0.223550447  1.07263347  0.688710334
## 109 -1.23485577  0.006960213 -0.105345069  0.42164676  0.359076203
## 110 -0.05235487  1.524040705  1.867546111  0.80465100  0.043860649
## 111  1.60468128 -0.004563622 -0.391128302  1.23772660 -0.080692668
## 112  1.61424690 -0.387972279  0.842347649  0.69243288 -0.426802675
## 113  0.32658587 -2.132637091  0.445692995 -0.74954122 -0.800216464
## 114  0.70268918 -0.417052703 -2.340140566 -1.18383433  0.713297473
## 115  1.02511703 -1.769647817  0.100341632 -0.42777125 -2.469928248
## 116  0.68961468 -0.079189664  1.376490268 -1.19141021 -0.147043690
## 117  0.81911937 -0.311286555  0.222811887 -0.25537703 -0.698655368
## 118 -0.75741123 -0.366941318  0.460353569  0.22189003  0.386297150
## 119  0.34263723 -0.702230088  1.121972375  0.54723227 -1.983069921
## 120 -0.91499939  0.061164003 -0.340114820 -1.45340822 -0.078965555
## 121 -1.03731702 -0.041814179  0.209922256  0.19200838 -0.549090664
## 122  0.14008411 -0.385471743 -0.170172475 -1.52152922 -0.332406469
## 123 -0.42200519 -0.676460855 -0.562725584 -0.46182297  0.757447221
## 124 -0.60104042 -0.846761525  1.756541331 -0.19560229 -1.257117355
## 125 -1.07519057 -0.934816779 -0.619078364  0.46459113  2.152964281
## 126  0.33889040  0.755655041  0.240265049 -0.65673980  0.132514197
## 127 -1.36924650 -0.915147117 -1.368582618  1.41680199  1.064989737
## 128  0.24081131 -2.142225697 -0.985333609  1.42741398  0.615731182
## 129 -0.61682560  1.508924923  0.208767211  0.53704038 -1.178467537
## 130  0.87758544 -0.965328193  0.827152263 -0.44669071  0.779729875
## 131 -0.04212154 -0.762322078  0.420764411 -1.42379454  0.257508197
## 132 -0.38505398  0.119916195 -1.050126054  0.20661634 -0.185086660
## 133 -0.23705789  0.775307222  0.900749928 -1.49602462 -0.823637640
## 134 -0.30287527  0.797022317  0.758023685  0.52279461  1.445148491
## 135 -0.51115668 -0.006255277  0.301622460  0.42027003  0.707668796
## 136  0.46586049  1.041229735  0.524464145 -0.12682739 -0.277980049
## 137  0.60365925  1.188830147 -0.817079132 -0.12128724  0.464060531
## 138  0.12272294 -1.075362428 -0.434922998  0.05954644 -0.855279916
## 139  0.69197411  0.631270379  0.134014027  0.64218993  0.472218897
## 140  0.69646804 -1.633497811 -1.092975574  2.00179388 -0.627313913
## 141 -0.56038253  0.682943806 -0.679916989  0.61244312  0.946845674
## 142  0.69852459 -0.659393088 -0.344446695  0.70361604 -0.635135172
## 143  0.30504314 -0.513611479 -0.748811100  0.46891336  0.179189214
## 144 -1.34570116 -0.570286064 -0.144816047  0.45839032 -1.207121818
## 145 -0.35376116  0.981044662 -1.429638964 -0.09368206  1.162920003
## 146 -0.11261822 -0.453151044 -0.484956813 -0.51907735 -0.669330938
## 147 -0.42112701  0.440120132 -0.104086741 -0.39268716  0.415480108
## 148 -0.21032930 -0.131775922  0.312188143  0.42885606  0.097908051
## 149  1.51720717  0.493736607  0.702808567 -1.11003369 -0.326685298
## 150 -0.01410601 -1.037348851  0.245662112 -0.06590432 -0.489767571
## 151  0.39529587 -2.506452534 -3.109045615  0.84119563 -0.251835607
## 152  0.77908578 -0.687135759  0.871936760  0.04818327  1.840405620
## 153  0.70938786 -1.556520281  0.667165213 -1.25957493  1.009708913
## 154 -2.45428755  0.510735456 -0.323038562  1.59482061 -0.773361508
## 155  0.47410948 -0.062205979 -0.273393229  0.86917047  0.328044838
## 156  0.19129488 -1.464868858  0.260686719 -0.80149126  0.466458463
## 157 -1.01091884 -1.591349495  0.683246859  1.61973681 -0.409184708
## 158 -0.52698982 -0.350416249 -0.551808441  0.42779373  0.881805078
## 159 -0.92976473 -0.205955285 -0.512511827  0.55283852 -0.478373059
## 160  0.92957719 -0.908349592 -0.721817841 -0.14673467  0.147955918
## 161  0.47947418  0.304151878 -0.150087210 -0.78493485 -1.128374573
## 162 -1.00906865  0.253938672  0.396941651  0.72180155  0.233642500
## 163  0.73698856 -0.445142214 -0.266251910  0.54798750  0.835797011
## 164  1.70608765 -0.059111564  0.835447400 -0.64954582 -0.525076808
## 165  1.35394008  1.071962271  0.555958650  0.05439842  0.563643381
## 166  0.83874193 -0.734872784 -1.583153875 -2.19207110 -0.294162064
## 167  0.60676456 -1.133299210  1.071997870  0.73507576  1.248604698
## 168  1.16436025 -0.438620039 -2.499561714 -0.92244094 -0.299574757
## 169 -1.93564239 -1.382411873 -0.349300843  1.38997022 -0.218926068
## 170 -0.37277473 -1.086487414 -0.465634826 -1.48166176  2.180636320
## 171  0.51470477  1.137837107 -0.783101750 -0.35249065  1.061473416
## 172  1.56284732 -0.325073355  0.109783721  0.46368444 -0.362265665
## 173  0.45928206  0.064835202  0.424008540 -0.48866380  0.038054952
## 174  1.05672699 -0.273948243 -0.681890615  0.55775302  0.074778818
## 175 -0.08598027 -3.216411402  1.477938256 -2.67350508  1.500761250
## 176 -0.36727100  0.249174061  1.110361775  0.71234133  2.728848073
## 177 -0.06154212  0.091141272  0.419201186 -1.89291317  1.273669345
## 178  2.32543416 -0.486717114  1.224353045 -0.41402031  0.387882281
## 179 -1.10551301 -0.133916159  0.815060582  0.11325315  0.962688633
## 180 -0.47528232 -0.101413569 -0.870734645 -0.55293436 -0.016853990
## 181 -0.79475697  1.399045796  0.709137202 -0.32815541 -0.009720862
## 182 -1.54626329 -1.077650510  2.115584279 -1.05899364  1.591139695
## 183  0.34253801  1.547567924 -0.271554156 -0.21341934 -0.620302344
## 184 -0.61068011 -0.790107139  0.311593790 -0.56900183  2.085195170
## 185 -1.41385852 -1.425717736  0.414830355  1.47763647 -1.953708979
## 186 -0.13719425  0.749500051 -1.013886174  0.13333705  0.236407617
## 187 -1.18947151  0.591482471  0.862176674 -1.54316959 -1.418768015
## 188 -1.65239635 -0.631681162  1.081806934  0.54132354 -0.699460532
## 189 -0.63193015  1.176723590 -0.041366136  1.07316539 -1.888374502
## 190 -1.27964098 -0.741772366 -0.618995217  1.34357598  1.550019456
## 191  0.42173637 -0.251510022 -0.862336577 -0.14529113 -0.168243703
## 192 -1.06707875 -1.022397082 -0.805901939 -0.35792432 -0.550806438
## 193  0.72980442  0.501320150  0.316160488  0.51857068  0.672123459
## 194  1.11622815  0.134209122  0.750522971 -0.74048178 -0.307087318
## 195  0.12274866  0.644470888 -0.429335954  0.42947020  0.431504439
## 196  0.66166336  0.407208967 -0.447809328 -0.27421466 -0.425865464
## 197  2.09958007  0.932870781 -0.711761163  0.59741008 -0.039300881
## 198 -0.27905992  0.323902464 -1.033361709 -0.47238084  0.710877520
## 199 -0.88501655  1.062078045  0.539880650  0.45236210  1.873351719
## 200  0.83231280  1.078768465 -0.708363313  0.45828843 -0.010913638
## 201  0.11475396  1.340583444 -0.592737914 -0.26824044  0.379876123
## 202  0.50642244 -0.273480497 -0.948580874  3.53809334  0.597028417
## 203 -0.70529887  1.570316820  0.974317621  1.24792009 -0.869630981
## 204  0.84151710 -0.228112388  0.919081345 -0.63008088 -0.153120707
## 205 -1.75406117 -0.138387201  2.457821661  1.66183132 -1.321079903
## 206  0.70101851  0.829377743  0.524161802  0.31655439  0.984845975
## 207  2.08088978  0.443042888 -0.193511562  1.34591163  0.154126225
## 208 -0.75881400  0.219700405 -1.157955189 -0.95830073 -0.270111661
## 209  2.41031407  0.097942669  1.171683177  1.02630746  0.909596297
## 210 -1.74653294  0.074309063 -0.069729051 -1.09873478 -0.067597144
## 211 -1.98193460  1.070284024 -0.785126716 -1.20984613 -1.140669667
## 212  3.14485393  0.202360163  1.456772608  1.03477129 -2.520170724
## 213  0.33334768 -1.334269131  0.851929831 -1.56007273 -0.801091060
## 214 -1.00295735 -0.119435600  0.092806740 -1.44818287 -0.214282870
## 215  0.05659396 -0.181580431  0.990463810  1.52300032 -1.034078439
## 216  1.89024188  2.177967407  0.601096639 -0.52056520 -0.344790954
## 217  0.36410013 -0.818495185 -0.754811921 -0.07774055  0.572451345
##             PC36         PC37        PC38         PC39          PC40
## 1   -0.543233480 -0.098227582  0.11111783  1.267847503  1.0098119752
## 2   -0.672121766  0.540038975 -1.01468275 -0.072874241 -0.8857081900
## 3    0.826109670  2.079026304  0.84424809  0.835371780 -0.2683133195
## 4    0.971739383  0.190995594 -0.35770923 -0.180805240  1.6457667186
## 5   -0.239301583  1.891112730  0.56309314 -0.543486447 -0.6259152997
## 6   -0.379080806 -0.607674996 -0.13711586  1.254985387 -0.8397625302
## 7    0.003811184 -0.742829733  0.11901226 -0.019789958  0.5746592584
## 8   -0.361834279 -0.419755348 -0.29643809  0.119210275 -0.0132576294
## 9    0.408079281 -0.212963915 -0.02996165 -0.079679246  0.0964654955
## 10   0.457061262  0.809161048 -0.02181607  0.749186295 -0.3234094094
## 11   1.369400509  0.691078639 -0.32920718  0.956550333  0.3534146299
## 12   1.315747536  1.054946841 -0.32848801  1.179379408 -1.2941116352
## 13   0.824767286 -0.440346618 -1.35987968 -0.114768705  0.4087562873
## 14  -0.354951933 -1.062745655 -0.54485622  0.035454847  0.0841854393
## 15   1.031895505  0.875947139 -0.63360301 -1.989184729 -0.3482438307
## 16   0.793672005 -0.167499972  2.27081856 -0.422310623 -1.4079384530
## 17  -0.282447360  1.143955635  0.86874988 -0.343753170  0.2773043030
## 18   0.279906945  0.408723918  1.18732091  1.078466804 -0.0352095662
## 19  -0.451928985  0.570876394  1.16114611  0.720414970  1.3468473555
## 20   0.849082655 -0.707084685  0.59994957  1.087588610 -0.3728740291
## 21  -0.422426219 -0.510166403  0.02995534  1.062741706  0.9898293507
## 22  -0.238341046 -1.425210335  0.77699130  1.112016730 -0.8101135454
## 23   0.430315354 -0.462988723  0.03844495 -0.933265280 -1.2604407474
## 24   0.835961838 -0.369887063  0.14393985  0.471262522  0.1316456373
## 25   0.440416411  1.046420954  0.05901343 -0.392407637  0.2419867214
## 26   0.886912350  0.716256575  0.87178383 -1.575985121 -1.1248989589
## 27  -0.040192071  1.168668266 -1.43028196 -0.057510442 -0.2323049192
## 28  -2.333904885  1.095391771 -0.10979924 -0.267867904  0.0952516063
## 29   0.164236260 -1.298101762  1.48362067  0.362053364  0.3856164940
## 30  -0.048121480  0.544524855  0.69768604  0.614605432 -0.6456451351
## 31  -0.333557404  0.438870980 -0.97301301  0.423772417 -0.6027439396
## 32   0.551829811 -2.017035374 -1.37154283  0.764169375  0.3577970023
## 33   0.150873128 -2.042244596 -0.03859803  0.020961978 -2.3010493152
## 34  -0.399522303 -0.138857034 -0.87917973  0.240279746 -0.0304105144
## 35   0.336773738  0.105335376  0.10814027 -0.204733763 -0.7655031050
## 36  -2.164823809  1.737680935  2.74368012  1.838341927 -0.8209056512
## 37   0.228529804  0.625915440  1.08254051 -0.509751875  0.3827705204
## 38  -0.369257037  0.541945082 -0.27461237  0.994938034  0.4293846331
## 39   0.607401052  0.075161587 -1.73477414  0.459350451 -0.6914484940
## 40  -0.675758272  0.608383917  0.25193837  0.435473255 -0.0027412367
## 41   0.359377651  0.027028571  0.85162432  1.043152514 -0.8592287145
## 42   0.412732376 -1.185820222  0.05010738  0.142683338  0.8042137955
## 43  -1.015738955 -0.545554261  0.48431698 -0.058080423  0.3013131605
## 44  -0.251948570  0.194873432 -0.32160777  0.435960470  0.1304791465
## 45  -0.206794944 -1.020602496  0.38585565  0.169791815  0.5241037429
## 46   0.112580650 -0.791137030 -1.12712640  0.010651256 -0.4940500381
## 47   1.390321842 -1.167796700 -0.06573142 -0.976509934  1.4410618046
## 48   0.543417085 -0.186488686  1.02134651  0.343521953  0.0271819280
## 49   0.345429293 -0.548280376 -0.15406986 -0.506770184 -0.6404282522
## 50   0.348664169  0.279202815 -0.18130277 -0.083815578 -0.4655481553
## 51  -0.763325100  0.706374497 -0.20336269  0.006396112  0.0165846780
## 52   0.197702525  0.990180426 -0.83851017 -1.158888040 -1.6126165238
## 53   0.495701107  0.304618385 -0.40344696 -0.056183664 -0.0312388462
## 54  -0.196760118  0.690048257 -1.35033344  0.441475457 -0.9087192472
## 55   1.225475778 -0.251843172 -0.33990074 -0.373385424  1.1002706748
## 56   2.141437671  0.795360335 -0.51245412 -1.738859165 -0.0177825052
## 57  -1.196834550  1.116292206  0.20273804 -0.914322358 -0.8621305378
## 58  -0.981366738 -0.850781359  0.36715594 -0.271159694 -0.2353363802
## 59   0.001339874  0.144021557 -0.37845335 -0.647906302  0.2654060697
## 60  -1.420589431 -0.666307616 -0.90744904 -0.287000562  0.2700887948
## 61  -1.165252307  0.631153706 -0.13913303 -0.530967784 -0.2575476271
## 62  -0.779503720  0.128454624 -1.65930675  0.698021904 -1.1874933886
## 63   0.254245571  0.900492152 -0.33816165  0.840792038  0.5568962793
## 64   0.860349569  0.249858423 -0.20660995 -0.512482715 -0.3468567560
## 65   0.786435655  1.285527540 -0.41284314  0.452970616  1.7677859492
## 66  -0.443171847 -0.786882169  1.05020926 -1.083224561 -0.7742726692
## 67   0.462881419  0.165054589 -0.94050502 -1.662909137  0.6130479074
## 68  -1.176211347  0.223778544 -1.24760746  0.487373096  0.0007595562
## 69   0.756389042 -0.071171017 -0.50247149 -1.011472526  0.7717909784
## 70   0.604122769 -1.055443961  0.52525075  0.701128201 -0.7444358507
## 71   1.690458347 -0.946893203  0.23800769 -1.066351627  0.5528276752
## 72   1.322961620 -0.842215120 -0.04138624 -2.214469768  0.8711587375
## 73   0.300967859 -1.027754005  0.18089216 -0.654265637 -0.3502483170
## 74   0.337914803 -0.292882299  0.18484121  1.020507469  0.0474513015
## 75   0.499521759 -0.149937933 -0.44065384  1.365533816  0.5039819833
## 76  -0.468578337 -0.092122135  0.08820931  0.211353269 -0.3337850099
## 77   0.278025095 -0.236547535 -0.16505808  1.678978623 -0.8143515112
## 78  -0.251402008  0.180311557  1.05831949 -0.228367898  0.6063475966
## 79   0.445923851 -1.092445124  1.39365686  0.867140715 -0.2711943273
## 80   0.292649410 -0.688285878  1.04031176 -0.188582731 -0.4080125326
## 81   0.197108870 -0.511381970 -0.18324886 -0.051601283  0.0915612338
## 82   0.184489454 -0.327646279 -0.13565678  0.245838266 -0.6051980044
## 83  -0.087535689 -0.821151782 -0.35645084  0.269264116 -0.7820085372
## 84   0.231285668 -0.488284530 -1.14401156 -0.832254338 -1.8082537371
## 85  -1.360182808 -1.716615476  0.21650337 -0.226409095  0.1878566427
## 86   0.578141970 -0.795714991  0.77294931 -0.432096369  1.1365421988
## 87   0.546855640 -1.586436749  0.43425877  0.260532320  0.1109874416
## 88  -0.981984488 -1.025728889 -3.07370352 -0.529248565  1.2798706511
## 89  -0.365533667 -1.125678553 -1.52639888  0.117501648 -0.3970370306
## 90   1.790728522 -0.495230830 -1.03308465  1.066436593 -1.0988059906
## 91  -0.733760090  2.404043909 -0.85799467  1.078411800  0.7007126454
## 92   0.891206008  0.446238324 -0.09180424 -0.131051463 -0.0753019272
## 93   0.001414879 -0.051186563 -0.25756601 -0.192482751  1.0455505364
## 94  -0.348905271 -1.649974834  0.88698569 -0.586113192  0.1177991028
## 95   0.170446706  0.031666194  1.17337562 -0.790873507 -1.0007924771
## 96  -0.152722989  0.978554163 -0.49096476  0.798015881  0.4689096226
## 97   0.009911138 -0.772441297 -1.17831984  1.452968690  0.7339830690
## 98  -0.526787991 -0.036699229 -0.86967315 -0.768493025 -0.5558552176
## 99  -0.436931385  0.744928116  0.80404538 -0.262543926  1.1007131724
## 100 -0.523929775 -0.009236245 -0.37723025  0.100946949 -0.7612583021
## 101  1.295695219 -0.431143932 -0.23209033  0.566665286  1.1879124202
## 102 -0.847717411  0.384226524 -1.20397158 -0.603962364 -0.7006162137
## 103  0.076074277 -1.166997135  0.57153889 -0.302169701 -0.0448388357
## 104 -0.467694535  0.630699834  0.78162338 -0.092975259 -1.1643373106
## 105  1.403626043  0.522639616 -0.06588565  1.093796642  0.3789532421
## 106 -0.984463461  0.646292280 -0.60998467  0.763563540  0.6215892086
## 107 -1.539551723 -0.886020538  0.96050356  0.475292779  0.1491804599
## 108  0.281844944 -0.743831832  0.45976178  0.290082687  0.3323686410
## 109 -0.869929581  0.244220316  0.50033669  1.649871977 -0.3649286742
## 110 -1.403032175  0.537034219 -0.53866104 -0.623227729  0.6879311683
## 111 -0.525114098 -2.457558576 -1.72598213  1.195227495  0.3252369901
## 112 -1.024394828 -1.076379416 -0.61043430  0.146841108  0.0925068668
## 113  0.148694221 -0.469085668  0.48409152 -0.373853707 -0.8190229428
## 114  0.638561059 -0.862563550  1.15763643  1.231290722 -0.2827748850
## 115 -0.191893563 -0.473692283 -1.45371134  1.555689882 -0.6239749684
## 116  0.158420241 -0.380800274  0.59106889 -0.447459334  0.2144462412
## 117  0.897236116 -0.225148851  0.19001257  1.458993087  0.8956833375
## 118  1.111377467  1.877061146  0.10957780  1.122317955  0.1739352381
## 119  1.871603788  1.189803844 -0.06648984  0.594429135 -0.1682791332
## 120 -0.933304842  0.538977465 -0.40497715  0.349201809  0.4289479897
## 121  0.881765528  1.207413465 -0.54164279  0.148413958  0.4480988126
## 122  0.889070799  1.248414720  0.36963186  0.341365772 -0.0773377654
## 123  0.908242013  0.388904417  0.26674885  0.614469370  0.1260250271
## 124 -0.323456925 -0.482018442 -0.44498746 -0.260579862  0.9338225290
## 125  1.378894277 -0.309106732  0.15963135  1.007907788 -0.4618686999
## 126 -0.419579239 -0.560345515 -0.21315643 -0.198614305  0.2072512856
## 127 -0.307594866 -0.609827982  1.01443040  0.379652676  0.0880574143
## 128 -2.840658300  0.156672002  0.65949177 -1.431001976  1.4253598747
## 129 -0.240183362 -0.435184339 -1.75093277 -0.501221282  0.3569500199
## 130  0.449903363  0.549656131  0.28296220 -1.055805932  1.5140021541
## 131 -0.113163889 -0.442601983 -0.27232401 -0.438013649  1.1982221127
## 132 -0.649570432  1.203596281 -0.79602820 -0.556192960 -0.9532918857
## 133 -0.402861524  1.530142874 -0.27913706 -0.323031772  0.7545203251
## 134  0.086250535  1.430191579  0.73761400 -0.642398236  0.1004269891
## 135  1.065292040 -0.151588615 -0.12349906  0.106752666  0.9551630764
## 136 -1.218806451 -0.344616866 -0.34952616 -0.564622737  0.0890146734
## 137  0.191290231 -0.235596799 -1.21142808  0.608713203 -0.0599183400
## 138  0.002157692 -1.259645247  1.23419011  0.179227703  0.5905692063
## 139 -0.195762581 -0.027104178  0.36704159  0.389385296  0.4155319938
## 140 -2.269568609  2.092370135 -0.05624956 -0.186564638  0.1782427790
## 141  0.395380056 -0.201899524  0.19323947 -0.056755556  0.1293803994
## 142  0.545470261  0.143415401 -0.92857365  0.661345873  0.3934529467
## 143  0.112316652 -0.311693769  0.12306966  1.180629476  0.6719307495
## 144  1.569412095 -0.277320741 -0.49290111 -0.129693114  0.8208443825
## 145  0.236616119 -0.743126033  0.04749553 -1.016309261 -0.1190444381
## 146  0.878767140  0.373869410  0.74150941 -0.129858360  0.7582011437
## 147  0.552678043 -0.705644942 -0.18752845 -1.952650278  0.1954780270
## 148 -0.542560850 -1.549891147 -1.12113364 -1.217723589 -0.7394292998
## 149  0.379172612  0.596463855  0.85704074 -0.438927604 -0.8171249489
## 150  0.274157306  0.041251250  0.08092779 -0.567838507 -0.4467181851
## 151 -1.735009629  1.075100176  0.94818209 -1.970710065  0.6239207618
## 152 -0.452534449  0.507223192 -0.13008988  2.287118754 -0.1804601045
## 153  0.168729512  1.153358603 -1.48122578 -1.349563917 -0.9510869986
## 154 -0.400110490  0.560575743 -0.03962072  0.591355172  1.4295886558
## 155  0.895549042  0.007059020  0.35740489 -2.063534672 -0.5049845617
## 156  0.826731884  0.883838453 -1.63838216 -0.431395986 -1.0771845455
## 157 -0.988862921 -1.711435437  1.66934957  0.436311209  0.9139802031
## 158  0.386400605 -0.618544261  0.49891869 -0.040774082 -0.4231455677
## 159 -0.019726676  0.007577703  0.35498726 -0.791323903  0.5715119724
## 160 -0.493481688 -0.880085484  0.11599136  0.611627077 -0.5004500220
## 161 -1.110017618  1.705345153 -0.84924989 -0.497128165  0.6939273631
## 162 -1.021964885  0.362467906 -1.91693016  0.107700194 -1.3313765210
## 163  0.746473310 -0.594989169  0.34163888  0.235518512 -2.2151479895
## 164 -0.160607720 -0.032111297  0.33602755  0.663093042 -1.0020251125
## 165 -0.824348704 -0.339337327  0.87040619  0.084971748 -1.2499516965
## 166  0.583748573 -0.125875804  1.36258728  0.424164159  0.0804548625
## 167  0.625245612 -0.060380748  0.95623819  0.086462492  0.0057304551
## 168 -0.163517292  0.749611243 -0.74230512  0.721381157 -0.4532132214
## 169  0.224978601  1.110222360  1.26616552 -0.310056916 -0.1536812992
## 170 -0.256444557  0.042181901 -0.63655594 -1.464796387 -0.6036664035
## 171  0.343186478 -0.272213789  0.09093995 -1.145738076  1.0918818580
## 172  0.752237448 -0.970788322 -0.69736877 -0.181496933  0.0148587698
## 173  0.498200450  0.565609350  1.06814267 -0.243359554  2.1722096137
## 174 -1.371448397  0.310496296  0.09758730 -1.143438455  0.6492118365
## 175 -2.959370223 -0.396244686  0.39938298  0.174920786  1.3639059722
## 176 -0.473875214  1.276601588 -0.56953344  0.415987999  0.0060289088
## 177 -0.864834877  0.079752748 -1.00504558 -1.127697432  0.0616371002
## 178  0.298032882  0.063836234 -0.11363999  0.120154808  0.7982959656
## 179 -0.661117955  0.780635027 -0.97582253 -0.316805414  0.4839856379
## 180 -0.128918368 -1.382925727 -0.05953708  0.425751409 -0.8934604368
## 181 -0.961281016 -1.178468391  0.29652483 -1.303155139 -0.3569721916
## 182 -0.408910144  0.152050668 -0.13904780 -0.262294593 -1.6077458502
## 183  0.509519519 -0.154967845  0.72765103 -0.446108472  0.1667992204
## 184  0.942892420  0.778353283 -0.04832392  0.043900700 -0.6887248884
## 185  0.751057402  1.098202217  0.12070325 -0.288247935 -0.2181097299
## 186 -0.901353940 -0.552640304  0.19186999  0.483316846 -1.4944204871
## 187 -2.547620631 -0.732176581 -0.08123474 -0.957250793 -1.5341916225
## 188 -0.074304225  0.017132326  0.68092226  0.281482298  0.4078752772
## 189  2.903878703  0.067142346 -0.72520707 -0.321474609  1.4030553071
## 190  0.599869430  1.136390184 -0.18839202 -0.802693249  0.6911883099
## 191  0.530905503 -0.447080966  1.43103320 -0.639886624 -0.7365513367
## 192  0.333714824 -1.314659112 -0.57233356 -0.887124258 -0.8761157954
## 193  1.137583614 -0.558921794  0.26563877 -0.517172328  0.0902328675
## 194 -0.026207532 -0.105116554 -0.82555678  1.109869838 -0.3415809220
## 195 -0.465565063 -0.217972470  2.10920578 -0.774928288  0.4256954829
## 196  0.472776305 -0.306101566  0.03635012  0.458013259  0.2448823031
## 197  0.542335194 -0.495606242  0.57883943 -1.519388050 -0.3100993683
## 198  0.438093358  1.596778071 -1.37689569  0.473450006  2.0991429053
## 199 -0.949236437  0.991714914 -0.25471820  1.034591016 -0.5515673935
## 200 -0.110507278  0.137789605  0.68298441  0.097084045 -0.1521812625
## 201 -1.061873530 -1.532223571  0.60489289  1.573271713  1.4063184928
## 202 -2.497951839  1.039700190  0.52698441 -1.514628045  0.8179715996
## 203  1.236276739 -0.517369443 -0.88396434 -1.454004267 -0.0005119518
## 204 -0.486402982 -0.080858785  0.30763952 -0.312074696 -0.5810312787
## 205  0.466957316  0.458788209  1.69270429 -0.766081616  0.1509276242
## 206 -0.419283286 -0.412717344 -0.02545526  0.263678876 -0.1174796458
## 207 -0.450759686  0.452511387  1.30229999 -0.240891042 -1.0845063145
## 208 -1.180121484 -1.754095011 -0.23417790 -0.275970543  1.5265746865
## 209  0.059770711  0.911538780  0.18723271  0.762487006 -1.6897427220
## 210 -1.125421349 -1.141915746  0.73105452  0.473855165  1.0482549828
## 211  0.013806350  0.922294034  0.54530145  0.611454478  1.0906946168
## 212 -1.027546175  0.081922606 -0.67955844  0.428878105  0.9094388365
## 213  0.869856609  1.950882948  1.94127556 -0.958578472 -0.8538155797
## 214  0.557774792  1.767181615  0.01463991  1.215691338 -0.2270663938
## 215  0.217107545  0.168887279  0.65799530  0.327702144 -0.2306901059
## 216 -0.715844956  0.207306250 -0.19532697  0.405671041 -1.0404142758
## 217  1.507003820  0.052621338  0.35286592  0.234725942  0.2414463125
##             PC41         PC42         PC43         PC44         PC45
## 1    1.481382690 -2.181902004  0.170544432  0.664239981  0.355634048
## 2   -0.675238247  0.612903426 -0.321132460 -0.756552676  0.400386372
## 3   -0.613821476 -0.276899577  1.108094055  1.513810052  0.038052992
## 4    0.138036685  1.425904704 -0.780932684 -0.674596885 -0.314813983
## 5   -0.538859859  0.229686187 -0.952697386 -0.742940621  0.816568235
## 6   -0.235180580 -0.176870840  0.558655829 -0.222308680 -0.190268761
## 7   -1.022484817  0.492032871 -0.191849782  0.407677853  0.893328061
## 8    0.030672078 -1.983155647  0.090992036  1.032412507 -1.900117801
## 9    1.000362383 -0.376745016  0.527396035  0.236627383  0.268892521
## 10   0.666706174  0.968601835 -0.453027595  0.184867271 -0.126385076
## 11   0.216903002 -0.759254352  0.206509673 -0.069410331 -0.020084716
## 12   0.963962090  0.051664466  0.169256285 -0.348499158 -0.018199981
## 13   0.606668344  0.072110857  0.403305040  0.009444046 -1.317771469
## 14   0.925823409  0.558685660 -1.278986927  0.214980926 -0.276813154
## 15  -0.003618511 -0.377327791 -1.809621866 -2.866176778 -1.345817334
## 16   0.909298530 -0.222655918 -0.927791795  1.342770690 -0.343714102
## 17   0.764725320  0.448049285  0.728958977 -0.401280621  0.391835299
## 18  -0.656747533 -1.002870523  1.674806294  0.181534014  0.084748366
## 19  -0.255999055 -0.087933733 -1.205051281  0.160648252 -0.505404678
## 20  -0.470205509  0.889902187 -1.136522634 -0.764447678 -1.125894370
## 21   0.382332838 -0.568363492 -0.551823486 -0.142066874  0.106335695
## 22  -0.056489772 -1.001473273 -0.884339314  0.079592894 -1.186711389
## 23   0.434524182 -0.516497738  1.288399548  0.394841865 -1.443382816
## 24  -0.355299870  0.256842129  0.014080051  0.260422200 -0.243759154
## 25  -0.155754374 -1.084151947 -0.063218205  0.406514237  0.262035542
## 26  -0.219953721 -0.828054125  1.004246098  0.912157480 -1.192501966
## 27   0.823286060 -1.850894993  0.724677203 -1.861248429  0.083774865
## 28   0.867853535  0.372465303 -0.568389532  0.293022225 -0.553113268
## 29   1.158000476  1.612374975 -0.095571804  0.856525861 -0.052415307
## 30   0.511239421  0.228178214 -0.671823941  1.128694668  1.967102894
## 31  -0.182516966  0.538930707  0.537203921 -1.394988509  0.294026292
## 32  -0.919582393 -0.979815940  0.103764001  0.258006815  0.691918304
## 33   0.087545991 -0.277247078  1.537096479  2.287354839 -0.436287377
## 34   0.186699532  0.831008137  0.182795032 -0.839236438 -0.894665058
## 35  -0.361399330  1.075347547  0.193113147  0.623812384 -0.702761597
## 36  -2.301417920 -1.200455759 -1.284389668 -1.647637471  0.109958373
## 37   0.909096059  0.812940686  0.009537141  0.333677946 -0.716093788
## 38   1.313471599 -1.631921530  0.749459370 -0.186904863  0.530420627
## 39   0.580439114  0.330978060  0.359115287 -0.359093447 -0.874474695
## 40  -0.190684575  0.642954503  0.087308749  1.120750019 -0.017071969
## 41   0.247572075  1.330080375  0.663016532  0.587387351  0.192162606
## 42  -1.338693581 -0.301075978  0.284299985  0.363247770  1.427529905
## 43   1.049976499  0.955020733 -0.762840539 -0.177985859 -0.348117590
## 44  -0.855438584  0.895918541  0.263378553  0.921605065 -0.821129622
## 45  -0.605578719  0.628367545 -0.135524299 -0.011364265  1.278375602
## 46  -0.748719678 -0.608031496  0.976466725 -0.583139248 -0.162623132
## 47   2.116637851  1.009378562  1.475302093 -0.046407121 -0.432405599
## 48  -0.583459362  0.214928059 -0.396218699 -0.119502989  1.132658923
## 49   0.026272957  0.472632731  0.726531495 -0.022432448 -0.652873472
## 50  -0.194756153  0.597116842  0.781753658  0.987017902 -0.624009779
## 51  -0.003932607  0.161975381  0.132443461 -0.152288213  0.398556710
## 52  -0.922059088 -0.604556105 -0.551561455 -1.392143168 -0.064982036
## 53  -0.996845209  0.641785716  0.657264606  1.010468140 -0.093926169
## 54   0.194238764 -0.997926576 -0.305593104 -0.078798810 -0.218143242
## 55   0.171831387 -0.419438180  0.075240484 -1.134543143 -0.637079987
## 56   1.312728440 -0.402657252 -0.828191112  0.240619354  1.995396421
## 57   0.068643731  0.042944111 -0.619881987 -0.040650323  0.212006233
## 58  -0.702759698 -0.422802291 -1.318708105 -1.318337414  0.918094378
## 59   0.170368395  1.250514835  0.772437066  0.316585204 -0.248627213
## 60   0.662348818  0.274805829  0.181027676 -0.544390517  0.184515055
## 61  -0.187295822 -1.698145494 -1.004082664  0.732155952 -1.092897390
## 62  -1.186089257 -0.296335440  0.153049381 -0.075049602 -1.348215499
## 63   0.761243730  0.342369502  1.271557507 -0.975138476  0.909221518
## 64  -0.467341250 -0.216710949  1.520337784  0.812250400 -0.005563716
## 65   0.229186541 -1.041416336 -0.367175778  0.373255605  1.268672290
## 66   0.913217994  0.365564472 -0.684299620 -1.595339790  0.051932913
## 67   0.755284711 -0.455146316  1.239695883  1.366690562  1.237184189
## 68  -0.282745544  0.613896756  0.240142116 -0.963213792  0.569397422
## 69  -1.368994907  0.733778803  1.329004734 -0.385450590 -0.790457804
## 70   0.771506311  0.876339218 -0.809855090  0.782656073 -0.590802418
## 71  -0.570006771  1.162566218 -0.404645155 -0.506004731  0.598344971
## 72  -2.357410922  0.057862642  1.218963631 -0.316512871  0.121431012
## 73  -0.662242719  0.591438885 -0.041988378 -0.986434094 -0.196460470
## 74   0.494472305  0.502200831 -0.213660668 -0.895945325 -0.566244114
## 75   0.320512409  0.696841748  0.033368112 -0.385182996 -0.272434262
## 76   0.395734370  0.471326979  0.446890045  0.099925145  0.460462476
## 77   0.090994671  0.514568917  0.361515058 -0.738847567 -0.207770926
## 78   0.655120004  0.765789431 -0.598455876  0.923031553 -0.036668985
## 79  -0.203222018  1.085700886  0.076909539  0.141778115  1.535038389
## 80   1.379465983  0.837728446  0.913779973  0.405442066  1.597916329
## 81  -0.265403232  0.216584562  0.159792725 -0.488561240  0.278498336
## 82  -0.250123434  0.266978728 -0.075760621  1.024904260 -0.109193417
## 83  -0.071559861 -0.434857273  0.642969672 -0.856084342  1.390362929
## 84  -0.056874542  1.005376803 -1.334580446 -0.123311157  0.163697452
## 85   1.354779250 -0.649035977  0.341535347 -0.888156100  0.734927887
## 86  -0.034805004 -0.611077819  0.169190908  0.939660326 -0.567328474
## 87   0.144205547 -0.896428122 -0.010269119  0.280445649  0.319234130
## 88   0.950709645  1.490955085 -1.843472527  0.209758009  0.767100939
## 89  -1.518153149 -0.258887142 -0.466284128 -0.584693962  0.603640602
## 90   0.185870626 -0.829438261  0.573880247 -0.064471257 -0.513270811
## 91   1.195789853 -1.436013930 -0.469914910  0.391859591 -0.372705114
## 92  -1.155152591  0.914558310  0.983225429  0.063045731 -0.414136155
## 93   1.090591504 -1.059769165  0.242839841 -0.050281550 -0.554662472
## 94   0.647023266 -0.405414671  0.063600053 -1.924959348 -1.158962838
## 95   0.686135224  0.164817443 -0.177615841 -0.023518032 -0.225119270
## 96   0.939824098  0.131097435 -0.201497642  0.468434036  0.458540516
## 97  -0.527334542  0.762106419 -0.527209151 -0.556865987 -0.303584702
## 98   0.045281778  0.797129172 -0.114479159 -0.736762762 -0.415875038
## 99  -0.978232978 -0.255324506  0.114945076  0.234056561  0.620011355
## 100  1.068862141  0.192605053 -0.053584366  0.080980264 -0.662961747
## 101 -1.526274872  0.517678188 -0.308387756 -0.241649424 -1.242171183
## 102  1.327166684  0.332317563  1.414255484 -0.644396697 -0.494428317
## 103 -0.295525599 -0.524882291 -1.569130798 -0.584530268  0.093649276
## 104  0.094443993  0.029920792 -0.149155412 -0.378981389 -0.325521796
## 105 -0.931680170 -0.016830235  0.160862242  0.380307408  0.212610631
## 106  0.090131784 -0.684291584  0.399779798 -0.428256739 -0.781944771
## 107 -0.641042473 -0.993691576 -0.014190558  0.485369460  0.326152753
## 108 -0.492896919 -0.411386040  0.565629629  0.156463516  0.294342093
## 109 -0.409701773 -0.671023997 -0.667758468  0.233534203  0.605263206
## 110 -0.391834082 -0.141326056 -0.582373130  0.037939751  0.856480152
## 111 -0.272632171  0.115835701  0.379880186  0.869019492  0.218272588
## 112  0.203040361  1.114469726  0.065661335  1.458410887  0.167599339
## 113 -0.354282091 -0.581174033 -1.327637738  0.170955717 -1.048223231
## 114 -0.238060152 -0.144616609 -0.348672754  0.403228509 -0.894546469
## 115 -0.405896218 -1.533294738  0.397748506 -0.193336009  0.101345888
## 116  0.036438700 -0.689569421 -0.161157309  0.100827730  0.371855692
## 117  0.118600537  0.304932262 -0.086107782  0.900186217  0.414448791
## 118  0.572496106  0.705067448 -0.860721680 -0.166546572  0.288472955
## 119  0.889619432 -0.088845486  0.214440660 -0.406193409  0.738464359
## 120  0.657952454 -0.760072659 -0.977671818  0.916535413  0.768092646
## 121  1.210374124 -0.396012103 -0.474217570  0.578876529 -0.161031534
## 122  0.455740206 -0.738051114  0.261297684  1.504996396 -0.026085486
## 123  0.663506349  0.101460673  0.349896658 -1.048495099 -0.616373462
## 124 -1.211735112 -1.420483587 -0.383569334  1.607710937 -1.473383881
## 125 -0.220788855 -0.316313061  0.898205412 -0.665084903  0.933393573
## 126  0.988825765 -1.340255199  0.456408879 -0.799061786  0.101785204
## 127 -0.120271733  0.393336580  0.574597215  0.608867830  0.316937816
## 128 -0.326738410  0.053666703  0.571003730  0.152764307 -1.381485217
## 129 -0.337365952 -0.472593744  0.337695520 -0.692117648  0.020805703
## 130 -0.951672766 -1.109520920 -0.177697905 -0.603647382  1.122057554
## 131 -0.267796094 -0.628588340 -0.218665913 -0.227185133  0.446394924
## 132  0.292148425  0.056061840  0.413060246  0.676449405 -0.534370210
## 133  0.115649677 -0.375062350  0.058260575  0.196759933  0.094410239
## 134 -0.620909038  0.599572755  0.076704776 -0.246582209  0.098944899
## 135 -0.622095744 -0.041442308  0.335204776 -0.849568785  0.685106290
## 136  1.026039831 -1.017111543  0.437474606  0.116261495  0.205766531
## 137  0.014293991  0.948978030 -0.419086073  1.027058966  0.156193806
## 138 -0.916352180 -0.890379318 -0.158252245  0.839681931 -0.295173879
## 139  1.159497991 -0.206036749 -0.165341276  0.364185112  0.170698862
## 140 -1.000799076  1.250789395  0.692082872  0.246862819 -1.397096795
## 141  0.829929052 -0.619158382 -0.449582860 -0.701271126 -0.275591071
## 142 -1.176369004  0.904700146  1.387089363 -0.495943842 -0.091340797
## 143  0.411931701  0.870587814 -0.344606519  0.038002792 -0.834646476
## 144  0.484896085  0.201794981 -0.152451030 -0.212397727  0.514260278
## 145  0.128369702 -0.334080393  0.376284557 -0.179190408 -0.280578290
## 146  0.374032891  0.391940409 -0.040826646 -0.776494293  0.223560465
## 147  0.304387401  0.008481691 -0.293310066  0.429538964 -0.628738856
## 148 -0.128747602 -1.811706439  0.228567702 -0.737221934  0.376316795
## 149  0.358436099 -0.244622731 -0.215322666 -0.195825493  0.449633857
## 150 -0.639203565  0.192113340 -0.622945543 -0.242595735  0.603375093
## 151 -0.710359080 -0.551082341  0.373683067 -0.211047658 -1.776708709
## 152  0.945664639  0.467748330  1.355272156 -0.665187900  0.139424286
## 153 -0.217886687 -0.067860703  0.121270077  0.771391249  0.389508749
## 154  0.321694224  0.041969404 -0.726921835  0.606352038  1.576060206
## 155 -0.312088449 -1.501039053 -0.832924309  0.911891098  0.968889037
## 156 -1.412758952  0.449444134 -0.986196538  0.988923525  0.022188700
## 157  1.582234218  0.619509485  1.420836752 -1.230183337 -0.626424195
## 158 -0.877730961 -0.246545707 -0.048257513 -0.856258122  0.725850781
## 159 -0.060620396  1.380454085 -1.242115406  0.661893747  0.339766646
## 160  0.643988386  0.006589233 -0.759033285 -0.190139303  0.543310269
## 161 -0.072186764  0.051405928  0.175717944 -0.431525658  0.695996951
## 162  0.009390941  1.038581893 -0.116175870  0.960324377  0.623577297
## 163 -0.244542871  0.621151130 -0.403363776  0.303179026 -0.002652101
## 164 -0.473785350 -0.540507777 -0.487570477 -0.400228406  0.647314526
## 165  0.579205869 -0.067360247 -0.948985116  0.117433957  0.725080198
## 166  0.160687413 -0.591735173 -0.528504813 -1.168434501 -0.647314974
## 167 -0.811119221  0.096597186 -0.642521640 -0.742052446 -0.152252317
## 168 -0.091947886  0.237326173 -0.276494573  0.010357144 -0.086175032
## 169  0.011304489 -0.312080399  0.296837815 -1.691307490 -0.194977455
## 170 -0.105641488  0.017242019 -0.017176931  0.742324548  1.361518708
## 171  0.676440580  0.335237278 -0.257863682 -0.398666890 -0.494135648
## 172  1.432990663  0.535148776  0.108057965  0.270789740 -0.541141899
## 173 -0.464852319  1.370156831  0.159839579 -0.808275562 -0.388490368
## 174 -0.054711227 -0.384352503 -0.379377321  0.894369855  0.155551739
## 175 -1.296468737 -0.486484843  2.653765501 -1.035642390  0.335308147
## 176  1.914101548 -0.816931115 -0.662803144  1.085596375 -0.250141984
## 177  0.016411115  1.331863744 -0.020431944  0.621831596 -0.866216069
## 178  0.204429636 -0.957883418 -0.826237855 -0.518524452 -1.125815877
## 179  0.995953518  0.420724405 -0.925209765 -0.210353494 -0.591606784
## 180  0.782707316 -1.122652686  0.429861884 -0.548851279 -0.431935165
## 181 -0.485956803 -0.217262844  0.869736595 -0.772990680  1.010534191
## 182 -1.512109338  0.132069158 -0.361892137 -0.602728379  1.262961540
## 183 -0.318873040  0.390021912  1.110128546 -0.423037789 -0.508724102
## 184 -1.284149901 -0.474146398  0.219975055 -0.214474103  0.578453251
## 185  1.027615024 -0.614549727  0.647866177 -0.459427868 -0.325500487
## 186 -0.257095737 -0.215458524 -0.347032291  0.569270579  0.709373721
## 187  0.067174310  0.010678950 -0.703908813  1.096593207 -0.163630902
## 188 -0.027192425  0.562581105  0.128991122 -0.570458125  1.208767042
## 189 -0.772400262 -1.231514736 -0.402566877  0.641780371 -0.769019716
## 190  0.125056632 -0.849461429 -1.642523429  0.339995264  0.353499128
## 191  0.699740252 -0.684251606 -0.626712116  0.126102752 -1.391395412
## 192  1.038242174  1.163444798 -1.100210618 -0.062182148 -0.738549437
## 193 -0.830781980  0.266898216  0.256543151  0.240675559 -0.566439483
## 194 -0.088820416  0.622078445 -0.241567039 -0.484768735  0.051350756
## 195 -0.654969481  0.413499738  0.300148629  0.468004366  0.059220282
## 196 -1.362357796 -1.509976024  0.840336524  0.375396338 -0.030349660
## 197  0.368092317  0.663250286  0.711397649 -0.171901203  0.532425044
## 198 -1.230020232 -0.104332364 -0.854297812  0.764518689 -0.985287287
## 199 -1.026520839  2.788064888 -0.834861102 -0.498993545  0.130726392
## 200  1.524014637 -0.457386064  0.858003573 -0.293263954  0.232680844
## 201 -1.187140415 -0.223466143 -2.013086065  0.771843304 -1.090264187
## 202 -0.463102884  1.286269224  1.271314993  1.140054794  0.345369283
## 203 -1.041955902  0.028603332 -0.542672664  0.224869175  1.571013526
## 204  0.770137252  0.239375866  0.149546177 -0.846089204 -0.244566201
## 205 -0.093328138  0.523991992 -0.391004927  0.361241896 -0.481868188
## 206 -1.337970929  0.051877802 -0.454426184  0.088733388 -0.514268460
## 207  1.063883534 -0.439304414 -0.342650731  0.029917360  1.110113920
## 208  0.180701735 -0.400699760  0.397227234  0.895096299  0.464573786
## 209 -0.794075016 -0.833874350  1.020371661  0.488510337 -0.451528428
## 210 -0.793608317  0.106991077  0.323951819  0.278695129  0.476711776
## 211  1.162166439  1.360709737  0.925929438 -0.413910664 -0.104947355
## 212  0.449021386 -0.223756403 -0.276026728 -0.203911551  0.905330543
## 213  0.129526491  0.012935660  1.120166041  0.188356634  0.464443796
## 214 -1.267981698  1.145559007  1.111867594  0.256179089 -0.688580051
## 215 -0.302696510  0.383813222 -1.205426440 -0.162600945 -0.493968561
## 216 -0.347590344  0.529144083  1.565005001  0.114973257  0.180297008
## 217 -0.247430292 -0.270568003 -1.399018788  0.454241461 -0.578448818
##              PC46         PC47          PC48        PC49 user_namecarlitos
## 1   -0.6036069627 -1.482024487  0.9241676441 -0.69945828                 1
## 2   -0.5849994654  0.699212392 -0.1221885473 -0.61355690                 0
## 3   -0.9087958378  0.305001928 -0.2070467437  0.82099649                 0
## 4   -0.8153505264  1.627367613  1.3783010000 -1.22261712                 0
## 5   -0.0278695222 -0.242940821 -0.2523953597 -0.65110649                 0
## 6    0.2619310588 -0.221672897  0.4946172075  0.18460745                 0
## 7    0.7659871395  0.662070833  0.2086914962  0.38441065                 0
## 8    1.8117914938 -0.195585921 -0.7697666015 -0.22949195                 0
## 9    0.5215767214 -0.750356107 -0.3050546017  0.27622591                 0
## 10   0.3551739850  0.471372608  0.1504548021 -0.45641306                 0
## 11   0.1323733413 -0.218629963 -0.2007926356  0.52193818                 0
## 12  -0.4050645158  0.588317901  0.3049367005 -0.38154416                 0
## 13  -0.1397710803 -0.349468065  0.5222198026  0.72745810                 0
## 14   0.3911266980  0.177881495 -0.1194526832 -0.43887344                 0
## 15  -0.4492689873  1.122167803 -1.0313701582  0.29764465                 0
## 16  -0.8040517168  0.480605052 -1.4118635954 -0.79710843                 0
## 17  -0.2031182486 -0.394262557 -0.8602906889  0.29514414                 0
## 18  -0.5355157657 -0.489142546  0.4091840970 -0.55226934                 0
## 19  -0.4884819863 -0.156142766  0.1617512641 -0.22057336                 0
## 20  -1.1010977838 -0.936559897 -0.0092174782  0.35830194                 0
## 21   0.1028503399  0.367908791 -0.1027850796  0.34460382                 0
## 22  -0.6914891738 -0.239131588  0.3537380982 -0.11121552                 0
## 23  -0.3751206674  0.878357537 -0.3785767773  0.09795874                 0
## 24  -0.3629770871  0.508831361 -0.5495861660  0.86002346                 0
## 25   0.4951997060  0.221547335  0.2216753538 -0.11353316                 0
## 26   0.6954265776  0.986639443  0.2014078157 -0.66179482                 0
## 27  -0.6501888537  1.797836613 -1.0094582501 -0.05797979                 1
## 28  -1.3691942474 -0.301844273 -1.1284379122 -0.06836469                 1
## 29   0.9851993585  0.691387117 -0.0283635985 -1.30057134                 1
## 30  -0.3143748853 -0.505733548 -0.4713653178 -0.70856458                 1
## 31   0.0842720559 -0.864802549 -0.4747741836  0.06677795                 1
## 32   0.4255891007  0.763645670  1.0392596578 -0.01430404                 1
## 33   0.1274096853 -0.496230257 -1.3112774884 -0.25466065                 0
## 34  -0.2539211555  0.171003541 -0.5859386936 -0.10455653                 0
## 35  -0.9463519135  0.691484979  0.0717083546  0.77724068                 0
## 36   2.1341448781  0.317009462 -1.3475945338  0.56665842                 0
## 37   0.7579089826  0.441001863 -0.9232542920  0.80892558                 0
## 38  -0.6479335725 -0.025476024  0.2020424490 -0.21977175                 0
## 39   0.0270363672 -0.956941550  0.5923922538 -0.41039910                 0
## 40  -1.1240866408  0.519440895  0.9310007018 -0.33819920                 0
## 41   0.4159525071  0.173758870  0.2087542595 -0.57441664                 0
## 42  -0.0629751791 -0.697662093 -0.7079498171 -0.15375117                 0
## 43   0.5981860881  0.003550551 -0.2744482167  0.43728893                 0
## 44   0.2024386858  0.226129798  0.4357954388 -0.46318923                 0
## 45   0.1376248438 -0.078597195 -0.6127074492 -0.35244474                 0
## 46   0.0666836849 -0.948576319  0.1165771995  0.50040469                 0
## 47   0.9190872816 -1.903590065 -0.1900797321  0.11875075                 0
## 48   0.2619668097  0.621253138  0.4699680092 -0.17519314                 0
## 49  -0.3195546272  1.159955805 -0.9219718355 -0.25074490                 0
## 50  -0.5975084367 -0.185373425 -1.1378843120 -0.23259330                 0
## 51  -0.1498698008 -0.069898907  0.1612621376  0.48933078                 0
## 52  -0.1276501328  0.603434032 -0.5154848875 -0.24850958                 0
## 53  -0.1266143740  0.796226853  0.9145924526  0.87808257                 0
## 54   0.4727153664 -0.894010567  0.8545139795 -0.41483155                 0
## 55   0.0075140616  0.407363487  0.5440972473  0.34438504                 0
## 56  -1.0412014660  0.108020406 -0.3929341812 -1.55985212                 0
## 57   0.0995860513 -1.066600949 -0.7330730971 -0.11934094                 0
## 58   0.0590977690  0.453625638  0.7065518799  0.68408141                 0
## 59  -0.2199924308  1.073376646 -0.8245880223 -0.16186352                 0
## 60   0.6375863828  0.155603598 -0.5645331743  1.03658880                 0
## 61   0.1708377057 -0.952924025 -0.8896041932  0.42880378                 0
## 62  -0.4906065302 -1.155374624 -0.5008786590 -1.99658100                 0
## 63  -0.2928584678 -0.978169018 -1.1154736705 -1.60536726                 0
## 64   0.6320527563  0.545893227  0.5761874900  0.53351988                 0
## 65   0.3314017990 -0.604813124 -0.9080010366  0.76377485                 0
## 66  -1.4656457136 -0.638721051  0.3429986127 -0.45552805                 0
## 67   1.0824684090  1.111113610  0.4744918834  0.04544047                 0
## 68   0.0653635088  0.548136165  0.3175599734  0.71699129                 1
## 69   0.9319455078  0.352456292  0.7794661067 -0.32262282                 1
## 70  -0.4507963370 -0.800566026  0.4870281766 -0.09959022                 1
## 71   1.7222892891 -1.015188752 -1.1300591912 -1.20326720                 1
## 72  -0.0299652574 -1.045402778 -0.2220598988 -0.07256783                 1
## 73  -0.7091210093 -0.758856845 -0.5343949231 -0.48538279                 1
## 74   0.8163377619 -0.257653019 -0.0736668687  1.08928698                 1
## 75   0.5376239381  0.364438185  0.3612680015 -0.10115590                 1
## 76   0.2752596460  0.020691759 -0.1891688060  0.03152052                 1
## 77   0.3183152306  1.499494495  0.3252560586 -0.59794810                 1
## 78   0.2172793854  0.164995527 -0.5138750345  0.28961617                 0
## 79   0.6881740911 -0.364679835 -0.6085343724  0.07434051                 0
## 80   0.9729495940  0.080961581 -0.2943398241 -0.18166523                 0
## 81   0.3803443942 -0.926404688 -0.2681492381 -0.14898630                 0
## 82   0.1920170058  0.634094776  0.0257652662  0.26153944                 0
## 83   0.6422238113 -0.681078616 -0.1719684160  0.86512055                 0
## 84   0.8552004656 -0.530991431  0.2039366175  0.10917043                 0
## 85  -0.2147837546  1.028241741 -0.0224118829  0.50791932                 0
## 86   1.3377709721  0.518376807 -0.0710440504  1.01441078                 0
## 87   0.1206270668  0.709384003  0.0429546374 -0.37791762                 0
## 88  -1.1397079440  1.137855025  0.0371362978  0.69612264                 0
## 89   0.1820593154 -0.262554706 -0.1699084380  0.42599575                 0
## 90   0.3210812436 -0.484684709 -0.3291648301 -0.11267208                 0
## 91  -0.5543517684 -0.091117507  0.1175311549  0.35327598                 0
## 92  -0.4609527588 -1.909068052 -0.4270045505  0.82892664                 0
## 93  -0.0313853415 -0.361929244  0.0316097288  0.75117505                 0
## 94  -0.2248352485  0.219548193 -0.8200031073 -1.30708220                 0
## 95  -0.9549280228 -1.638781942  1.1211100947  0.71581836                 0
## 96   0.8930434845 -0.720364715 -0.3611042587 -0.22593158                 0
## 97   0.5347827829 -2.211784919  0.5134031720  0.95619958                 0
## 98   0.9105138921  0.535697702  0.0007502552  0.38736345                 0
## 99   0.4831745628  0.288238812 -0.2787981482 -0.11346343                 1
## 100  0.1710536531  0.432745005 -0.0976049965  1.02210724                 1
## 101  0.5303097146  0.577983521 -0.3332597133  0.07965563                 1
## 102  0.2375964335  0.301977639 -0.1734570030  1.67177479                 1
## 103  1.1879710604  0.023166979  1.7038490868 -0.67113681                 0
## 104  0.2721677740 -0.179321535  0.2595790722  0.04179083                 0
## 105 -0.2882095475 -0.415287527  0.4360489723 -0.19064634                 0
## 106 -1.1585128604 -0.323463887  0.2944610661  1.75192293                 0
## 107 -0.7066009265 -0.044562720 -0.8012991054  1.29676556                 0
## 108  0.0861606537  0.130246435  0.9283469653  0.05309781                 0
## 109 -0.5167079624 -0.129933081  0.0321200237  0.73921873                 0
## 110 -0.3985204762 -0.539087663 -0.3046120454  0.49797336                 0
## 111 -0.2994066667 -0.401056340 -0.6838957676  0.17008336                 0
## 112 -0.2194275384 -0.011964654 -0.6147593609  0.10908310                 0
## 113 -0.0996038849  0.788148044  0.2814116226 -0.32854534                 0
## 114  0.1506912971  0.023907743 -0.9372721219  0.92755059                 0
## 115  0.6864905172 -0.148669911 -0.2867487337 -1.26417243                 0
## 116  0.1510077943  0.448669873  0.5855440290 -0.12198325                 1
## 117 -0.9529340548  1.190227956  0.4685546888 -0.48506520                 1
## 118 -0.1957224580 -0.100284418  0.1103323060 -0.22465521                 1
## 119 -1.5574695736  0.039863237 -0.5798649528  0.20266418                 1
## 120 -0.6985197847  0.006706727 -0.2775145651 -0.07185540                 1
## 121  0.3391163862 -0.956573459  0.4027666322 -0.91506085                 1
## 122 -0.1376896767  0.370990106  1.0894722777 -0.20864800                 1
## 123  0.5587403937 -0.103980991  0.3644003235 -0.69162586                 0
## 124  1.8569602515  0.457161830 -0.4395679119 -0.13318536                 0
## 125 -0.4661838820 -0.479531126  0.1639363191  0.52787529                 0
## 126  0.1487784038  0.364143986  0.3629891830  0.34033409                 0
## 127 -0.9005025664  0.384403700  0.5527548025 -0.02076167                 0
## 128 -0.1528977526  0.533961758 -0.3040830253 -0.48371987                 0
## 129 -0.5708586530 -0.273430686  0.2014085806 -0.85691990                 0
## 130 -0.4271519206 -0.306172701 -0.1163452482  0.21945539                 0
## 131  0.7477336105 -0.335622207 -0.1802383022  0.37590295                 0
## 132  0.6462328310 -0.411960035  0.6979807807 -0.69907948                 0
## 133 -0.1370711055 -0.459480400  1.0074916768 -0.45201052                 0
## 134  0.4218489126  0.595751155  0.1997409311 -0.01234101                 0
## 135 -0.2844656354 -0.240553592 -0.5299303791  0.04276426                 0
## 136 -0.3301011892  0.353681102  0.5472544928  0.20890602                 0
## 137 -0.9191324051 -0.583581789 -0.0770091616  1.09138182                 0
## 138  0.2221994452  0.124847967  0.9446073914  0.77294593                 0
## 139 -0.6926642388  0.318517700 -0.2868366790  0.08885394                 0
## 140 -1.2104893065 -0.481828299  0.5191861143  0.61560704                 1
## 141  0.2217364458 -0.564114034  0.0332720744 -0.65657675                 0
## 142  0.0106039009 -0.513110267 -0.4153397337 -0.43053325                 0
## 143  0.0602937687  0.507467686  0.4734404772 -0.69327173                 0
## 144 -0.1828224697 -0.722805870 -0.0145762150  0.39123070                 0
## 145 -0.1478412864  0.644313122  0.2175412585 -0.72331137                 0
## 146  0.3309361738  0.547075435 -0.0023659315  0.15254699                 0
## 147 -0.7328426342 -0.243401768  0.0917434862 -0.23609795                 0
## 148 -0.2643774272  0.584981626 -0.2018875888 -0.21353338                 0
## 149  0.6999132684  0.118131767  0.5822115353 -0.82376017                 0
## 150 -0.0448569700 -0.455873232 -0.1996320737  0.51336476                 0
## 151  0.0612906852 -0.510431331 -0.0453766627 -0.55457161                 0
## 152 -0.3218745304  0.373800781 -0.6769944237 -0.30461458                 0
## 153  1.4130045861  1.051924929  0.8882489968  1.00909309                 0
## 154  0.2525015770  1.021718300  0.5318803999  0.58258291                 0
## 155  0.3645134978 -0.704162958  0.2799511678 -0.13772331                 0
## 156  0.1356584898 -1.289151330 -0.6102304077 -0.82681022                 0
## 157  0.5006786045 -0.975055475  2.0242890836 -1.13408177                 0
## 158 -0.6948111546  0.469026497 -0.9495395039  1.34346705                 1
## 159  0.0932709761  0.373881615 -0.3683796370  1.06553811                 1
## 160  0.1237528970  0.412442741 -0.0059915301  0.63240200                 1
## 161  0.4022181119 -0.387695471  0.6309580238 -0.41766782                 0
## 162  0.3660211249  1.195358329  0.7524377812 -0.23365459                 0
## 163  0.0025103493  0.386335641  0.0989616125  0.48760286                 0
## 164 -0.7498696510 -0.541413639  0.6988260151 -0.46177166                 0
## 165  0.3048643795  0.721533633  0.2835217526 -1.14726699                 0
## 166  0.3036961832  0.191737487 -0.0032740872 -0.20914663                 0
## 167  0.0739938705 -0.052939804 -0.6102657464  0.58515950                 0
## 168  2.4294956535 -0.413978338  0.0670324557  0.80323077                 0
## 169  0.5139612052  0.380048723  0.5389740186  0.75298776                 0
## 170 -0.1713247719 -0.752328974 -0.1709418229  0.71768964                 0
## 171  0.2356968257  0.400997720  0.1238303165 -0.70815704                 0
## 172 -0.1190242544  0.265714873 -0.3477190615  0.14364216                 0
## 173 -0.0160306583  0.286819492 -0.4114723915 -0.30321877                 0
## 174 -0.4181142441  0.078665795  0.3901937476 -0.07386107                 1
## 175 -0.6382274262 -0.129292447  1.4191885934 -0.80771891                 0
## 176  1.0255980366  0.716250068 -2.1556788518 -1.46084258                 1
## 177  0.6746830697 -0.762564155  1.2786000596 -0.02648620                 1
## 178  1.4104263561 -0.003805388 -0.2025121215 -0.26377062                 1
## 179 -0.4352097824 -0.164321898  0.2442816190  1.10379269                 1
## 180 -0.7467985098 -0.476981464 -0.0981109423 -0.76262847                 1
## 181 -1.2132149828  0.199940108  0.2157399539 -1.00737140                 1
## 182 -1.3160077231  0.362275942  0.2329548574 -1.02818875                 1
## 183  0.3169884907  0.282587572  0.0888926508  0.38885326                 1
## 184 -0.5531826028 -0.250144158  0.5012155314  0.22399096                 1
## 185  0.3432160285  1.000761603  0.8269933305  0.50335206                 1
## 186  0.2669589708  0.352865246 -0.0510980026 -0.31291132                 0
## 187 -0.5424161004 -0.789822302 -0.5173494113 -0.31853536                 0
## 188  0.1046482263  0.002955424 -0.5889282401  0.27821657                 0
## 189 -0.9569548285 -0.539456565 -0.1481657141  0.04285068                 0
## 190  0.5672227327 -0.589844541  0.9498982241 -0.15869221                 0
## 191 -1.4920486564 -0.226506369  0.9167445555  1.22825930                 0
## 192 -0.7552249693 -0.798699011  0.7915261736 -0.07179519                 0
## 193 -1.2460614740  0.983882288 -1.5458928688  0.71008919                 0
## 194  0.7243729001  0.972555013 -1.2112985198  0.02387146                 0
## 195  0.6039157549 -0.719447874 -0.1656258825  0.06119941                 0
## 196 -0.7727614087  1.614090712 -0.4090790635 -1.09268738                 0
## 197  0.1481277093  0.093808498 -0.6354823081  0.26313153                 0
## 198 -0.4992833415  0.711016982 -0.7350829696 -1.16947596                 0
## 199  0.8567043208 -0.030298903  1.6945203402 -0.57214474                 0
## 200 -0.1864942980  0.227508394  1.4169047653  0.30407356                 0
## 201 -0.6052489690  0.589108811  0.5748064913 -0.69244432                 0
## 202  0.0002623613 -0.819554615 -0.2201241155 -1.09855999                 0
## 203 -0.4233011450  0.752403524  1.1739861346 -0.03602922                 0
## 204  0.5887093808  0.104599280 -0.3920981011  0.61452623                 0
## 205 -0.6135938648  0.113720998  1.1511486314  0.42072100                 0
## 206  0.7317553515 -0.422519479 -0.0966921616 -0.41567363                 0
## 207  0.6312069936 -0.460753915  0.8734637878  0.82391600                 0
## 208 -0.9775953020 -0.025063129 -0.5723473791  1.05607316                 0
## 209  0.3756589042 -0.164833295  1.1674134336  0.19332506                 0
## 210 -0.9339128600  0.193190677 -0.7861095826 -0.68852563                 0
## 211 -0.2107035584 -0.166184436 -1.2064436183 -0.45068954                 0
## 212  1.1859764823 -0.752066525 -0.6126521000 -0.31481242                 0
## 213 -0.1441696946 -0.794914536 -0.3468791662  0.53028082                 0
## 214 -1.0135160590  0.238754081 -0.0288176996  0.38013604                 0
## 215 -0.4435375840 -0.705590145 -0.5935763574 -0.01697910                 0
## 216  0.5374883286  1.559003276  0.0722026459  0.40983408                 1
## 217 -0.6061322920 -0.674526489  0.9643024267  0.31648647                 1
##     user_namecharles user_nameeurico user_namejeremy user_namepedro
## 1                  0               0               0              0
## 2                  0               0               0              1
## 3                  0               0               0              1
## 4                  0               0               0              1
## 5                  0               0               0              1
## 6                  0               0               0              1
## 7                  0               0               0              1
## 8                  0               0               0              1
## 9                  0               0               0              1
## 10                 0               0               0              1
## 11                 0               0               0              1
## 12                 0               0               0              1
## 13                 0               0               0              1
## 14                 1               0               0              0
## 15                 0               1               0              0
## 16                 0               1               0              0
## 17                 0               1               0              0
## 18                 0               1               0              0
## 19                 0               1               0              0
## 20                 0               1               0              0
## 21                 0               1               0              0
## 22                 0               1               0              0
## 23                 0               1               0              0
## 24                 0               1               0              0
## 25                 0               1               0              0
## 26                 0               1               0              0
## 27                 0               0               0              0
## 28                 0               0               0              0
## 29                 0               0               0              0
## 30                 0               0               0              0
## 31                 0               0               0              0
## 32                 0               0               0              0
## 33                 1               0               0              0
## 34                 1               0               0              0
## 35                 1               0               0              0
## 36                 1               0               0              0
## 37                 1               0               0              0
## 38                 1               0               0              0
## 39                 1               0               0              0
## 40                 1               0               0              0
## 41                 1               0               0              0
## 42                 1               0               0              0
## 43                 1               0               0              0
## 44                 1               0               0              0
## 45                 1               0               0              0
## 46                 1               0               0              0
## 47                 1               0               0              0
## 48                 1               0               0              0
## 49                 1               0               0              0
## 50                 0               1               0              0
## 51                 0               1               0              0
## 52                 0               1               0              0
## 53                 0               0               0              1
## 54                 0               0               0              1
## 55                 0               0               0              1
## 56                 0               0               0              1
## 57                 0               0               0              1
## 58                 0               0               0              1
## 59                 0               0               0              1
## 60                 0               0               0              1
## 61                 0               0               0              1
## 62                 0               0               0              1
## 63                 0               1               0              0
## 64                 0               1               0              0
## 65                 0               1               0              0
## 66                 0               1               0              0
## 67                 0               1               0              0
## 68                 0               0               0              0
## 69                 0               0               0              0
## 70                 0               0               0              0
## 71                 0               0               0              0
## 72                 0               0               0              0
## 73                 0               0               0              0
## 74                 0               0               0              0
## 75                 0               0               0              0
## 76                 0               0               0              0
## 77                 0               0               0              0
## 78                 1               0               0              0
## 79                 1               0               0              0
## 80                 1               0               0              0
## 81                 1               0               0              0
## 82                 1               0               0              0
## 83                 1               0               0              0
## 84                 1               0               0              0
## 85                 1               0               0              0
## 86                 1               0               0              0
## 87                 1               0               0              0
## 88                 1               0               0              0
## 89                 1               0               0              0
## 90                 1               0               0              0
## 91                 1               0               0              0
## 92                 1               0               0              0
## 93                 1               0               0              0
## 94                 0               1               0              0
## 95                 0               1               0              0
## 96                 0               1               0              0
## 97                 0               1               0              0
## 98                 0               0               0              1
## 99                 0               0               0              0
## 100                0               0               0              0
## 101                0               0               0              0
## 102                0               0               0              0
## 103                0               0               0              1
## 104                0               0               0              1
## 105                0               0               0              1
## 106                0               0               0              1
## 107                0               0               0              1
## 108                0               0               0              1
## 109                0               0               0              1
## 110                0               1               0              0
## 111                0               1               0              0
## 112                0               1               0              0
## 113                0               1               0              0
## 114                0               1               0              0
## 115                0               1               0              0
## 116                0               0               0              0
## 117                0               0               0              0
## 118                0               0               0              0
## 119                0               0               0              0
## 120                0               0               0              0
## 121                0               0               0              0
## 122                0               0               0              0
## 123                1               0               0              0
## 124                1               0               0              0
## 125                1               0               0              0
## 126                1               0               0              0
## 127                1               0               0              0
## 128                1               0               0              0
## 129                1               0               0              0
## 130                1               0               0              0
## 131                1               0               0              0
## 132                1               0               0              0
## 133                1               0               0              0
## 134                1               0               0              0
## 135                1               0               0              0
## 136                1               0               0              0
## 137                0               1               0              0
## 138                0               0               0              1
## 139                0               0               0              1
## 140                0               0               0              0
## 141                0               0               0              1
## 142                0               0               0              1
## 143                0               0               0              1
## 144                0               0               0              1
## 145                0               0               0              1
## 146                0               0               0              1
## 147                0               0               0              1
## 148                0               0               0              1
## 149                1               0               0              0
## 150                1               0               0              0
## 151                1               0               0              0
## 152                0               1               0              0
## 153                0               1               0              0
## 154                0               1               0              0
## 155                0               1               0              0
## 156                0               1               0              0
## 157                0               1               0              0
## 158                0               0               0              0
## 159                0               0               0              0
## 160                0               0               0              0
## 161                1               0               0              0
## 162                1               0               0              0
## 163                1               0               0              0
## 164                1               0               0              0
## 165                1               0               0              0
## 166                1               0               0              0
## 167                1               0               0              0
## 168                0               1               0              0
## 169                0               1               0              0
## 170                0               1               0              0
## 171                0               0               0              1
## 172                0               0               0              1
## 173                0               0               0              1
## 174                0               0               0              0
## 175                0               1               0              0
## 176                0               0               0              0
## 177                0               0               0              0
## 178                0               0               0              0
## 179                0               0               0              0
## 180                0               0               0              0
## 181                0               0               0              0
## 182                0               0               0              0
## 183                0               0               0              0
## 184                0               0               0              0
## 185                0               0               0              0
## 186                0               0               0              1
## 187                0               0               0              1
## 188                1               0               0              0
## 189                1               0               0              0
## 190                1               0               0              0
## 191                1               0               0              0
## 192                1               0               0              0
## 193                1               0               0              0
## 194                1               0               0              0
## 195                1               0               0              0
## 196                1               0               0              0
## 197                1               0               0              0
## 198                1               0               0              0
## 199                1               0               0              0
## 200                1               0               0              0
## 201                0               1               0              0
## 202                0               1               0              0
## 203                0               1               0              0
## 204                0               1               0              0
## 205                0               1               0              0
## 206                0               1               0              0
## 207                0               0               0              1
## 208                0               0               0              1
## 209                0               0               0              1
## 210                0               0               0              1
## 211                0               0               0              1
## 212                0               0               0              1
## 213                0               0               0              1
## 214                0               0               0              1
## 215                0               0               0              1
## 216                0               0               0              0
## 217                0               0               0              0
## 
## $usekernel
## [1] TRUE
## 
## $varnames
##  [1] "PC1"               "PC2"               "PC3"              
##  [4] "PC4"               "PC5"               "PC6"              
##  [7] "PC7"               "PC8"               "PC9"              
## [10] "PC10"              "PC11"              "PC12"             
## [13] "PC13"              "PC14"              "PC15"             
## [16] "PC16"              "PC17"              "PC18"             
## [19] "PC19"              "PC20"              "PC21"             
## [22] "PC22"              "PC23"              "PC24"             
## [25] "PC25"              "PC26"              "PC27"             
## [28] "PC28"              "PC29"              "PC30"             
## [31] "PC31"              "PC32"              "PC33"             
## [34] "PC34"              "PC35"              "PC36"             
## [37] "PC37"              "PC38"              "PC39"             
## [40] "PC40"              "PC41"              "PC42"             
## [43] "PC43"              "PC44"              "PC45"             
## [46] "PC46"              "PC47"              "PC48"             
## [49] "PC49"              "user_namecarlitos" "user_namecharles" 
## [52] "user_nameeurico"   "user_namejeremy"   "user_namepedro"   
## 
## $xNames
##  [1] "PC1"               "PC2"               "PC3"              
##  [4] "PC4"               "PC5"               "PC6"              
##  [7] "PC7"               "PC8"               "PC9"              
## [10] "PC10"              "PC11"              "PC12"             
## [13] "PC13"              "PC14"              "PC15"             
## [16] "PC16"              "PC17"              "PC18"             
## [19] "PC19"              "PC20"              "PC21"             
## [22] "PC22"              "PC23"              "PC24"             
## [25] "PC25"              "PC26"              "PC27"             
## [28] "PC28"              "PC29"              "PC30"             
## [31] "PC31"              "PC32"              "PC33"             
## [34] "PC34"              "PC35"              "PC36"             
## [37] "PC37"              "PC38"              "PC39"             
## [40] "PC40"              "PC41"              "PC42"             
## [43] "PC43"              "PC44"              "PC45"             
## [46] "PC46"              "PC47"              "PC48"             
## [49] "PC49"              "user_namecarlitos" "user_namecharles" 
## [52] "user_nameeurico"   "user_namejeremy"   "user_namepedro"   
## 
## $problemType
## [1] "Classification"
## 
## $tuneValue
##   fL usekernel
## 2  0      TRUE
## 
## $obsLevels
## [1] "A" "B" "C" "D" "E"
## 
## attr(,"class")
## [1] "NaiveBayes"
```
