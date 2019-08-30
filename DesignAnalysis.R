library(SnowballC)
library(tm)
library(syuzhet)
library(ggplot2)
library(stringr)
library(wordcloud)
library(qdap)
library(plyr)
library(caret)
library(quanteda)
library(e1071)
library(irlba)
library(tidytext)

tweets.df <- read.csv('E:\\Masters\\Project\\Kickstarter\\Data\\DesignTech_with_ProjectName.csv', header=TRUE)

set.seed(32984)
indexes <- createDataPartition(tweets.df$Category, times = 1,
                               p = 0.7, list = FALSE)

train <- tweets.df[indexes,]
test <- tweets.df[-indexes,]

prop.table(table(train$Category))
prop.table(table(test$Category))


levels(train$Category)
design <- train[train$Category %in% c("Design & Tech"),]
design_test <- test[test$Category %in% c("Design & Tech"),]
#########################################################

design.tweet = gsub("RT((?:\\b\\W*@\\w+)+)","", design$Text)
design.tweet = gsub("http[^[:blank:]]+","",design.tweet) # removes http://
design.tweet = gsub("@\\w+","",design.tweet)
design.tweet = gsub("#\\w+","",design.tweet)
design.tweet = gsub('[[:punct:]]', ' ', design.tweet) # removes punctuation 
design.tweet = gsub('[^[:alnum:]]', ' ', design.tweet)


design.tokens <- tokens(design.tweet, what = "word", 
                     remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE, remove_hyphens = TRUE)


# Lower case the tokens.
design.tokens <- tokens_tolower(design.tokens)
design.tokens[[9]]


# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to
#        your problem/domain.
design.tokens <- tokens_select(design.tokens, stopwords(), 
                            selection = "remove")
design.tokens[[9]]


# Perform stemming on the tokens.
design.tokens <- tokens_wordstem(design.tokens, language = "english")
design.tokens[[9]]

word.list_design = str_split(design.tokens, '\\s+') # splits the tweets by word in a list

words_design = unlist(word.list_design) # turns the list into vector

word.df_design <- as.vector(design.tweet)

sentiment_design <- get_sentiment(word.df_design)
most.positive_design <- word.df_design[sentiment_design == max(sentiment_design)]

most.positive_design
most.negative_design <- word.df_design[sentiment_design <= min(sentiment_design)]
most.negative_design 

positive.tweets_design <- word.df_design[sentiment_design > 0]

head(positive.tweets_design)

negative.tweets_design <- word.df_design[sentiment_design < 0]
head(negative.tweets_design)

neutral.tweets_design <- word.df_design[sentiment_design == 0] 
head(neutral.tweets_design)

category_senti_design <- ifelse(sentiment_design < 0, "Negative", ifelse(sentiment_design > 0, "Positive", "Neutral"))

head(category_senti_design)

category_senti2_design <- cbind(design.tweet,category_senti_design,sentiment_design) 

head(category_senti2_design)

incomplete.cases_design <- which(!complete.cases(category_senti2_design))
design$Text[incomplete.cases_design]


# Fix incomplete cases
category_senti2_design[incomplete.cases_design,] <- rep(0.0, ncol(category_senti2_design))
dim(category_senti2_design)
sum(which(!complete.cases(category_senti2_design)))


# Make a clean data frame using the same process as before.
category_senti2.df_design <- cbind(Category = design$Category, Project= design$Project, data.frame(category_senti2_design))
names(category_senti2.df_design) <- make.names(names(category_senti2.df_design))

sum_senti_design <- summary(category_senti2_design)
write.csv(category_senti2.df_design, 'E:\\Masters\\Project\\Kickstarter\\Data\\CategoryWiseAnalyse\\Sentiment_scores_design.csv', row.names = FALSE)

qplot(category_senti2.df_design$category_senti_design, data=category_senti2.df_design, fill=Category)+
  ggtitle("Distribution of Sentiments of Tweets across Design & Tech")
qplot(category_senti2.df_design$category_senti_design, data=category_senti2.df_design, fill=Project)+
  ggtitle("Distribution of Sentiments of Tweets across Projects of Design & Tech")
#############################################################
##############################################################################
# Create our first bag-of-words model.
design.tokens.dfm <- dfm(design.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
design.tokens.matrix <- as.matrix(design.tokens.dfm)
#View(train.tokens.matrix[1:20, 1:100])
dim(design.tokens.matrix)

# Setup a the feature data frame with labels.
design.tokens.df <- cbind(Category = design$Category, data.frame(design.tokens.dfm))


# Often, tokenization requires some additional pre-processing
names(design.tokens.df)[c(146, 148, 235, 238)]

# Cleanup column names.
names(design.tokens.df) <- make.names(names(design.tokens.df))

term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}


# First step, normalize all documents via TF.
design.tokens.df <- apply(design.tokens.matrix, 1, term.frequency)
dim(design.tokens.df)

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
design.tokens.idf <- apply(design.tokens.matrix, 2, inverse.doc.freq)
str(design.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
design.tokens.tfidf <-  apply(design.tokens.df, 2, tf.idf, idf = design.tokens.idf)
dim(design.tokens.tfidf)


# Transpose the matrix
design.tokens.tfidf <- t(design.tokens.tfidf)
dim(design.tokens.tfidf)


# Check for incopmlete cases.
incomplete.cases_design <- which(!complete.cases(design.tokens.tfidf))
design$Text[incomplete.cases_design]


# Fix incomplete cases
design.tokens.tfidf[incomplete.cases_design,] <- rep(0.0, ncol(design.tokens.tfidf))
dim(design.tokens.tfidf)
sum(which(!complete.cases(design.tokens.tfidf)))


# Make a clean data frame using the same process as before.
design.tokens.tfidf.df <- cbind(Sentiment = category_senti2.df_design$category_senti_design, data.frame(design.tokens.tfidf))
names(design.tokens.tfidf.df) <- make.names(names(design.tokens.tfidf.df))


library(doSNOW)
set.seed(48743)
cv.folds <- createMultiFolds(category_senti2.df_design$category_senti_design, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)

# Time the code execution
start.time <- Sys.time()

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm as our first model. We will graduate to using more 
# powerful algorithms later when we perform feature extraction to shrink
# the size of our data.
rpart.cv.2 <- train(Sentiment ~ ., data = design.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.2

gc()

library(irlba)


# Time the code execution
start.time <- Sys.time()

# Perform SVD. Specifically, reduce dimensionality down to 300 columns
# for our latent semantic analysis (LSA).
design.irlba <- irlba(t(design.tokens.tfidf), nv = 300, maxit = 600)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time


# Take a look at the new feature data up close.
#View(train.irlba$v)


# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
#
#
sigma.inverse <- 1 / design.irlba$d
u.transpose <- t(design.irlba$u)
document <- design.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
design.irlba$v[1, 1:10]



#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
design.svd <- data.frame(Sentiment = category_senti2.df_design$category_senti_design, design.irlba$v)

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.4 <- train(Sentiment ~ ., data = design.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.4

##########################################################################################
svmfit_design <- svm(Sentiment~., data=design.svd, kernel="radial")
print(svmfit_design)

library(randomForest)
rf_design <- randomForest(Sentiment~., data=design.tokens.tfidf.df)
rf_design_train <- randomForest(Sentiment~., data=design.svd)

nb_design <- naiveBayes(Sentiment ~., data= design.svd)

varImpPlot(rf_design)
######################################################################################
###################### Testing the model #############################################

design_test.tweet = gsub("RT((?:\\b\\W*@\\w+)+)","", design_test$Text)
design_test.tweet = gsub("http[^[:blank:]]+","",design_test.tweet) # removes http://
design_test.tweet = gsub("@\\w+","",design_test.tweet)
design_test.tweet = gsub("#\\w+","",design_test.tweet)
design_test.tweet = gsub('[[:punct:]]', ' ', design_test.tweet) # removes punctuation 
design_test.tweet = gsub('[^[:alnum:]]', ' ', design_test.tweet)

design_test.tokens <- tokens(design_test.tweet, what = "word", 
                          remove_numbers = TRUE, remove_punct = TRUE,
                          remove_symbols = TRUE, remove_hyphens = TRUE)

# Lower case the tokens.
design_test.tokens <- tokens_tolower(design_test.tokens)

# Stopword removal.
design_test.tokens <- tokens_select(design_test.tokens, stopwords(), 
                                 selection = "remove")

# Stemming.
design_test.tokens <- tokens_wordstem(design_test.tokens, language = "english")

word.df.design_test <- as.vector(design_test.tweet)

sentiment_test <- get_sentiment(word.df.design_test)
most.positive_test <- word.df.design_test[sentiment_test == max(sentiment_test)]

most.positive_test
most.negative_test <- word.df.design_test[sentiment_test <= min(sentiment_test)]
most.negative_test

positive.tweets_test <- word.df.design_test[sentiment_test > 0]

head(positive.tweets_test)

negative.tweets_test <- word.df.design_test[sentiment_test < 0]
head(negative.tweets_test)

neutral.tweets_test <- word.df.design_test[sentiment_test == 0] 
head(neutral.tweets_test)

category_senti_test <- ifelse(sentiment_test < 0, "Negative", ifelse(sentiment_test > 0, "Positive", "Neutral"))

head(category_senti_test)

category_senti2.design_test <- cbind(design_test.tweet,category_senti_test,sentiment_test) 

head(category_senti2.design_test)

incomplete.cases_test <- which(!complete.cases(category_senti2.design_test))
design_test$Text[incomplete.cases_test]


# Fix incomplete cases
category_senti2.design_test[incomplete.cases_test,] <- rep(0.0, ncol(category_senti2.design_test))
dim(category_senti2.design_test)
sum(which(!complete.cases(category_senti2.design_test)))


# Make a clean data frame using the same process as before.
category_senti2.design_test.df <- cbind(Category = design_test$Category, Project= design_test$Project, data.frame(category_senti2.design_test))
names(category_senti2.design_test.df) <- make.names(names(category_senti2.design_test.df))

sum_senti <- summary(category_senti2.design_test)
################################################################################

# Convert n-grams to quanteda document-term frequency matrix.
design_test.tokens.dfm <- dfm(design_test.tokens, tolower = FALSE)

# Explore the train and design_test quanteda dfm objects.
design.tokens.dfm
design_test.tokens.dfm

# Ensure the design_test dfm has the same n-grams as the training dfm.
#
# NOTE - In production we should expect that new text messages will 
#        contain n-grams that did not exist in the original training
#        data. As such, we need to strip those n-grams out.
#
design_test.tokens.dfm <- dfm_select(design_test.tokens.dfm, pattern = design.tokens.dfm,
                                  selection = "keep")
design_test.tokens.matrix <- as.matrix(design_test.tokens.dfm)
design_test.tokens.dfm




# With the raw design_test features in place next up is the projecting the term
# counts for the unigrams into the same TF-IDF vector space as our training
# data. The high level process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values

# Normalize all documents via TF.
design_test.tokens.df <- apply(design_test.tokens.matrix, 1, term.frequency)
str(design_test.tokens.df)

# Lastly, calculate TF-IDF for our training corpus.
design_test.tokens.tfidf <-  apply(design_test.tokens.df, 2, tf.idf, idf = design.tokens.idf)
dim(design_test.tokens.tfidf)
#View(design_test.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
design_test.tokens.tfidf <- t(design_test.tokens.tfidf)

# Fix incomplete cases
summary(design_test.tokens.tfidf[1,])
design_test.tokens.tfidf[is.na(design_test.tokens.tfidf)] <- 0.0
summary(design_test.tokens.tfidf[1,])

# With the design_test data projected into the TF-IDF vector space of the training
# data we can now to the final projection into the training LSA semantic
# space (i.e. the SVD matrix factorization).
design_test.svd.raw <- t(sigma.inverse * u.transpose %*% t(design_test.tokens.tfidf))


# Lastly, we can now build the design_test data frame to feed into our trained
# machine learning model for predictions. First up, add Label and TextLength.
design_test.irlba <- irlba(t(design_test.tokens.tfidf), nv = 300, maxit = 600)

#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
design_test.svd <- data.frame(Sentiment = category_senti2.design_test.df$category_senti_test, design_test.svd.raw)
#
# NOTE - The following code was updated post-video recoding due to a bug.
#
compareTable <- table(category_senti2.design_test.df$category_senti_test, predict(rpart.cv.4,design_test.svd))
misClassify <- mean(category_senti2.design_test.df$category_senti_test != predict(rpart.cv.4,design_test.svd))
print(paste("Accuracy of RPART is",1-misClassify))

compareTable_svm <- table(category_senti2.design_test.df$category_senti_test, predict(svmfit_design,design_test.svd))
misClassify_svm <- mean(category_senti2.design_test.df$category_senti_test != predict(svmfit_design,design_test.svd))
print(paste("Accuracy of SVM Kernel is",1-misClassify_svm))

compareTable_rf <- table(category_senti2.design_test.df$category_senti_test, predict(rf_design_train,design_test.svd))
misClassify_rf <- mean(category_senti2.design_test.df$category_senti_test != predict(rf_design_train,design_test.svd))
print(paste("Accuracy of RF is",1-misClassify_rf))

compareTable_nb <- table(category_senti2.design_test.df$category_senti_test, predict(nb_design,design_test.svd))
misClassify_nb <- mean(category_senti2.design_test.df$category_senti_test != predict(nb_design,design_test.svd))
print(paste("Accuracy of NB is",1-misClassify_nb))

