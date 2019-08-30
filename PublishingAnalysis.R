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
library(qdapRegex)
library(sentimentr)

setwd("E:/Masters/Project/Kickstarter/Data")

tweets.df <- read.csv('DesignTech_with_ProjectName.csv', header=TRUE)

levels(tweets.df$Category)
Publishing <- train[tweets.df$Category %in% c("Publishing"),]

#########################################################
Publishing.tweet <- replace_contraction(Publishing$Text)

Publishing.tweet = gsub("RT((?:\\b\\W*@\\w+)+)","", Publishing.tweet)
Publishing.tweet = gsub("http[^[:blank:]]+","",Publishing.tweet) # removes http://
Publishing.tweet = gsub("@\\w+","",Publishing.tweet)
Publishing.tweet = gsub("#\\w+","",Publishing.tweet)
Publishing.tweet = gsub('[[:punct:]]', ' ', Publishing.tweet) # removes punctuation 
Publishing.tweet = gsub('[^[:alnum:]]', ' ', Publishing.tweet)


df <- extract_sentiment_terms(Publishing.tweet)
Publishing.tokens <- tokens(Publishing.tweet, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)


# Lower case the tokens.
Publishing.tokens <- tokens_tolower(Publishing.tokens)
Publishing.tokens[[9]]


# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to
#        your problem/domain.
Publishing.tokens <- tokens_select(Publishing.tokens, stopwords(), 
                              selection = "remove")
Publishing.tokens[[9]]


# Perform stemming on the tokens.
Publishing.tokens <- tokens_wordstem(Publishing.tokens, language = "english")
Publishing.tokens[[9]]

word.list_Publishing = str_split(Publishing.tokens, '\\s+') # splits the tweets by word in a list

words_Publishing = unlist(word.list_Publishing) # turns the list into vector

word.df_Publishing <- as.vector(Publishing.tweet)

sentiment_Publishing <- get_sentiment(word.df_Publishing)
most.positive_Publishing <- word.df_Publishing[sentiment_Publishing == max(sentiment_Publishing)]

most.positive_Publishing
most.negative_Publishing <- word.df_Publishing[sentiment_Publishing <= min(sentiment_Publishing)]
most.negative_Publishing 

positive.tweets_Publishing <- word.df_Publishing[sentiment_Publishing > 0]

head(positive.tweets_Publishing)

negative.tweets_Publishing <- word.df_Publishing[sentiment_Publishing < 0]
head(negative.tweets_Publishing)

neutral.tweets_Publishing <- word.df_Publishing[sentiment_Publishing == 0] 
head(neutral.tweets_Publishing)

category_senti_Publishing <- ifelse(sentiment_Publishing < 0, "Negative", ifelse(sentiment_Publishing > 0, "Positive", "Neutral"))

head(category_senti_Publishing)

category_senti2_Publishing <- cbind(Publishing.tweet,category_senti_Publishing,sentiment_Publishing) 

head(category_senti2_Publishing)

incomplete.cases_Publishing <- which(!complete.cases(category_senti2_Publishing))
Publishing$Text[incomplete.cases_Publishing]


# Fix incomplete cases
category_senti2_Publishing[incomplete.cases_Publishing,] <- rep(0.0, ncol(category_senti2_Publishing))
dim(category_senti2_Publishing)
sum(which(!complete.cases(category_senti2_Publishing)))
nrc<- get_sentiment(method="nrc", language="english")

# Make a clean data frame using the same process as before.
category_senti2.df_Publishing <- cbind(Category = Publishing$Category, Project= Publishing$Project, data.frame(category_senti2_Publishing))
names(category_senti2.df_Publishing) <- make.names(names(category_senti2.df_Publishing))

sum_senti_Publishing <- summary(category_senti2_Publishing)
write.csv(category_senti2.df_Publishing, 'E:\\Masters\\Project\\Kickstarter\\Data\\CategoryWiseAnalyse\\Sentiment_scores_Publishing.csv', row.names = FALSE)

qplot(category_senti2.df_Publishing$category_senti_Publishing, data=category_senti2.df_Publishing, fill=Category)+
  ggtitle("Distribution of Sentiments of Tweets across Publishing")
qplot(category_senti2.df_Publishing$category_senti_Publishing, data=category_senti2.df_Publishing, fill=Project)+
  ggtitle("Distribution of Sentiments of Tweets across Projects of Publishing")

#############################################################
##############################################################################
# Create our first bag-of-words model.
Publishing.tokens.dfm <- dfm(Publishing.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
Publishing.tokens.matrix <- as.matrix(Publishing.tokens.dfm)
#View(train.tokens.matrix[1:20, 1:100])
dim(Publishing.tokens.matrix)

# Often, tokenization requires some additional pre-processing
names(Publishing.tokens.df)[c(146, 148, 235, 238)]

# Cleanup column names.
names(Publishing.tokens.df) <- make.names(names(Publishing.tokens.df))

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
Publishing.tokens.df <- apply(Publishing.tokens.matrix, 1, term.frequency)
dim(Publishing.tokens.df)

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
Publishing.tokens.idf <- apply(Publishing.tokens.matrix, 2, inverse.doc.freq)
str(Publishing.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
Publishing.tokens.tfidf <-  apply(Publishing.tokens.df, 2, tf.idf, idf = Publishing.tokens.idf)
dim(Publishing.tokens.tfidf)


# Transpose the matrix
Publishing.tokens.tfidf <- t(Publishing.tokens.tfidf)
dim(Publishing.tokens.tfidf)


# Check for incopmlete cases.
incomplete.cases_Publishing <- which(!complete.cases(Publishing.tokens.tfidf))
Publishing$Text[incomplete.cases_Publishing]


# Fix incomplete cases
Publishing.tokens.tfidf[incomplete.cases_Publishing,] <- rep(0.0, ncol(Publishing.tokens.tfidf))
dim(Publishing.tokens.tfidf)
sum(which(!complete.cases(Publishing.tokens.tfidf)))


# Make a clean data frame using the same process as before.
Publishing.tokens.tfidf.df <- cbind(Sentiment = category_senti2.df_Publishing$category_senti_Publishing, data.frame(Publishing.tokens.tfidf))
names(Publishing.tokens.tfidf.df) <- make.names(names(Publishing.tokens.tfidf.df))


library(doSNOW)
set.seed(48743)
cv.folds <- createMultiFolds(category_senti2.df_Publishing$category_senti_Publishing, k = 10, times = 3)

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
rpart.cv.2 <- train(Sentiment ~ ., data = Publishing.tokens.tfidf.df, method = "rpart", 
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
Publishing.irlba <- irlba(t(Publishing.tokens.tfidf), nv = 300, maxit = 600)

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
sigma.inverse <- 1 / Publishing.irlba$d
u.transpose <- t(Publishing.irlba$u)
document <- Publishing.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
Publishing.irlba$v[1, 1:10]



#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
Publishing.svd <- data.frame(Sentiment = category_senti2.df_Publishing$category_senti_Publishing, Publishing.irlba$v)

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.4 <- train(Sentiment ~ ., data = Publishing.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.4

##########################################################################################
svmfit_Publishing <- svm(Sentiment~., data=category_senti2.df_Publishing, kernel="radial")
print(svmfit_Publishing)

library(randomForest)
rf_Publishing <- randomForest(Sentiment~., data=Publishing.svd)

nb_Publishing <- naiveBayes(Sentiment ~., data= Publishing.svd)

rf_Publishing_imp <- randomForest(Sentiment~., data=Publishing.tokens.tfidf.df)
varImpPlot(rf_Publishing_imp)

Publishing_glm <-  glm(Sentiment ~ ., data=Publishing.tokens.tfidf.df)
importance(rf_Publishing_imp)
########################################################

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.5 <- train(Sentiment ~ ., data = Publishing.svd, method = "knn", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.5
########################################################
# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.6 <- train(Sentiment ~ ., data = Publishing.svd, method = "nnet", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.6

######################################################################################
###################### Testing the model #############################################

art_test.tweet = gsub("RT((?:\\b\\W*@\\w+)+)","", art_test$Text)
art_test.tweet = gsub("http[^[:blank:]]+","",art_test.tweet) # removes http://
art_test.tweet = gsub("@\\w+","",art_test.tweet)
art_test.tweet = gsub("#\\w+","",art_test.tweet)
art_test.tweet = gsub('[[:punct:]]', ' ', art_test.tweet) # removes punctuation 
art_test.tweet = gsub('[^[:alnum:]]', ' ', art_test.tweet)

art_test.tokens <- tokens(art_test.tweet, what = "word", 
                          remove_numbers = TRUE, remove_punct = TRUE,
                          remove_symbols = TRUE, remove_hyphens = TRUE)

# Lower case the tokens.
art_test.tokens <- tokens_tolower(art_test.tokens)

# Stopword removal.
art_test.tokens <- tokens_select(art_test.tokens, stopwords(), 
                                 selection = "remove")

# Stemming.
art_test.tokens <- tokens_wordstem(art_test.tokens, language = "english")

word.df.art_test <- as.vector(art_test.tweet)

sentiment_test <- get_sentiment(word.df.art_test)
most.positive_test <- word.df.art_test[sentiment_test == max(sentiment_test)]

most.positive_test
most.negative_test <- word.df.art_test[sentiment_test <= min(sentiment_test)]
most.negative_test

positive.tweets_test <- word.df.art_test[sentiment_test > 0]

head(positive.tweets_test)

negative.tweets_test <- word.df.art_test[sentiment_test < 0]
head(negative.tweets_test)

neutral.tweets_test <- word.df.art_test[sentiment_test == 0] 
head(neutral.tweets_test)

category_senti_test <- ifelse(sentiment_test < 0, "Negative", ifelse(sentiment_test > 0, "Positive", "Neutral"))

head(category_senti_test)

category_senti2.art_test <- cbind(art_test.tweet,category_senti_test,sentiment_test) 

head(category_senti2.art_test)

incomplete.cases_test <- which(!complete.cases(category_senti2.art_test))
art_test$Text[incomplete.cases_test]


# Fix incomplete cases
category_senti2.art_test[incomplete.cases_test,] <- rep(0.0, ncol(category_senti2.art_test))
dim(category_senti2.art_test)
sum(which(!complete.cases(category_senti2.art_test)))


# Make a clean data frame using the same process as before.
category_senti2.art_test.df <- cbind(Category = art_test$Category, Project= art_test$Project, data.frame(category_senti2.art_test))
names(category_senti2.art_test.df) <- make.names(names(category_senti2.art_test.df))

sum_senti <- summary(category_senti2.art_test)
################################################################################

# Convert n-grams to quanteda document-term frequency matrix.
art_test.tokens.dfm <- dfm(art_test.tokens, tolower = FALSE)

# Explore the train and art_test quanteda dfm objects.
art.tokens.dfm
art_test.tokens.dfm

# Ensure the art_test dfm has the same n-grams as the training dfm.
#
# NOTE - In production we should expect that new text messages will 
#        contain n-grams that did not exist in the original training
#        data. As such, we need to strip those n-grams out.
#
art_test.tokens.dfm <- dfm_select(art_test.tokens.dfm, pattern = art.tokens.dfm,
                                  selection = "keep")
art_test.tokens.matrix <- as.matrix(art_test.tokens.dfm)
art_test.tokens.dfm




# With the raw art_test features in place next up is the projecting the term
# counts for the unigrams into the same TF-IDF vector space as our training
# data. The high level process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values

# Normalize all documents via TF.
art_test.tokens.df <- apply(art_test.tokens.matrix, 1, term.frequency)
str(art_test.tokens.df)

# Lastly, calculate TF-IDF for our training corpus.
art_test.tokens.tfidf <-  apply(art_test.tokens.df, 2, tf.idf, idf = art.tokens.idf)
dim(art_test.tokens.tfidf)
View(art_test.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
art_test.tokens.tfidf <- t(art_test.tokens.tfidf)

# Fix incomplete cases
summary(art_test.tokens.tfidf[1,])
art_test.tokens.tfidf[is.na(art_test.tokens.tfidf)] <- 0.0
summary(art_test.tokens.tfidf[1,])

# With the art_test data projected into the TF-IDF vector space of the training
# data we can now to the final projection into the training LSA semantic
# space (i.e. the SVD matrix factorization).
art_test.svd.raw <- t(sigma.inverse * u.transpose %*% t(art_test.tokens.tfidf))


# Lastly, we can now build the art_test data frame to feed into our trained
# machine learning model for predictions. First up, add Label and TextLength.
art_test.irlba <- irlba(t(art_test.tokens.tfidf), nv = 300, maxit = 600)

#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
art_test.svd <- data.frame(Sentiment = category_senti2.art_test.df$category_senti_test, art_test.svd.raw)
#
# NOTE - The following code was updated post-video recoding due to a bug.
#
compareTable <- table(category_senti2.art_test.df$category_senti_test, predict(rpart.cv.4,art_test.svd))
misClassify <- mean(category_senti2.art_test.df$category_senti_test != predict(rpart.cv.4,art_test.svd))
print(paste("Accuracy of RPART is",1-misClassify))

compareTable_svm <- table(category_senti2.art_test.df$category_senti_test, predict(svmfit_art,category_senti2.art_test))
misClassify_svm <- mean(category_senti2.art_test.df$category_senti_test != predict(svmfit_art,art_test.svd))
print(paste("Accuracy of SVM Kernel is",1-misClassify_svm))

compareTable_rf <- table(category_senti2.art_test.df$category_senti_test, predict(rf_art,art_test.svd))
misClassify_rf <- mean(category_senti2.art_test.df$category_senti_test != predict(rf_art,art_test.svd))
print(paste("Accuracy of RF is",1-misClassify_rf))

compareTable_nb <- table(category_senti2.art_test.df$category_senti_test, predict(nb_art,art_test.svd))
misClassify_nb <- mean(category_senti2.art_test.df$category_senti_test != predict(nb_art,art_test.svd))
print(paste("Accuracy of NB is",1-misClassify_nb))

compareTable_knn <- table(category_senti2.art_test.df$category_senti_test, predict(rpart.cv.5,art_test.svd))
misClassify_knn <- mean(category_senti2.art_test.df$category_senti_test != predict(rpart.cv.5,art_test.svd))
print(paste("Accuracy of NB is",1-misClassify_knn))

compareTable_nnet <- table(category_senti2.art_test.df$category_senti_test, predict(rpart.cv.6,art_test.svd))
misClassify_nnet <- mean(category_senti2.art_test.df$category_senti_test != predict(rpart.cv.6,art_test.svd))
print(paste("Accuracy of NB is",1-misClassify_nnet))
