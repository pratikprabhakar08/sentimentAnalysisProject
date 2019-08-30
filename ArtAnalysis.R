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
art <- train[tweets.df$Category %in% c("Art"),]

#########################################################
art.tweet <- replace_contraction(art$Text)

art.tweet = gsub("RT((?:\\b\\W*@\\w+)+)","", art.tweet)
art.tweet = gsub("http[^[:blank:]]+","",art.tweet) # removes http://
art.tweet = gsub("@\\w+","",art.tweet)
art.tweet = gsub("#\\w+","",art.tweet)
art.tweet = gsub('[[:punct:]]', ' ', art.tweet) # removes punctuation 
art.tweet = gsub('[^[:alnum:]]', ' ', art.tweet)


df <- extract_sentiment_terms(art.tweet)
art.tokens <- tokens(art.tweet, what = "word", 
                     remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE, remove_hyphens = TRUE)


# Lower case the tokens.
art.tokens <- tokens_tolower(art.tokens)
art.tokens[[9]]


# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to
#        your problem/domain.
art.tokens <- tokens_select(art.tokens, stopwords(), 
                            selection = "remove")
art.tokens[[9]]


# Perform stemming on the tokens.
art.tokens <- tokens_wordstem(art.tokens, language = "english")
art.tokens[[9]]

word.list_art = str_split(art.tokens, '\\s+') # splits the tweets by word in a list

words_art = unlist(word.list_art) # turns the list into vector

word.df_art <- as.vector(art.tweet)

sentiment_art <- get_sentiment(word.df_art)
most.positive_art <- word.df_art[sentiment_art == max(sentiment_art)]

most.positive_art
most.negative_art <- word.df_art[sentiment_art <= min(sentiment_art)]
most.negative_art 

positive.tweets_art <- word.df_art[sentiment_art > 0]

head(positive.tweets_art)

negative.tweets_art <- word.df_art[sentiment_art < 0]
head(negative.tweets_art)

neutral.tweets_art <- word.df_art[sentiment_art == 0] 
head(neutral.tweets_art)

category_senti_art <- ifelse(sentiment_art < 0, "Negative", ifelse(sentiment_art > 0, "Positive", "Neutral"))

head(category_senti_art)

category_senti2_art <- cbind(art.tweet,category_senti_art,sentiment_art) 

head(category_senti2_art)

incomplete.cases_art <- which(!complete.cases(category_senti2_art))
art$Text[incomplete.cases_art]


# Fix incomplete cases
category_senti2_art[incomplete.cases_art,] <- rep(0.0, ncol(category_senti2_art))
dim(category_senti2_art)
sum(which(!complete.cases(category_senti2_art)))
nrc<- get_sentiment(method="nrc", language="english")

# Make a clean data frame using the same process as before.
category_senti2.df_art <- cbind(Category = art$Category, Project= art$Project, data.frame(category_senti2_art))
names(category_senti2.df_art) <- make.names(names(category_senti2.df_art))

sum_senti_art <- summary(category_senti2_art)
write.csv(category_senti2.df_art, 'E:\\Masters\\Project\\Kickstarter\\Data\\CategoryWiseAnalyse\\Sentiment_scores_art.csv', row.names = FALSE)

qplot(category_senti2.df_art$category_senti_art, data=category_senti2.df_art, fill=Category)+
  ggtitle("Distribution of Sentiments of Tweets across Art")
qplot(category_senti2.df_art$category_senti_art, data=category_senti2.df_art, fill=Project)+
  ggtitle("Distribution of Sentiments of Tweets across Projects of Art")

#############################################################
##############################################################################
# Create our first bag-of-words model.
art.tokens.dfm <- dfm(art.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
art.tokens.matrix <- as.matrix(art.tokens.dfm)
#View(train.tokens.matrix[1:20, 1:100])
dim(art.tokens.matrix)

# Often, tokenization requires some additional pre-processing
names(art.tokens.df)[c(146, 148, 235, 238)]

# Cleanup column names.
names(art.tokens.df) <- make.names(names(art.tokens.df))

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
art.tokens.df <- apply(art.tokens.matrix, 1, term.frequency)
dim(art.tokens.df)

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
art.tokens.idf <- apply(art.tokens.matrix, 2, inverse.doc.freq)
str(art.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
art.tokens.tfidf <-  apply(art.tokens.df, 2, tf.idf, idf = art.tokens.idf)
dim(art.tokens.tfidf)


# Transpose the matrix
art.tokens.tfidf <- t(art.tokens.tfidf)
dim(art.tokens.tfidf)


# Check for incopmlete cases.
incomplete.cases_art <- which(!complete.cases(art.tokens.tfidf))
art$Text[incomplete.cases_art]


# Fix incomplete cases
art.tokens.tfidf[incomplete.cases_art,] <- rep(0.0, ncol(art.tokens.tfidf))
dim(art.tokens.tfidf)
sum(which(!complete.cases(art.tokens.tfidf)))


# Make a clean data frame using the same process as before.
art.tokens.tfidf.df <- cbind(Sentiment = category_senti2.df_art$category_senti_art, data.frame(art.tokens.tfidf))
names(art.tokens.tfidf.df) <- make.names(names(art.tokens.tfidf.df))


library(doSNOW)
set.seed(48743)
cv.folds <- createMultiFolds(category_senti2.df_art$category_senti_art, k = 10, times = 3)

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
rpart.cv.2 <- train(Sentiment ~ ., data = art.tokens.tfidf.df, method = "rpart", 
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
art.irlba <- irlba(t(art.tokens.tfidf), nv = 300, maxit = 600)

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
sigma.inverse <- 1 / art.irlba$d
u.transpose <- t(art.irlba$u)
document <- art.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
art.irlba$v[1, 1:10]



#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
art.svd <- data.frame(Sentiment = category_senti2.df_art$category_senti_art, art.irlba$v)

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.4 <- train(Sentiment ~ ., data = art.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.4

##########################################################################################
svmfit_art <- svm(Sentiment~., data=category_senti2.df_art, kernel="radial")
print(svmfit_art)

library(randomForest)
rf_art <- randomForest(Sentiment~., data=art.svd)

nb_art <- naiveBayes(Sentiment ~., data= art.svd)

rf_art_imp <- randomForest(Sentiment~., data=art.tokens.tfidf.df)
varImpPlot(rf_art_imp)

art_glm <-  glm(Sentiment ~ ., data=art.tokens.tfidf.df)
importance(rf_art_imp)
########################################################

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

# This will be the last run using single decision trees. With a much smaller
# feature matrix we can now use more powerful methods like the mighty Random
# Forest from now on!
rpart.cv.5 <- train(Sentiment ~ ., data = art.svd, method = "knn", 
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
rpart.cv.6 <- train(Sentiment ~ ., data = art.svd, method = "nnet", 
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
