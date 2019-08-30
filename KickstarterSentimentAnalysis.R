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
library(sentimentr)

setwd("E:/Masters/Project/Kickstarter/Data")

tweets.df <- read.csv('DesignTech_with_ProjectName.csv', header=TRUE)

tweets.df$FavoriteCount <- as.integer(tweets.df$FavoriteCount)
tweets.df$RetweetCount <- as.integer(tweets.df$RetweetCount)
tweets.df$Source <- as.String(tweets.df$Source)
tweets.df$Category <- as.factor(tweets.df$Category)
tweets.df$Project <- as.factor(tweets.df$Project)

str(tweets.df)
pos.df <- scan('E:\\Masters\\Project\\Kickstarter\\positive-words.csv',what = 'character')
neg.df <- scan('E:\\Masters\\Project\\Kickstarter\\negative-words.csv',what = 'character')

set.seed(32984)
indexes <- createDataPartition(tweets.df$Category, times = 1,
                               p = 0.7, list = FALSE)

train <- tweets.df[indexes,]
test <- tweets.df[-indexes,]

prop.table(table(tweets.df$Category))
prop.table(table(test$Category))

levels(train$Category)
qplot(tweets.df$Category, data=tweets.df, fill=Category)+
  ggtitle("Distribution of Number of Tweets across various Categories")

qplot(tweets.df$FavoriteCount, tweets.df$RetweetCount, data=tweets.df, fill=Category)+
  ggtitle("Distribution of Favorite Count and Retweet Count across various Categories")

#########################################################

train.tweet <- replace_contraction(train$Text)

# removes RT tokens 
train.tweet = gsub("RT((?:\\b\\W*@\\w+)+)","", train.tweet) 
# removes http://
train.tweet = gsub("http[^[:blank:]]+","",train.tweet) 
# removes UserID followed by words 
train.tweet = gsub("@\\w+","",train.tweet) 
# removes hash-tag followed by words
train.tweet = gsub("#\\w+","",train.tweet) 
# removes punctuation 
train.tweet = replace_emoticon(train.tweet)

train.tweet = gsub('[[:punct:]]', ' ', train.tweet) 
# removes numbers with words
train.tweet = gsub('[^[:alnum:]]', ' ', train.tweet) 

pos_tag<-pos(train.tweet)
tagged_pos <- pos_tag$POStagged
p_prop <- pos_tag$POSrnp

train.tokens <- tokens(train.tweet, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

# Take a look at a specific SMS message and see how it transforms.
train.tokens[[15]]


# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[15]]

# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to
#        your problem/domain.
train.tokens <- tokens_select(train.tokens, c(stopwords(),"kickstarter"), 
                              selection = "remove")
train.tokens[[4]]

# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[4]]


#########################################################################
################################ Polarity of tweets ##########################################
tweets.polarity <- polarity(train.tweet)
tweets_each_polarity <- tweets.polarity$all

polarity <- ifelse(tweets_each_polarity$polarity < 0, "Negative", ifelse(tweets_each_polarity$polarity > 0.15, "Positive", "Neutral"))

#qplot(polarity)
tweet.new_df <- cbind(Favorites = train$FavoriteCount, Retweets = train$RetweetCount, 
                      Project = train$Project, 
                      Category = train$Category, Score= tweets_each_polarity$polarity, 
                      Text = tweets_each_polarity$text.var, data.frame(polarity))


qplot(tweet.new_df$polarity, data=tweet.new_df, fill=Category)+
  ggtitle("Distribution of Sentiments of Tweets across various Categories using Polarity")

table(tweet.new_df$Category, tweet.new_df$polarity)

ggplot(tweet.new_df, aes(x=Score)) + 
  geom_histogram(binwidth=1) + 
  xlab("Sentiment score") + 
  ylab("Frequency") + 
  ggtitle("Distribution of sentiment score") +
  theme_bw()  + 
  theme(axis.title.x = element_text(vjust = -0.5, size = 14)) + 
  theme(axis.title.y=element_text(size = 14, angle=90, vjust = -0.25)) + 
  theme(plot.margin = unit(c(1,1,2,2), "lines"))

sum(is.na(tweet.new_df))
####################################################################################

score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    sentence = gsub(c(stopwords(), "kickstarter"),' ', sentence)
    sentence = tolower(sentence)
    
    # split into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    words = unlist(word.list)
    
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}

result <- score.sentiment(train.tweet,pos.df,neg.df)
#result$score
sentiment_cat <- ifelse(result$score < 0, "Negative", ifelse(result$score > 0, "Positive", "Neutral"))

sentiment.df <- cbind(Category=train$Category, Project=train$Project, Text=result$text, score=result$score, data.frame(sentiment_cat))

qplot(sentiment.df$sentiment_cat, data=sentiment.df, fill=Category)+
  ggtitle("Distribution of Sentiments of Tweets across various Categories using BING")


table(sentiment.df$Category, sentiment.df$sentiment_cat)

ggplot(sentiment.df, aes(x=score)) + 
  geom_histogram(binwidth=1) + 
  xlab("Sentiment score") + 
  ylab("Frequency") + 
  ggtitle("Distribution of sentiment score") +
  theme_bw()  + 
  theme(axis.title.x = element_text(vjust = -0.5, size = 14)) + 
  theme(axis.title.y=element_text(size = 14, angle=90, vjust = -0.25)) + 
  theme(plot.margin = unit(c(1,1,2,2), "lines"))

qplot(sentiment.df$sentiment_cat, xlab = "Sentiments", ylab="Counts")
###################################################################################
word.list = str_split(train.tokens, '\\s+') # splits the tweets by word in a list

words = unlist(word.list) # turns the list into vector

word.df <- as.vector(train.tweet)

sentiment_afinn <- get_sentiment(word.df)
most.positive_afinn <- word.df[sentiment_afinn == max(sentiment_afinn)]

most.positive_afinn
most.negative_afinn <- word.df[sentiment_afinn <= min(sentiment_afinn)]
most.negative_afinn 

positive.tweets_afinn <- word.df[sentiment_afinn > 0]

head(positive.tweets_afinn)

negative.tweets_afinn <- word.df[sentiment_afinn < 0]
head(negative.tweets_afinn)

neutral.tweets_afinn <- word.df[sentiment_afinn == 0] 
head(neutral.tweets_afinn)

category_senti_afinn <- ifelse(sentiment_afinn < -0.5, "Negative", ifelse(sentiment_afinn > 0.49, "Positive", "Neutral"))

head(category_senti_afinn)

category_senti2_afinn <- cbind(Text=train.tweet,Category=train$Category,Sentiment = category_senti_afinn,score=sentiment_afinn) 

head(category_senti2_afinn)

incomplete.cases <- which(!complete.cases(category_senti2_afinn))
train.tweet[incomplete.cases]


# Fix incomplete cases
category_senti2_afinn[incomplete.cases,] <- rep(0.0, ncol(category_senti2_afinn))
dim(category_senti2_afinn)
sum(which(!complete.cases(category_senti2_afinn)))


# Make a clean data frame using the same process as before.
category_senti2.df_afinn <- cbind(Category = train$Category, Project= train$Project, data.frame(category_senti2_afinn))
names(category_senti2.df_afinn) <- make.names(names(category_senti2.df_afinn))

sum_senti_afinn <- summary(category_senti2_afinn)

qplot(category_senti2.df_afinn$Sentiment, data=category_senti2.df_afinn, fill=Category)+
  ggtitle("Distribution of Sentiments of Tweets across Categories")

table(category_senti2.df_afinn$Category, category_senti2.df_afinn$Sentiment)

qplot(category_senti2.df_afinn$Sentiment, xlab="Sentiment", ylab="Count")

ggplot(sentiment.df, aes(sentiment.df$sentiment_cat, sentiment.df$score)) +
  geom_point()

#############################################################

nrc_sentiment <- get_nrc_sentiment(train.tweet)
Sentiment_Scores <- data.frame(colSums(nrc_sentiment[,]))
names(Sentiment_Scores) <- "Score"

Sentiment_Scores <- cbind('Sentiment'= rownames(Sentiment_Scores), Sentiment_Scores)
rownames(Sentiment_Scores) <- NULL

qplot(Sentiment, data=Sentiment_Scores, weight=Score, geom="bar")+
  ggtitle("Total Sentiment Score Based on Tweets")
##############################################################################
# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
#View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)

# Often, tokenization requires some additional pre-processing
names(train.tokens.dfm)[c(146, 148, 235, 238)]

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
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)


# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)


# Check for incopmlete cases.
incomplete.cases_train <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases_train]


# Fix incomplete cases
train.tokens.tfidf[incomplete.cases_train,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

########################################################################
train.tokens.tfidf_polarity.df <- cbind(Sentiment = tweet.new_df$polarity,data.frame(train.tokens.tfidf))
names(train.tokens.tfidf_polarity.df) <- make.names(names(train.tokens.tfidf_polarity.df))

train.tokens.tfidf_user.df <- cbind(Sentiment = sentiment.df$sentiment_cat,data.frame(train.tokens.tfidf))
names(train.tokens.tfidf_user.df) <- make.names(names(train.tokens.tfidf_user.df))


train.tokens.tfidf_sentiment.df <- cbind(Sentiment = category_senti2.df_afinn$Sentiment,data.frame(train.tokens.tfidf))
names(train.tokens.tfidf_sentiment.df) <- make.names(names(train.tokens.tfidf_sentiment.df))

################################################################################################
library(doSNOW)
set.seed(48743)
cv.folds <- createMultiFolds(tweet.new_df$polarity, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)

# Time the code execution
start.time <- Sys.time()

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

rpart.cv.2 <- train(Sentiment ~ ., data = train.tokens.tfidf_user.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.2

knn_fit <- knn3(Sentiment ~ ., data= train.tokens.tfidf_user.df)

gc()

library(irlba)


# Time the code execution
start.time <- Sys.time()

# Perform SVD. Specifically, reduce dimensionality down to 300 columns
# for our latent semantic analysis (LSA).
train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time


#lsa_model <- textmodel_lsa(train.tokens.dfm)
#length(lsa_model$docs)

# Take a look at the new feature data up close.
#View(train.irlba$v)


# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
#
#
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
train.irlba$v[1, 1:10]



#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
train.svd <- data.frame(Sentiment = train.tokens.tfidf_polarity.df$Sentiment, train.irlba$v)

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

rpart.cv.4 <- train(Sentiment ~ ., data = train.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.4

gc()

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()

rpart.cv.knn <- train(Sentiment ~ ., data = train.svd, method = "knn", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.
rpart.cv.knn

imp1 <- varImp(rpart.cv.2)
imp1

imp2 <- varImp(rpart.cv.knn)
plot(imp2)
##########################################################################################
svmfit <- svm(Sentiment~., data=train.svd, kernel="radial")
print(svmfit)

library(randomForest)
rf <- randomForest(Sentiment~., data=train.svd)
varImpPlot(rf)

nb <- naiveBayes(Sentiment ~., data= train.svd)

rf_art_imp <- randomForest(Sentiment~., data=art.tokens.tfidf.df)

art_glm <-  glm(Sentiment ~ ., data=art.tokens.tfidf.df)
importance(rf_art_imp)
########################################################

confusionMatrix(train.tokens.tfidf_user.df$Sentiment, tweet.new_df$polarity)
confusionMatrix(train.tokens.tfidf_sentiment.df$Sentiment, tweet.new_df$polarity)
###################### Testing the model #############################################

test.tweet <- replace_contraction(test$Text)

test.tweet = gsub("RT((?:\\b\\W*@\\w+)+)","", test.tweet)
test.tweet = gsub("http[^[:blank:]]+","",test.tweet) # removes http://
test.tweet = gsub("@\\w+","",test.tweet)
test.tweet = gsub("#\\w+","",test.tweet)
test.tweet = gsub('[[:punct:]]', ' ', test.tweet) # removes punctuation 
test.tweet = gsub('[^[:alnum:]]', ' ', test.tweet)

test.tokens <- tokens(test.tweet, what = "word", 
                          remove_numbers = TRUE, remove_punct = TRUE,
                          remove_symbols = TRUE, remove_hyphens = TRUE)

# Lower case the tokens.
test.tokens <- tokens_tolower(test.tokens)

# Stopword removal.
test.tokens <- tokens_select(test.tokens, stopwords(), 
                                 selection = "remove")

# Stemming.
test.tokens <- tokens_wordstem(test.tokens, language = "english")
#############################################################################

test.polarity <- polarity(test.tweet)
test_each_polarity <- test.polarity$all

test_polarity <- ifelse(test_each_polarity$polarity < 0, "Negative", ifelse(test_each_polarity$polarity > 0, "Positive", "Neutral"))

test.new_df <- cbind(Favorites = test$FavoriteCount, Retweets = test$RetweetCount, 
                      Project = test$Project, 
                      Category = test$Category, Score= test_each_polarity$polarity, 
                      Text = test_each_polarity$text.var, data.frame(test_polarity))


ggplot(test.new_df, aes(x=Score)) + 
  geom_histogram(binwidth=1) + 
  xlab("Sentiment score") + 
  ylab("Frequency") + 
  ggtitle("Distribution of sentiment score") +
  theme_bw()  + 
  theme(axis.title.x = element_text(vjust = -0.5, size = 14)) + 
  theme(axis.title.y=element_text(size = 14, angle=90, vjust = -0.25)) + 
  theme(plot.margin = unit(c(1,1,2,2), "lines"))

sum(is.na(test.new_df))


############################################################################


word.df.test <- as.vector(test.tweet)

sentiment_test <- get_sentiment(word.df.test)
most.positive_test <- word.df.test[sentiment_test == max(sentiment_test)]

most.positive_test
most.negative_test <- word.df.test[sentiment_test <= min(sentiment_test)]
most.negative_test

positive.tweets_test <- word.df.test[sentiment_test > 0]

head(positive.tweets_test)

negative.tweets_test <- word.df.test[sentiment_test < 0]
head(negative.tweets_test)

neutral.tweets_test <- word.df.test[sentiment_test == 0] 
head(neutral.tweets_test)

category_senti_test <- ifelse(sentiment_test < 0, "Negative", ifelse(sentiment_test > 0, "Positive", "Neutral"))

head(category_senti_test)

category_senti2.test <- cbind(test.tweet,category_senti_test,sentiment_test) 

head(category_senti2.test)

incomplete.cases_test <- which(!complete.cases(category_senti2.test))
test$Text[incomplete.cases_test]


# Fix incomplete cases
category_senti2.test[incomplete.cases_test,] <- rep(0.0, ncol(category_senti2.test))
dim(category_senti2.test)
sum(which(!complete.cases(category_senti2.test)))


# Make a clean data frame using the same process as before.
category_senti2.test.df <- cbind(Category = test$Category, Project= test$Project, data.frame(category_senti2.test))
names(category_senti2.test.df) <- make.names(names(category_senti2.test.df))

sum_senti <- summary(category_senti2.test)
################################################################################

result_test <- score.sentiment(test.tweet,pos.df,neg.df)
#result$score
sentiment_cat_test <- ifelse(result_test$score < 0, "Negative", ifelse(result_test$score > 0, "Positive", "Neutral"))

sentiment.df_test <- cbind(Category=test$Category, Project=test$Project, Text=result_test$text, score=result_test$score, data.frame(sentiment_cat_test))

qplot(sentiment.df_test$sentiment_cat_test, data=sentiment.df_test, fill=Category)+
  ggtitle("Distribution of Sentiments of Tweets across various Categories using BING")


table(sentiment.df_test$Category, sentiment.df_test$sentiment_cat)

################################################################################
# Convert n-grams to quanteda document-term frequency matrix.
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)


# Ensure the art_test dfm has the same n-grams as the training dfm.
#
# NOTE - In production we should expect that new text messages will 
#        contain n-grams that did not exist in the original training
#        data. As such, we need to strip those n-grams out.
#
test.tokens.dfm <- dfm_select(test.tokens.dfm, pattern = train.tokens.dfm,
                                  selection = "keep")
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.tokens.dfm




# With the raw test features in place next up is the projecting the term
# counts for the unigrams into the same TF-IDF vector space as our training
# data. The high level process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values

# Normalize all documents via TF.
test.tokens.df <- apply(test.tokens.matrix, 1, term.frequency)
str(test.tokens.df)

# Lastly, calculate TF-IDF for our training corpus.
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(test.tokens.tfidf)
View(test.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
test.tokens.tfidf <- t(test.tokens.tfidf)

# Fix incomplete cases
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
summary(test.tokens.tfidf[1,])

# With the test data projected into the TF-IDF vector space of the training
# data we can now to the final projection into the training LSA semantic
# space (i.e. the SVD matrix factorization).
test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))


# Lastly, we can now build the test data frame to feed into our trained
# machine learning model for predictions. First up, add Label and TextLength.
test.irlba <- irlba(t(test.tokens.tfidf), nv = 300, maxit = 600)

#
# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
#
test.svd <- data.frame(Sentiment = test.new_df$test_polarity, test.svd.raw)
#
# NOTE - The following code was updated post-video recoding due to a bug.
#
compareTable <- table(test.new_df$test_polarity, predict(rpart.cv.4,test.svd))
misClassify <- mean(test.new_df$test_polarity != predict(rpart.cv.4,test.svd))
print(paste("Accuracy of RPART is",1-misClassify))

compareTable_svm <- table(test.new_df$test_polarity, predict(svmfit,test.svd))
misClassify_svm <- mean(test.new_df$test_polarity != predict(svmfit,test.svd))
print(paste("Accuracy of SVM Kernel is",1-misClassify_svm))

compareTable_rf <- table(test.new_df$test_polarity, predict(rf,test.svd))
misClassify_rf <- mean(test.new_df$test_polarity != predict(rf,test.svd))
print(paste("Accuracy of RF is",1-misClassify_rf))

compareTable_nb <- table(test.new_df$test_polarity, predict(nb,test.svd))
misClassify_nb <- mean(test.new_df$test_polarity != predict(nb,test.svd))
print(paste("Accuracy of NB is",1-misClassify_nb))

compareTable_knn <- table(test.new_df$test_polarity, predict(rpart.cv.5,test.svd))
misClassify_knn <- mean(test.new_df$test_polarity != predict(rpart.cv.5,test.svd))
print(paste("Accuracy of NB is",1-misClassify_knn))

####################################################################