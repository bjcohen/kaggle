library(lda)
library(kernlab)
library(MASS)
library(ggplot2)
library(randomForest)
library(tm)
library(penalized)
library(topicmodels)
library(Matrix)
library(glmnet)

# read in data
location.tree <- read.csv('Location_Tree.csv')
data.train <- read.csv('Train.csv') # Id, Title, FullDescription, LocationRaw, LocationNormalized, ContractType, ContractTime, Company, Category, SalaryRaw, SalaryNormalized, SourceName
data.valid <- read.csv('Valid.csv')

# read in corpus
fullDescription.corpus = c();
fullDescription.corpus$documents <- read.documents("FullDescription_documents.dat")
fullDescription.corpus$vocab <- read.vocab("FullDescription_vocab.dat")

title.corpus = c();
title.corpus$documents <- read.documents("Title_documents.dat")
title.corpus$vocab <- read.vocab("Title_vocab.dat")

# lda/slda models
num.topics <- 25
fd.lda.model <- lda.collapsed.gibbs.sampler(fullDescription.corpus$documents, num.topics, fullDescription.corpus$vocab, num.iterations=10, alpha=0.1, eta=0.1)
title.lda.model <- lda.collapsed.gibbs.sampler(title.corpus$documents, num.topics, title.corpus$vocab, num.iterations=10, alpha=0.1, eta=0.1)

## slda.params <- sample(c(-1, 1), num.topics, replace=T)
## log.salary <- log(data.train$SalaryNormalized)

## fd.slda.model <- slda.em(fullDescription.corpus$documents, num.topics, fullDescription.corpus$vocab, num.e.iterations=10, num.m.iterations=4, alpha=0.1, eta=0.1, annotations=log.salary, params=slda.params, variance=.30, logistic=F, lambda=10, regularise=T)
## title.slda.model <- slda.em(title.corpus$documents, num.topics, title.corpus$vocab, num.e.iterations=10, num.m.iterations=4, alpha=0.1, eta=0.1, annotations=log.salary, params=slda.params, variance=.30, logistic=F, lambda=10, regularise=T)

# Plot coefficients by topic
# slda.topics <- apply(top.topic.words(slda.model$topics, 5, by.score=T), 2, paste, collapse=" ")
# slda.coefs <- data.frame(coef=coef(slda.model$model))
# theme_set(theme_bw())
# slda.coefs <- cbind(slda.coefs, topic=factor(slda.topics))
# slda.coefs <- slda.coefs[order(slda.coefs),]
# qplot(topic, coef, size=coef, data=slda.coefs) + coord_flip()

# get predicted docsums
fd.topics <- t(fd.lda.model$document_sums)
colnames(fd.topics) <- apply(top.topic.words(fd.lda.model$topics, 5, by.score=T), 2, paste, collapse=" ")
title.topics <- t(title.lda.model$document_sums)
colnames(title.topics) <- apply(top.topic.words(title.lda.model$topics, 5, by.score=T), 2, paste, collapse=" ")


## slda.predicted.docsums <- slda.predict.docsums(corpus$documents, slda.model$topics, alpha=.1, eta=.1, num.iterations=10, average.iterations=10, trace=100L)
## slda.topics <- t(slda.predicted.docsums)
## colnames(slda.topics) <- apply(top.topic.words(slda.model$topics, 5, by.score=T), 2, paste, collapse=" ")

# create model matrix
# benchmark was on fulldesc, title, locraw, locnorm
modelmat <- sparse.model.matrix(SalaryNormalized~-1+ContractType+ContractTime+Category+LocationNormalized, data.train)
modelmat.topics <- cBind(modelmat[,colnames(modelmat) != "(Intercept)"], fd.topics, title.topics)

# pick training sample
N <- dim(modelmat.topics)[[1]]
n <- 1000;
use.rows <- sample(N, n, replace=F)
SalaryNormalized <- data.train$SalaryNormalized

# svr model
svr.model <- ksvm(modelmat.topics[use.rows], SalaryNormalized[use.rows], kernel="anovadot", type="eps-svr", kpar=list(sigma=1), C=.1, epsilon=.1, cache=1024, cross=5)
train.pred <- predict(svr.model, modelmat.topics[use.rows])
svr.mae <- mean(abs(train.pred - SalaryNormalized[use.rows]))

# ols model
## lm.model <- lm(SalaryNormalized ~ ., data=modelmat.topics)
## lm.mae <- mean(abs(predict(lm.model, as.data.frame(modelmat.topics)) - SalaryNormalized[use.rows]))

# glm model
ridge.model <- glmnet(modelmat.topics, SalaryNormalized, family="gaussian", rep(1, dim(modelmat.topics)[1]), alpha=0)
ridge.pred <- predict(ridge.model, modelmat.topics)
ridge.mae <- mean(abs(ridge.pred - SalaryNormalized))

lasso.model <- glmnet(modelmat.topics, SalaryNormalized, family="gaussian", rep(1, dim(modelmat.topics)[1]), alpha=1)
lasso.pred <- predict(lasso.model, modelmat.topics)
lasso.mae <- mean(abs(lasso.pred - SalaryNormalized))

lasso.rest.model <- glmnet(modelmat.topics, SalaryNormalized, family="gaussian", rep(1, dim(modelmat.topics)[1]), alpha=1, pmax=1000)
lasso.rest.pred <- predict(lasso.rest.model, modelmat.topics)
lasso.rest.mae <- mean(abs(lasso.rest.pred - SalaryNormalized))

# ridge regression model
lm.ridge.model <- lm.ridge(SalaryNormalized ~ ., data=modelmat.topics, lambda=seq(141, 182, 1)) ## remember to adjust ridge coefficients
lm.ridge.pred <- scale(modelmat.topics, center=T, scale=lm.ridge.model$scales) %*% lm.ridge.model$coef[,which.min(lm.ridge.model$GCV)] + lm.ridge.model$ym
lm.ridge.mae <- mean(abs(lm.ridge.pred - SalaryNormalized))

# random forest
rf.model <- randomForest(modelmat.topics, SalaryNormalized, subset=use.rows, ntree=50, importance=T, proximity=T, mtry=30)
rf.pred <- predict(rf.model, modelmat.topics)
rf.mae <- mean(abs(rf.pred - SalaryNormalized))
