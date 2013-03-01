library(lda)
library(kernlab)
library(MASS)
library(ggplot2)
library(randomForest)
library(tm)
library(penalized)

# read in data
location.tree <- read.csv('Location_Tree.csv')
data.train <- read.csv('Train.csv') # Id, Title, FullDescription, LocationRaw, LicationNormalized, ContractType, ContractTime, Company, Category, SalaryRaw, SalaryNormalized, SourceName
data.valid <- read.csv('Valid.csv')

# read in corpus
corpus = c();
corpus$documents <- read.documents("documents.dat")
corpus$vocab <- read.vocab("vocab.dat")

# lda/slda models
num.topics <- 25
model <- lda.collapsed.gibbs.sampler(corpus$documents, num.topics, corpus$vocab, num.iterations=10, alpha=0.1, eta=0.1)
slda.params <- sample(c(-1, 1), num.topics, replace=T)
log.salary <- log(SalaryNormalized)
slda.model <- slda.em(corpus$documents, num.topics, corpus$vocab, num.e.iterations=10, num.m.iterations=4, alpha=0.1, eta=0.1, annotations=log.salary, params=slda.params, variance=.25, logistic=F, lambda=10, regularise=T)

# plot coefficients by topic
slda.topics <- apply(top.topic.words(slda.model$topics, 5, by.score=T), 2, paste, collapse=" ")
slda.coefs <- data.frame(coef=coef(slda.model$model))
theme_set(theme_bw())
slda.coefs <- cbind(slda.coefs, topic=factor(slda.topics))
# slda.coefs <- slda.coefs[order(slda.coefs),]
qplot(topic, coef, size=coef, data=slda.coefs) + coord_flip()

# plot rating density
# slda.predictions <- slda.predict(corpus$documents, slda.model$topics, slda.model$model, alpha=.1, eta=.1)
# qplot(slda.predictions, fill=factor(poliblog.ratings), xlab = "predicted rating", ylab = "density", alpha=I(0.5), geom="density") + geom_vline(aes(xintercept=0))

# get predicted docsums
document.topics <- t(model$document_sums)
colnames(document.topics) <- apply(top.topic.words(model$topics, 5, by.score=T), 2, paste, collapse=" ")

slda.predicted.docsums <- slda.predict.docsums(corpus$documents, slda.model$topics, alpha=.1, eta=.1, num.iterations=10, average.iterations=10, trace=100L)
slda.topics <- t(slda.predicted.docsums)
colnames(slda.topics) <- apply(top.topic.words(slda.model$topics, 5, by.score=T), 2, paste, collapse=" ")

# create model matrix
## title.bow <- TermDocumentMatrix(Corpus(VectorSource(data.train$Title)), control=c(removePunctuation=T, removeNumbers=T, stopwords=T, stemming=T))
## location.raw.bow
## location.norm.bow
## company
modelmat <- model.matrix(SalaryNormalized~ContractType+ContractTime+Category, data.train)
modelmat.topics <- cbind(modelmat[,colnames(modelmat) != "(Intercept)"], slda.topics)

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
lm.model <- lm(SalaryNormalized ~ ., data=as.data.frame(cbind(modelmat.topics, SalaryNormalized)))
lm.mae <- mean(abs(predict(lm.model, as.data.frame(modelmat.topics)) - SalaryNormalized[use.rows]))

# ridge regression model
lm.ridge.model <- lm.ridge(SalaryNormalized ~ ., data=as.data.frame(cbind(modelmat.topics, SalaryNormalized)), lambda=seq(141, 182, 1))
lm.ridge.pred <- scale(modelmat.topics, center=T, scale=lm.ridge.model$scales) %*% lm.ridge.model$coef[,which.min(lm.ridge.model$GCV)] + lm.ridge.model$ym
lm.ridge.mae <- mean(abs(lm.ridge.pred - SalaryNormalized))

# random forest
rf.model <- randomForest(modelmat.topics[use.rows,], y=as.matrix(SalaryNormalized[use.rows]), importance=T, proximity=T)
rf.pred <- predict(rf.model, modelmat.topics)
rf.mae <- mean(abs(rf.pred - SalaryNormalized))
