library(lda) ## http://cran.r-project.org/web/packages/lda/lda.pdf
library(Metrics)
library(ggplot2)

review.data = read.csv("yelp_training_set_review.csv", stringsAsFactors = FALSE)

tfidf = function(corpus, use.tf = TRUE, use.idf = TRUE, max.df = 1) {
  corp.idf = idf(corpus, vocab = NULL)
  if (use.idf) {
    corpus = lapply(corpus, function(x) { c = corp.idf[as.character(x[1,])]; x[2,] = x[2,] * c; x })
  }
  if(use.tf) {
    corpus = lapply(corpus, function(x) { x[2,] = x[2,] / sum(x[2,]); x })
  } 
  lapply(corpus, function(x) { c = corp.idf[as.character(x[1,])]; x[,(1/c) < max.df] })
}

idf = function(corpus, vocab = NULL) {
  D = length(corpus)
  all.words = matrix(unlist(corpus), nrow = 2)
  all.words[2,] = 1
  if (is.null(vocab)) {
    result = tapply(all.words[2,], all.words[1,], sum)
  }
  else {
    result = tapply(all.words[2,], list(word = ordered(vocab[all.words[1,] + 1], levels = vocab)), sum)
    result[is.na(result)] <- 0
  }
  ((D+1) / (result+1))
}

## corpus = lexicalize(review.data$text)

## docs = tfidf(corpus$documents, use.tf = FALSE, use.idf = FALSE, max.df = 1.0)
## docs = filter.words(corpus$documents, as.numeric(names(wc)[wc <= 2]))
## wc = word.counts(corpus$documents)
docs = read.documents('text.documents')
vocab = read.vocab('text.vocab')

num.topics = 100
set.seed(1)
params = sample(c(-1, 1), num.topics, replace=TRUE)

filter.out = which(sapply(docs, length) == 0)

slda.1 = slda.em(documents=docs[-filter.out], K=num.topics, vocab=vocab,
  num.e.iterations=10, num.m.iterations=4,
  alpha=1.0, eta=0.1,
  annotations=review.data$votes_useful[-filter.out] > 0,
  params=params,
  variance=0.25, lambda=1.0,
  logistic=FALSE, regularise=TRUE,
  method="sLDA", trace=100L)

Topics = apply(top.topic.words(slda.1$topics, 5, by.score=TRUE), 2, paste, collapse=" ")
coefs = data.frame(coef(summary(slda.1$model)))
coefs = cbind(coefs, Topics=factor(Topics, Topics[order(coefs$Estimate)]))
coefs = coefs[order(coefs$Estimate),]

qplot(Topics, Estimate, colour=Estimate, size=abs(z.value), data=coefs) +
  geom_errorbar(width=0.5, aes(ymin=Estimate-Std..Error,
                  ymax=Estimate+Std..Error)) + coord_flip()

predictions = slda.predict(docs, slda.1$topics, slda.1$model, alpha = 1.0, eta=0.1)

predictions.1 = predictions
predictions.1[predictions.1 < 0] = 0
rmsle(predictions.1, review.data$votes_useful[-filter.out])

qplot(predictions, fill=factor(review.data$votes_useful[-filter.out]),
      xlab = "predicted rating", ylab = "density",
      alpha=I(0.5), geom="density") +
  geom_vline(aes(xintercept=0)) # + opts(legend.position = "none")

predicted.docsums = slda.predict.docsums(docs, slda.1$topics, alpha = 1.0, eta=0.1)
predicted.proportions = t(predicted.docsums) / colSums(predicted.docsums)

qplot(`Topic 1`, `Topic 2`, data = structure(data.frame(predicted.proportions), names = paste("Topic", 1:10)), 
      size = `Topic 3`)

