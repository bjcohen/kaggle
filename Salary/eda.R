library(lda)

location.tree <- read.csv('Location_Tree.csv')
data.train <- read.csv('Train.csv', nrows=1000)

corpus <- lexicalize(data.train$FullDescription)

num.topics <- 50
lda.collapsed.gibbs.sampler(corpus$documents, num.topics, corpus$vocab, num.iterations=10, alpha=, eta=)
