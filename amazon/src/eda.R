## overall mean: .942, top scores: ~0.92 auc
## ACTION(2)                 ACTION is 1 if the resource was approved, 0 if the resource was not
## RESOURCE(7518)            An ID for each resource
## MGR_ID(4243)              The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time
## ROLE_ROLLUP_1(128)        Company role grouping category id 1 (e.g. US Engineering)
## ROLE_ROLLUP_2(177)        Company role grouping category id 2 (e.g. US Retail)
## ROLE_DEPTNAME(449)        Company role department description (e.g. Retail)
## ROLE_TITLE(343)           Company role business title description (e.g. Senior Engineering Retail Manager)
## ROLE_FAMILY_DESC(2358)    Company role family extended description (e.g. Retail Manager, Software Engineering)
## ROLE_FAMILY(67)           Company role family description (e.g. Retail Manager)
## ROLE_CODE(343)            Company role code; this code is unique to each role (e.g. Manager)

library(verification)
library(plyr)
library(ggplot2)

library(randomForest)
## library(rpart)
## library(FSelector)
library(lme4)
## library(party)

## options(java.parameters = "-Xmx512m")
## library(RWeka)


###

train.data.all = read.csv('../data/train.csv')
test.data.all = read.csv('../data/test.csv')

train.data.all[-1] = as.data.frame(lapply(train.data.all[-1], factor))
test.data.all[-1] = as.data.frame(lapply(test.data.all[-1], factor))

###

graph.data = train.data.all
rollup.links = ddply(graph.data, .(ROLE_ROLLUP_1), function(x) ddply(x, .(ROLE_ROLLUP_2), nrow))
rollup.links = ddply(rollup.links, .(ROLE_ROLLUP_1), function(x) {x$c1 = nrow(x); x})
rollup.links = ddply(rollup.links, .(ROLE_ROLLUP_2), function(x) {x$c2 = nrow(x); x})

ggplot(rollup.links[rollup.links$c1 > 1 | rollup.links$c2 > 1,], aes(x=0, xend=1, y=ROLE_ROLLUP_1, yend=ROLE_ROLLUP_2)) +
  geom_segment() + geom_text(aes(label=ROLE_ROLLUP_1, x=0, y=ROLE_ROLLUP_1, size=5)) + geom_text(aes(label=ROLE_ROLLUP_2, x=1, y=ROLE_ROLLUP_2, size=5))

###

ggplot(train.data.all, aes(x=ROLE_CODE, fill=factor(ACTION))) + geom_bar(position="fill")

###

test.data = train.data.all[27000:32769,]
train.data = train.data.all[1:27000,]

RESOURCE.probs = dlply(train.data, .(RESOURCE), function(x) mean(x$ACTION))
MGR_ID.probs = dlply(train.data, .(MGR_ID), function(x) mean(x$ACTION))
ROLE_ROLLUP_1.probs = dlply(train.data, .(ROLE_ROLLUP_1), function(x) mean(x$ACTION))
ROLE_ROLLUP_2.probs = dlply(train.data, .(ROLE_ROLLUP_2), function(x) mean(x$ACTION))
ROLE_DEPTNAME.probs = dlply(train.data, .(ROLE_DEPTNAME), function(x) mean(x$ACTION))
ROLE_TITLE.probs = dlply(train.data, .(ROLE_TITLE), function(x) mean(x$ACTION))
ROLE_FAMILY_DESC.probs = dlply(train.data, .(ROLE_FAMILY_DESC), function(x) mean(x$ACTION))
ROLE_FAMILY.probs = dlply(train.data, .(ROLE_FAMILY), function(x) mean(x$ACTION))
ROLE_CODE.probs = dlply(train.data, .(ROLE_CODE), function(x) mean(x$ACTION))

model.mat = data.frame(ACTION = train.data$ACTION)
model.mat$RESOURCE.probs = unlist(lapply(train.data$RESOURCE, function(x) RESOURCE.probs[as.character(x)]))
model.mat$MGR_ID.probs = unlist(lapply(train.data$MGR_ID, function(x) MGR_ID.probs[as.character(x)]))
model.mat$ROLE_ROLLUP_1.probs = unlist(lapply(train.data$ROLE_ROLLUP_1, function(x) ROLE_ROLLUP_1.probs[as.character(x)]))
model.mat$ROLE_ROLLUP_2.probs = unlist(lapply(train.data$ROLE_ROLLUP_2, function(x) ROLE_ROLLUP_2.probs[as.character(x)]))
model.mat$ROLE_DEPTNAME.probs = unlist(lapply(train.data$ROLE_DEPTNAME, function(x) ROLE_DEPTNAME.probs[as.character(x)]))
model.mat$ROLE_TITLE.probs = unlist(lapply(train.data$ROLE_TITLE, function(x) ROLE_TITLE.probs[as.character(x)]))
model.mat$ROLE_FAMILY_DESC.probs = unlist(lapply(train.data$ROLE_FAMILY_DESC, function(x) ROLE_FAMILY_DESC.probs[as.character(x)]))
model.mat$ROLE_FAMILY.probs = unlist(lapply(train.data$ROLE_FAMILY, function(x) ROLE_FAMILY.probs[as.character(x)]))
#model.mat$ROLE_CODE.probs = unlist(lapply(train.data$ROLE_CODE, function(x) ROLE_CODE.probs[as.character(x)]))

model.mat.test = data.frame(ACTION=test.data$ACTION)
model.mat.test$RESOURCE.probs = unlist(sapply(test.data$RESOURCE,
  function(x) if (as.character(x) %in% names(RESOURCE.probs)) RESOURCE.probs[as.character(x)] else 0.94))
model.mat.test$MGR_ID.probs = unlist(sapply(test.data$MGR_ID,
  function(x) if (as.character(x) %in% names(MGR_ID.probs)) MGR_ID.probs[as.character(x)] else 0.94))
model.mat.test$ROLE_ROLLUP_1.probs = unlist(sapply(test.data$ROLE_ROLLUP_1,
  function(x) if (as.character(x) %in% names(ROLE_ROLLUP_1.probs)) ROLE_ROLLUP_1.probs[as.character(x)] else 0.94))
model.mat.test$ROLE_ROLLUP_2.probs = unlist(sapply(test.data$ROLE_ROLLUP_2,
  function(x) if (as.character(x) %in% names(ROLE_ROLLUP_2.probs)) ROLE_ROLLUP_2.probs[as.character(x)] else 0.94))
model.mat.test$ROLE_DEPTNAME.probs = unlist(sapply(test.data$ROLE_DEPTNAME,
  function(x) if (as.character(x) %in% names(ROLE_DEPTNAME.probs)) ROLE_DEPTNAME.probs[as.character(x)] else 0.94))
model.mat.test$ROLE_TITLE.probs = unlist(sapply(test.data$ROLE_TITLE,
  function(x) if (as.character(x) %in% names(ROLE_TITLE.probs)) ROLE_TITLE.probs[as.character(x)] else 0.94))
model.mat.test$ROLE_FAMILY_DESC.probs = unlist(sapply(test.data$ROLE_FAMILY_DESC,
  function(x) if (as.character(x) %in% names(ROLE_FAMILY_DESC.probs)) ROLE_FAMILY_DESC.probs[as.character(x)] else 0.94))
model.mat.test$ROLE_FAMILY.probs = unlist(sapply(test.data$ROLE_FAMILY,
  function(x) if (as.character(x) %in% names(ROLE_FAMILY.probs)) ROLE_FAMILY.probs[as.character(x)] else 0.94))
#model.mat.test$ROLE_CODE.probs = unlist(sapply(test.data$ROLE_CODE,
#  function(x) if (as.character(x) %in% names(ROLE_CODE.probs)) ROLE_CODE.probs[as.character(x)] else 0.94))

  
glm.1 = glm(factor(ACTION) ~ ., data = model.mat, family = "binomial")

glmnet.1 = glmnet(data.matrix(model.mat[,-1]), model.mat[,1], family = "binomial", intercept = TRUE)

## write.csv(data.frame(Id=model.mat.test$id, Action=predict(glm.1, model.mat.test, type = "response")), "rsrc_mgr_logreg.csv", row.names=FALSE, quote=FALSE)

roc.area(test.data$ACTION, predict(glm.1, model.mat.test[,-1]))

roc.area(test.data$ACTION, predict(glmnet.1, data.matrix(model.mat.test[,-1]), type = "response")[,4])
roc.plot(test.data$ACTION, predict(glmnet.1, data.matrix(model.mat.test[,-1]), type = "response")[,2])

rf.1 = randomForest(factor(ACTION) ~ ., data = model.mat)

roc.area(test.data$ACTION, predict(rf.1, model.mat.test[,-1], type="prob")[,2])

### 

glm.1 = glm(ACTION ~ RESOURCE + (ROLE_ROLLUP_1 + ROLE_ROLLUP_2 + ROLE_DEPTNAME) ^ 3 , train.data, family = "binomial", subset = 1:27000)
glmer.1 = glmer(ACTION ~ RESOURCE + (1 | MGR_ID) + , train.data, family="binomial", subset=1:27000)

roc.area(as.numeric(as.character(train.data[27000:32769,]$ACTION)), predict(glm.1, train.data[27000:32769,]))

### 

RF = make_Weka_classifier('weka/classifiers/trees/RandomForest')
WOW(RF)
RF(ACTION ~ ., train.data, control = Weka_control(I = 10, S = 1, depth = 10, D = TRUE))

rf.1 = randomForest(ACTION ~ ., data = train.data, ntree = 10, importance = TRUE, subset = 1:27000)

cf.1 = cforest(ACTION ~ ., train.data, subset = 1:100, controls = ctree_control(mtry = 3, maxdepth = 5))

###

ulist = list(eval = function(y, wt, parms) {
  wmean <- sum(y*wt)/sum(wt)
  rss <- sum(wt*(y-wmean)^2)
  list(label = wmean, deviance = rss)
}, split = function(y, wt, x, parms, continuous)
  {
    n <- length(y)
    if (continuous) {
      ## continuous x variable
      y <- y- sum(y*wt)/sum(wt)
      temp <- cumsum(y*wt)[-n]
      left.wt <- cumsum(wt)[-n]
      right.wt <- sum(wt) - left.wt
      lmean <- temp/left.wt
      rmean <- -temp/right.wt
      goodness <- (left.wt*lmean^2 + right.wt*rmean^2)/sum(wt*y^2)
      goodness[mask] <- 0
      list(goodness = goodness, direction = sign(lmean))
    } else {
      ## Categorical X variable
      ux <- sort(unique(x))
      wtsum <- tapply(wt, x, sum)
      ysum <- tapply(y*wt, x, sum)
      means <- ysum/wtsum
      ## For anova splits, we can order the categories by their means
      ## then use the same code as for a non-categorical
      ord <- order(means)
      n <- length(ord)
      yent = vector("numeric", n-1)
      for (i in 1:(n-1)) {
        sel = x %in% ux[ord][1:i]
        yent[i] = entropy(table(y)) - (entropy(table(y[sel])) * sum(sel) +
              entropy(table(y[!sel])) * (n-sum(sel))) / length(sel)
      }
      goodness <- yent
      list(goodness = goodness,
           direction = ux[ord])
    }
  }, init = function(y, offset, parms, wt) {
    if (length(offset)) y <- y - offset
    sfun <- function(yval, dev, wt, ylevel, digits ) {
      paste(" mean=", format(signif(yval, digits)),
            ", MSE=" , format(signif(dev/wt, digits)),
            sep = '')
    }
    environment(sfun) <- .GlobalEnv
    list(y = c(y), parms = parms, numresp = 1, numy = 1, summary = sfun)
  })

## sample drawn with replacement
## best split among random subeset of featues
## average prediction / or vote

rpart.1 = rpart(ACTION ~ ., train.data, subset = 1:27000, method = ulist, parms = list(prior = c(.06, .94)))


