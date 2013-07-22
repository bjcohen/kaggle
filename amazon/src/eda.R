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
library(ggplot2)
library(plyr)

train.data = read.csv('train.csv')
test.data = read.csv('test.csv')
train.data[-1] = as.data.frame(lapply(train.data[-1], factor))
test.data[-1] = as.data.frame(lapply(test.data[-1], factor))

library(e1071)

###

nn = function (test.row) {
  hamming.dist = apply(train.data[1:27000,-c(1,2)], 1, function(x) sum(x != test.row))
  print(min(hamming.dist))
  nn.idx = which(hamming.dist == min(hamming.dist))
  mean(train.data[nn.idx,1])
}

apply(train.data[27001:27002,-c(1,2)], 1, nn)

apply(train.data[27001:32769,-c(1,2)], 1, nn)

###

rollup = tapply(train.data$ROLE_ROLLUP_2, train.data$ROLE_ROLLUP_1, paste)
rollup = lapply(rollup, table)

logit.1 = glm(ACTION ~ RESOURCE + MGR_ID + ROLE_ROLLUP_1 + ROLE_ROLLUP_2 + ROLE_DEPTNAME + ROLE_TITLE + ROLE_FAMILY_DESC + ROLE_FAMILY + ROLE_CODE, data = train.data, family = 'binomial')

logit.2 = glm(ACTION ~ RESOURCE, data = train.data, family = 'binomial')


roc.area(train.data$ACTION[27001:32769], unlist(read.table("libfm.model")))
