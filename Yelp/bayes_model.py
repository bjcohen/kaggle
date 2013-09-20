import eda
import os
import operator
import datetime
import pandas as pd
import numpy as np
from pymc import deterministic, HalfNormal, Poisson, Uniform, MCMC, stochastic, Pareto, Normal, Binomial, Model, potential
from pymc.Matplot import plot

### training data

business_fn = 'yelp_training_set_business.json'
review_fn = 'yelp_training_set_review.json'
user_fn = 'yelp_training_set_user.json'
checkin_fn = 'yelp_training_set_checkin.json'

train_directory = 'yelp_training_set'

business_data = eda.read_streaming_json(os.path.join(train_directory, business_fn))
review_data = eda.read_streaming_json(os.path.join(train_directory, review_fn))
user_data = eda.read_streaming_json(os.path.join(train_directory, user_fn))
checkin_data = eda.read_streaming_json(os.path.join(train_directory, checkin_fn))

## preprocess

categories = set([i for l in business_data['categories'] for i in l])
del business_data['neighborhoods']

vote_types = ['useful', 'funny', 'cool']

for v in vote_types:
    review_data['votes_' + v] = review_data['votes'].map(operator.itemgetter(v))
    user_data['votes_' + v] = user_data['votes'].map(operator.itemgetter(v))

del review_data['votes']
del user_data['votes']

review_data['date'] = review_data['date'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

### Business plate
n_categories
n_businesses
beta = Normal("beta", mu=0, tau=, size=3)
eta = Normal("eta", mu=0, tau=, size=n_categories)

business_review_count = Poisson("business_review_count", mu=, observed=True)
business_stars = Multinomial("business_stars", p=np.array([.4, .2, .2, .2, .4]), observed=True)

business_score = Normal("business_score", x=beta[0] * business_review_count + beta[1] * business_stars + beta[2] * is_open + np.dot(eta, categories))

### User plate
n_users
gamma = Normal("gamma", mu=, tau=, size=2)

user_review_count = Poisson("user_review_count", mu=, observed=True)
user_stars = Multinomial("user_stars", p=np.array([.4, .2, .2, .2, .4]), observed=True)

user_score = Normal("user_score", mu=gamma[0] * review_count + gamma[1] * average_stars, tau=)

### Checkin plate
checkins_by_day_prior = Dirichlet("checkins_by_day_prior", np.ones(7))
checkins_by_hour_prior = Dirichlet("checkins_by_hour_prior", np.ones(24))
checkins_by_day = Multinomial("checkins_by_day", 7, business_score * checkins_by_day_prior)
checkins_by_hour = Mutinomial("checkins_by_hour", 24, business_score * checkins_by_hour_prior)

### Review plate
n_feat
alpha = Normal("alpha", mu=0, tau=, size=n_feat)
text_feat = Bernoulli("text_feat", p=, value=, observed=True)
theta = Normal("theta", mu=0, tau=, size=5)

@deterministic
def text_score_mean(alpha = alpha, text_vec = text_vec):
    return sum(alpha * text_feat)

text_score = Normal("text_score", mu=text_score_mean, tau=)

votes_useful = HalfNormal("votes_useful", x=theta[0] * stars + theta[1] * date + theta[2] * text_score + theta[3] * business_score + theta[4] * user_score,
                          tau=,
                          value=,
                          observed=True)

### Loss function
@potential
def error(pred_votes_useful = pred_votes_useful, votes_useful = votes_useful):
    '''RMSLE loss function'''
    return np.sqrt(np.mean((np.log(pred_votes_useful + 1) - np.log(votes_useful + 1)) ** 2))

### whole model
model = Model([alpha, stars, date, text_score, business_score, user_score, votes_useful])

M = MCMC(model)
M.sample(iter=10000, burn=1000, thin=10)
M.trace('votes_useful')[:]
plot(M)

## use sp to maximize loss fct wrt posterior
