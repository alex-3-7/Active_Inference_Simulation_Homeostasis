# import relevant modules
from scipy.stats import norm
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# define the bernoulli distribution that models the prior hidden glucose state
# this bernoulli distribution will be considered as the generative process for the hidden states
p_process = 0.7
# it is also the "good" prior for the generative model, since it coincides perfectly with the generative process
# p_1 is the prior probability of being in the hyperglycemic state
p_1 = p_process
# correspondingly 1 - p_1 is the prior probability of being in the hypoglycemic state

# now define a second Bernoulli distribution that models not so good priors (since it does not coincide with the generative process
# here, the probabilities of the hypo- and hyperglycemic states are equal, thus 1-p_2 = 0.5 is the probability of the hypoglycemic state
p_2 = 0.5

# plotting the probability mass function of the first bernoulli
# red denotes the hyper-, blue the hypoglycemic states
fig, ax = plt.subplots(1, 1)
ax.plot(0, (1-p_1), "bo", ms=10, mec = "blue")
ax.plot(1, p_1, "ro", ms=10, mec="r")
ax.vlines(0, 0, (1-p_1), colors="blue", lw=5)
ax.vlines(1, 0, p_1, colors="r", lw=5)
plt.ylim((0, 1))
plt.show()

# plotting the probability mass function of the second bernoulli
# red denotes the hyper-, blue the hypoglycemic states
fig, ax = plt.subplots(1, 1)
ax.plot(0, (1-p_2), "bo", ms=10, mec = "blue")
ax.plot(1, p_2, "ro", ms=10, mec="r")
ax.vlines(0, 0, (1-p_2), colors="blue", lw=5)
ax.vlines(1, 0, p_2, colors="r", lw=5)
plt.ylim((0, 1))
plt.show()

#######
######


# next, we define the likelihood functions as Gaussians
# these Gaussians model the distribution of (observable) firing rates depending on the (hidden) glucose state

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 15, 0.001)

def likelihood_high_pre_1(x):
    return norm.pdf(x,0,2)

def likelihood_high_pre_2(x):
    return norm.pdf(x,8,2)

def likelihood_low_pre_1(x):
    return norm.pdf(x,0,5)

def likelihood_low_pre_2(x):
    return norm.pdf(x,8,5)


# plot the likelihoods
# blue is the distribution of firing rates when the hidden state is hypoglycemic
# red is the distribution of firing rates when the hidden state is hyperglycemic
plt.plot(x_axis, likelihood_high_pre_1(x_axis), color = "b")
plt.plot(x_axis, likelihood_high_pre_2(x_axis), color = "r")
plt.ylim((0, 0.2))
plt.show()

# check that the likelihoods integrate to one (optional)
#print(scipy.integrate.quad(likelihood_high_pre_1, -np.inf, np.inf))
#print(scipy.integrate.quad(likelihood_high_pre_2, -np.inf, np.inf))

plt.plot(x_axis, likelihood_low_pre_1(x_axis), color = "b")
plt.plot(x_axis, likelihood_low_pre_2(x_axis), color = "r")
plt.ylim((0, 0.2))
plt.show()


######
######
# choose a pseudo-empirical observed firing rate of the gluco-sensory neuron
o = 9

# define the conditional distributions, obtained by multiplying likelihoods and priors
def conditional_density_good_prior_high_pre_like_1(y):
     return (1-p_1)*likelihood_high_pre_1(y)

def conditional_density_good_prior_high_pre_like_2(y):
     return p_1*likelihood_high_pre_2(y)

def conditional_density_bad_prior_high_pre_like_1(y):
     return (1-p_2)*likelihood_high_pre_1(y)

def conditional_density_bad_prior_high_pre_like_2(y):
     return p_2*likelihood_high_pre_2(y)

# plot the conditional distributions, which represent the firing rates, scaled by the proportion of the respective hidden glucose state
plt.plot(x_axis, conditional_density_good_prior_high_pre_like_1(x_axis), color = "b")
plt.plot(x_axis, conditional_density_good_prior_high_pre_like_2(x_axis), color = "r")

plt.ylim((0, 0.2))

# plot the value of the observed firing rate of the gluco-sensory neuron
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=conditional_density_good_prior_high_pre_like_2(o), ls='--', color = "black")

plt.show()

# repeat the same plot as above for the priors not matching the generative process
plt.plot(x_axis, conditional_density_bad_prior_high_pre_like_1(x_axis), color = "b")
plt.plot(x_axis, conditional_density_bad_prior_high_pre_like_2(x_axis), color = "r")

plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=conditional_density_bad_prior_high_pre_like_2(o), ls='--', color = "black")

plt.show()

###
###

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 15, 0.001)

# define the mixture density obtained by marginalising (summing) over the hidden glucose states
# this mixture density corresponds to the pdf of the random variable from which observations of the glucosensory neuron are sampled (special case of a Gaussian mixture model with k = 2 groups here)
def mixture_density_good_prior_high_pre_like(y):
    return conditional_density_good_prior_high_pre_like_1(y)+conditional_density_good_prior_high_pre_like_2(y)

# Mean = 0, SD = 2.
plt.plot(x_axis, mixture_density_good_prior_high_pre_like(x_axis), color = "black")
plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=mixture_density_good_prior_high_pre_like(o), ls='--', color = "black")
plt.show()

# repeat the same as above for the priors not matching the generative process
def mixture_density_bad_prior_high_pre_like(y):
    return conditional_density_bad_prior_high_pre_like_1(y)+conditional_density_bad_prior_high_pre_like_2(y)

x_axis = np.arange(-10, 15, 0.001)
plt.plot(x_axis, mixture_density_bad_prior_high_pre_like(x_axis), color = "black")
plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=mixture_density_bad_prior_high_pre_like(o), ls='--', color = "black")
plt.show()

# check to make sure that we actually obtain probability densities, which should integrate to 1 (optional)
#print(scipy.integrate.quad(mixture_density_good_prior_high_pre_like, -np.inf, np.inf))
#print(scipy.integrate.quad(mixture_density_bad_prior_high_pre_like, -np.inf, np.inf))


####
####

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 15, 0.001)

# define the surprise by taking the negative logarithm of the density above
def surprise_good_prior_high_pre_like(y):
    return -np.log(mixture_density_good_prior_high_pre_like(y))

plt.plot(x_axis, surprise_good_prior_high_pre_like(x_axis), color = "black")
plt.ylim((0, 14))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=surprise_good_prior_high_pre_like(o), ls='--', color = "black")
plt.show()

# repeat the same as above for the priors not matching the generative process
def surprise_bad_prior_high_pre_like(y):
    return -np.log(mixture_density_bad_prior_high_pre_like(y))

plt.plot(x_axis, surprise_bad_prior_high_pre_like(x_axis), color = "black")
plt.ylim((0, 14))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=surprise_bad_prior_high_pre_like(o), ls='--', color = "black")
plt.show()


###
###

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 15, 0.001)
#o = 9


def conditional_density_good_prior_high_pre_like_1(y):
     return (1-p_1)*likelihood_high_pre_1(y)

def conditional_density_good_prior_high_pre_like_2(y):
     return p_1*likelihood_high_pre_2(y)

def conditional_density_good_prior_low_pre_like_1(y):
     return (1-p_1)*likelihood_low_pre_1(y)

def conditional_density_good_prior_low_pre_like_2(y):
     return p_1*likelihood_low_pre_2(y)

def conditional_density_bad_prior_low_pre_like_1(y):
    return (1-p_2)*likelihood_low_pre_1(y)

def conditional_density_bad_prior_low_pre_like_2(y):
    return (1-p_2)*likelihood_low_pre_2(y)


plt.plot(x_axis, conditional_density_good_prior_high_pre_like_1(x_axis), color = "b")
plt.plot(x_axis,conditional_density_good_prior_high_pre_like_2(x_axis), color = "r")
plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=conditional_density_good_prior_high_pre_like_2(o), ls='--', color = "black")
plt.show()

plt.plot(x_axis,conditional_density_good_prior_low_pre_like_1(x_axis), color = "b")
plt.plot(x_axis, conditional_density_good_prior_low_pre_like_2(x_axis), color = "r")
plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=conditional_density_good_prior_low_pre_like_2(o), ls='--', color = "black")
plt.show()

plt.plot(x_axis,conditional_density_bad_prior_low_pre_like_1(x_axis), color = "b")
plt.plot(x_axis, conditional_density_bad_prior_low_pre_like_2(x_axis), color = "r")
plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=conditional_density_bad_prior_low_pre_like_2(o), ls='--', color = "black")
plt.show()


####
####

import scipy
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 15, 0.001)

def mixture_density_good_prior_high_pre_like(y):
    return conditional_density_good_prior_high_pre_like_1(y)+conditional_density_good_prior_high_pre_like_2(y)

plt.plot(x_axis, mixture_density_good_prior_high_pre_like(x_axis), color = "black")
plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=mixture_density_good_prior_high_pre_like(o), ls='--', color = "black")
plt.show()

#print(scipy.integrate.quad(mixture_density_1, o, np.inf))

#print(scipy.integrate.quad(mixture_density_good_prior_high_pre_like, -np.inf, np.inf))

def mixture_density_good_prior_low_pre_like(y):
    return conditional_density_good_prior_low_pre_like_1(y)+conditional_density_good_prior_low_pre_like_2(y)

x_axis = np.arange(-10, 15, 0.001)

plt.plot(x_axis, mixture_density_good_prior_low_pre_like(x_axis), color = "black")


plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=mixture_density_good_prior_low_pre_like(o), ls='--', color = "black")
plt.show()

#print(scipy.integrate.quad(mixture_density_good_prior_low_pre_like, o, np.inf))


print(scipy.integrate.quad(mixture_density_good_prior_low_pre_like, -np.inf, np.inf))


def mixture_density_bad_prior_low_pre_like(y):
    return conditional_density_bad_prior_low_pre_like_1(y)+conditional_density_bad_prior_low_pre_like_2(y)

x_axis = np.arange(-10, 15, 0.001)

plt.plot(x_axis, mixture_density_bad_prior_low_pre_like(x_axis), color = "black")


plt.ylim((0, 0.2))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=mixture_density_bad_prior_low_pre_like(o), ls='--', color = "black")
plt.show()

#print(scipy.integrate.quad(mixture_density_good_prior_low_pre_like, o, np.inf))


#print(scipy.integrate.quad(mixture_density_bad_prior_low_pre_like, -np.inf, np.inf))


####
####

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 15, 0.001)


###
def surprise_good_prior_high_pre_like(y):
    return -np.log(mixture_density_good_prior_high_pre_like(y))

def surprise_bad_prior_high_pre_like(y):
    return -np.log(mixture_density_bad_prior_high_pre_like(y))


plt.plot(x_axis, surprise_good_prior_high_pre_like(x_axis), color = "black")
plt.ylim((0, 14))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=surprise_good_prior_high_pre_like(o), ls='--', color = "black")
plt.show()

def surprise_good_prior_low_pre_like(y):
    return -np.log(mixture_density_good_prior_low_pre_like(y))

plt.plot(x_axis, surprise_good_prior_low_pre_like(x_axis), color = "black")
plt.ylim((0, 14))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=surprise_good_prior_low_pre_like(o), ls='--', color = "black")
plt.show()


def surprise_bad_prior_low_pre_like(y):
    return -np.log(mixture_density_bad_prior_low_pre_like(y))

plt.plot(x_axis, surprise_bad_prior_low_pre_like(x_axis), color = "black")
plt.ylim((0, 14))
plt.axvline(x=o, ls='--', color = "black")
plt.axhline(y=surprise_bad_prior_low_pre_like(o), ls='--', color = "black")
plt.show()

####
####

# now we define the analytical posterior distributions obtained by using Bayes's Theorem
# note that this is not the way how the approximate posteriors are usually obtained in Active Inference
# here, we have the special case that we are using the "correct" model structure, only comparing different parameter values
def posterior_good_prior_high_pre_like_s_minus(x):
    return conditional_density_good_prior_high_pre_like_1(x)/mixture_density_good_prior_high_pre_like(x)

def posterior_good_prior_high_pre_like_s_plus(x):
    return conditional_density_good_prior_high_pre_like_2(x)/mixture_density_good_prior_high_pre_like(x)

plt.plot(x_axis, posterior_good_prior_high_pre_like_s_minus(x_axis), color = "blue")
plt.plot(x_axis, posterior_good_prior_high_pre_like_s_plus(x_axis), color = "red")
#plt.plot(x_axis, posterior_good_prior_high_pre_like_s_plus(x_axis)+posterior_good_prior_high_pre_like_s_minus(x_axis), color = "black") # optional check that we obtain a posterior by plotting the sum for both hidden glucose states, showing that they sum to one
plt.axvline(x=o, ls='--', color = "black")
plt.xlim((-10,15))
plt.show()


def posterior_bad_prior_high_pre_like_s_minus(x):
    return conditional_density_bad_prior_high_pre_like_1(x)/mixture_density_bad_prior_high_pre_like(x)

def posterior_bad_prior_high_pre_like_s_plus(x):
    return conditional_density_bad_prior_high_pre_like_2(x)/mixture_density_bad_prior_high_pre_like(x)


plt.plot(x_axis, posterior_bad_prior_high_pre_like_s_minus(x_axis), color = "blue")
plt.plot(x_axis, posterior_bad_prior_high_pre_like_s_plus(x_axis), color = "red")
plt.axvline(x=o, ls='--', color = "black")
plt.xlim((-10,15))
plt.show()


def posterior_good_prior_low_pre_like_s_minus(x):
    return conditional_density_good_prior_low_pre_like_1(x)/mixture_density_good_prior_low_pre_like(x)

def posterior_good_prior_low_pre_like_s_plus(x):
    return conditional_density_good_prior_low_pre_like_2(x)/mixture_density_good_prior_low_pre_like(x)


plt.plot(x_axis, posterior_good_prior_low_pre_like_s_minus(x_axis), color = "blue")
plt.plot(x_axis, posterior_good_prior_low_pre_like_s_plus(x_axis), color = "red")
plt.axvline(x=o, ls='--', color = "black")
plt.xlim((-10,15))
plt.show()


def posterior_bad_prior_low_pre_like_s_minus(x):
    return conditional_density_bad_prior_low_pre_like_1(x)/mixture_density_bad_prior_low_pre_like(x)


def posterior_bad_prior_low_pre_like_s_plus(x):
    return conditional_density_bad_prior_low_pre_like_2(x)/mixture_density_bad_prior_low_pre_like(x)


plt.plot(x_axis, posterior_bad_prior_low_pre_like_s_minus(x_axis), color = "blue")
plt.plot(x_axis, posterior_bad_prior_low_pre_like_s_plus(x_axis), color = "red")
plt.axvline(x=o, ls='--', color = "black")
plt.xlim((-10,15))
plt.show()

# now we simulate samples from the generative process
# number of samples
N = 1000

samples_generative = np.array([])

samples_good_prior_good_likelihood = []

samples_bad_prior_good_likelihood = []

samples_good_prior_bad_likelihood = []

samples_bad_prior_bad_likelihood = []

samples = []
samples_free_energy_good_prior_high_pre_likelihood = []
samples_free_energy_bad_prior_high_pre_likelihood = []
samples_free_energy_good_prior_low_pre_likelihood = []
samples_free_energy_bad_prior_low_pre_likelihood = []

# we first consider the case where we assume that the "good" generative model precisely matches the generative process
def process_mixture_density(x):
    return mixture_density_good_prior_high_pre_like(x)

def process_posterior_s_plus(x):
    return posterior_good_prior_high_pre_like_s_plus(x)

def process_posterior_s_minus(x):
    return posterior_good_prior_high_pre_like_s_minus(x)

# looping over N to generate the samples and calculate the surprises and free energies for the observations
for i in range(N):
    new_sample = np.random.choice(x_axis, size=1, p=process_mixture_density(x_axis) / sum(process_mixture_density(x_axis)))
    samples = np.append(samples, new_sample)
    samples_good_prior_good_likelihood = np.append(samples_good_prior_good_likelihood, surprise_good_prior_high_pre_like(new_sample))
    samples_bad_prior_good_likelihood = np.append(samples_bad_prior_good_likelihood, surprise_bad_prior_high_pre_like(new_sample))
    samples_good_prior_bad_likelihood = np.append(samples_good_prior_bad_likelihood, surprise_good_prior_low_pre_like(new_sample))
    samples_bad_prior_bad_likelihood = np.append(samples_bad_prior_bad_likelihood, surprise_bad_prior_low_pre_like(new_sample))
    samples_free_energy_good_prior_high_pre_likelihood = np.append(samples_free_energy_good_prior_high_pre_likelihood, np.log((1-posterior_good_prior_high_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus(new_sample)))+posterior_good_prior_high_pre_like_s_plus(new_sample)*np.log((posterior_good_prior_high_pre_like_s_plus(new_sample)*(1-process_posterior_s_plus(new_sample)))/(process_posterior_s_plus(new_sample)*(1-posterior_good_prior_high_pre_like_s_plus(new_sample))))+surprise_good_prior_high_pre_like(new_sample))
    samples_free_energy_bad_prior_high_pre_likelihood = np.append(samples_free_energy_bad_prior_high_pre_likelihood, np.log((1-posterior_bad_prior_high_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus(new_sample)))+posterior_bad_prior_high_pre_like_s_plus(new_sample)*np.log((posterior_bad_prior_high_pre_like_s_plus(new_sample)*(1-process_posterior_s_plus(new_sample)))/(process_posterior_s_plus(new_sample)*(1-posterior_bad_prior_high_pre_like_s_plus(new_sample))))+surprise_bad_prior_high_pre_like(new_sample))
    samples_free_energy_good_prior_low_pre_likelihood = np.append(samples_free_energy_good_prior_low_pre_likelihood, np.log((1-posterior_good_prior_low_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus(new_sample)))+posterior_good_prior_low_pre_like_s_plus(new_sample)*np.log((posterior_good_prior_low_pre_like_s_plus(new_sample)*(1-process_posterior_s_plus(new_sample)))/(process_posterior_s_plus(new_sample)*(1-posterior_good_prior_low_pre_like_s_plus(new_sample))))+surprise_good_prior_low_pre_like(new_sample))
    samples_free_energy_bad_prior_low_pre_likelihood = np.append(samples_free_energy_bad_prior_low_pre_likelihood, np.log((1-posterior_bad_prior_low_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus(new_sample)))+posterior_bad_prior_low_pre_like_s_plus(new_sample)*np.log((posterior_bad_prior_low_pre_like_s_plus(new_sample)*(1-process_posterior_s_plus(new_sample)))/(process_posterior_s_plus(new_sample)*(1-posterior_bad_prior_low_pre_like_s_plus(new_sample))))+surprise_bad_prior_low_pre_like(new_sample))


# turn the lists into numpy arrays
samples_good_prior_good_likelihood = np.array(samples_good_prior_good_likelihood)
samples_bad_prior_good_likelihood = np.array(samples_bad_prior_good_likelihood)
samples_good_prior_bad_likelihood = np.array(samples_good_prior_bad_likelihood)
samples_bad_prior_bad_likelihood = np.array(samples_bad_prior_bad_likelihood)


# alternative option: plot the resulting samples, evaluated by their surprise for the respective generative model
# however, the density plots may be better for visual inspection
# also print the mean values of the samples, if numerical comparison is desired
#plt.hist(samples_good_prior_good_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_good_prior_good_likelihood))

#plt.hist(samples_bad_prior_good_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_bad_prior_good_likelihood))

#plt.hist(samples_good_prior_bad_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_good_prior_bad_likelihood))


#plt.hist(samples_bad_prior_bad_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_bad_prior_bad_likelihood))


# plot the resulting samples, evaluated by their surprise for the respective generative model
sns.kdeplot(samples_good_prior_good_likelihood)
plt.ylim((0,4))
plt.xlim((0,5))
plt.show()

sns.kdeplot(samples_bad_prior_good_likelihood)
plt.ylim((0,4))
plt.xlim((0,5))
plt.show()


sns.kdeplot(samples_good_prior_bad_likelihood)
plt.ylim((0,4))
plt.xlim((0,5))
plt.show()


sns.kdeplot(samples_bad_prior_bad_likelihood)
plt.ylim((0,4))
plt.xlim((0,5))
plt.show()


# alternative option: plotting of the samples as histograms, density plots below may however be better for visual inspection
#plt.hist(samples_good_prior_good_likelihood, color = "green")
#plt.hist(samples_bad_prior_good_likelihood, color = "cyan")
#plt.hist(samples_good_prior_bad_likelihood, color = "orange")
#plt.hist(samples_bad_prior_bad_likelihood, color = "red")
#plt.show()

sns.kdeplot(samples_good_prior_good_likelihood, color = "green", fill = True, label = "Generative Model 1")
sns.kdeplot(samples_bad_prior_good_likelihood, color = "cyan", fill = True, label = "Generative Model 2")
sns.kdeplot(samples_good_prior_bad_likelihood,color = "orange", fill = True, label = "Generative Model 3")
sns.kdeplot(samples_bad_prior_bad_likelihood, color = "red", fill = True, label = "Generative Model 4")
plt.legend(loc="upper right")
plt.xlabel("Surprise of Observation")
plt.ylim((0,4.0))
plt.show()


## plotting all the (variational) free energies for the sampled observations in one plot for comparison
samples_free_energy_good_prior_high_pre_likelihood = np.array(samples_free_energy_good_prior_high_pre_likelihood)
samples_free_energy_bad_prior_high_pre_likelihood = np.array(samples_free_energy_bad_prior_high_pre_likelihood)
samples_free_energy_good_prior_low_pre_likelihood = np.array(samples_free_energy_good_prior_low_pre_likelihood)
samples_free_energy_bad_prior_low_pre_likelihood = np.array(samples_free_energy_bad_prior_low_pre_likelihood)



# alternative option: plot the resulting samples, evaluated by their surprise for the respective generative model as histograms
#plt.hist(samples_free_energy_good_prior_high_pre_likelihood)
#plt.ylim((0, 250))
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_free_energy_good_prior_high_pre_likelihood))

#plt.hist(samples_free_energy_bad_prior_high_pre_likelihood)
#plt.ylim((0, 250))
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_free_energy_bad_prior_high_pre_likelihood))

#plt.hist(samples_free_energy_good_prior_low_pre_likelihood)
#plt.ylim((0, 250))
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_free_energy_good_prior_low_pre_likelihood))


#plt.hist(samples_free_energy_bad_prior_low_pre_likelihood)
#plt.ylim((0, 250))
#plt.xlim((2.2,3.3))
#plt.show()

#print(np.mean(samples_free_energy_bad_prior_low_pre_likelihood))

# alternative option: all histograms in one plot for better visual comparison, however densities below may be better for visual inspection
#plt.hist(samples_free_energy_good_prior_high_pre_likelihood, color = "green", label = "Generative Model 1")
#plt.hist(samples_free_energy_bad_prior_high_pre_likelihood, color = "cyan", label = "Generative Model 2")
#plt.hist(samples_free_energy_good_prior_low_pre_likelihood, color = "orange", label = "Generative Model 3")
#plt.hist(samples_free_energy_bad_prior_low_pre_likelihood, color = "red", label = "Generative Model 4")
#plt.show()

# now all densities in one plot for better visual comparison
sns.kdeplot(samples_free_energy_good_prior_high_pre_likelihood, color = "green", fill = True, label = "Generative Model 1")
sns.kdeplot(samples_free_energy_bad_prior_high_pre_likelihood, color = "cyan", fill = True, label = "Generative Model 2")
sns.kdeplot(samples_free_energy_good_prior_low_pre_likelihood, color = "orange", fill = True, label = "Generative Model 3")
sns.kdeplot(samples_free_energy_bad_prior_low_pre_likelihood, color = "red", fill = True, label = "Generative Model 4")
plt.legend(loc="upper right")
plt.ylim((0,4.0))
plt.xlabel("Variational Free Energy of Observation")
plt.show()


####
# now we repeat the same procedure while changing the generative process to be equal to the bad prior/low precision likelihood model
# (technically that model is now the best model, since it is identical to the generative process)
# for simplicity, we only  consider the case where we assume that the previously "bad" generative model precisely matches the generative process, other choices are also possible
def process_mixture_density_2(x):
    return mixture_density_bad_prior_low_pre_like(x)

def process_posterior_s_plus_2(x):
    return posterior_bad_prior_low_pre_like_s_plus(x)

def process_posterior_s_minus_2(x):
    return posterior_bad_prior_low_pre_like_s_minus(x)

# number of samples
N = 1000

samples_generative = np.array([])

samples_good_prior_good_likelihood = []

samples_bad_prior_good_likelihood = []

samples_good_prior_bad_likelihood = []

samples_bad_prior_bad_likelihood = []


samples = []
samples_free_energy_good_prior_high_pre_likelihood = []
samples_free_energy_bad_prior_high_pre_likelihood = []
samples_free_energy_good_prior_low_pre_likelihood = []
samples_free_energy_bad_prior_low_pre_likelihood = []

for i in range(N):
    new_sample = np.random.choice(x_axis, size=1, p=process_mixture_density_2(x_axis) / sum(process_mixture_density_2(x_axis)))
    samples = np.append(samples, new_sample)
    samples_good_prior_good_likelihood = np.append(samples_good_prior_good_likelihood, surprise_good_prior_high_pre_like(new_sample))
    samples_bad_prior_good_likelihood = np.append(samples_bad_prior_good_likelihood, surprise_bad_prior_high_pre_like(new_sample))
    samples_good_prior_bad_likelihood = np.append(samples_good_prior_bad_likelihood, surprise_good_prior_low_pre_like(new_sample))
    samples_bad_prior_bad_likelihood = np.append(samples_bad_prior_bad_likelihood, surprise_bad_prior_low_pre_like(new_sample))
    samples_free_energy_good_prior_high_pre_likelihood = np.append(samples_free_energy_good_prior_high_pre_likelihood, np.log((1-posterior_good_prior_high_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus_2(new_sample)))+posterior_good_prior_high_pre_like_s_plus(new_sample)*np.log((process_posterior_s_plus_2(new_sample)*(1-process_posterior_s_plus_2(new_sample)))/(process_posterior_s_plus_2(new_sample)*(1-posterior_good_prior_high_pre_like_s_plus(new_sample))))+surprise_good_prior_high_pre_like(new_sample))
    samples_free_energy_bad_prior_high_pre_likelihood = np.append(samples_free_energy_bad_prior_high_pre_likelihood, np.log((1-posterior_bad_prior_high_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus_2(new_sample)))+posterior_bad_prior_high_pre_like_s_plus(new_sample)*np.log((process_posterior_s_plus_2(new_sample)*(1-process_posterior_s_plus_2(new_sample)))/(process_posterior_s_plus_2(new_sample)*(1-posterior_bad_prior_high_pre_like_s_plus(new_sample))))+surprise_bad_prior_high_pre_like(new_sample))
    samples_free_energy_good_prior_low_pre_likelihood = np.append(samples_free_energy_good_prior_low_pre_likelihood, np.log((1-posterior_good_prior_low_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus_2(new_sample)))+posterior_good_prior_low_pre_like_s_plus(new_sample)*np.log((process_posterior_s_plus_2(new_sample)*(1-process_posterior_s_plus_2(new_sample)))/(process_posterior_s_plus_2(new_sample)*(1-posterior_good_prior_low_pre_like_s_plus(new_sample))))+surprise_good_prior_low_pre_like(new_sample))
    samples_free_energy_bad_prior_low_pre_likelihood = np.append(samples_free_energy_bad_prior_low_pre_likelihood, np.log((1-posterior_bad_prior_low_pre_like_s_plus(new_sample))/(1-process_posterior_s_plus_2(new_sample)))+posterior_bad_prior_low_pre_like_s_plus(new_sample)*np.log((process_posterior_s_plus_2(new_sample)*(1-process_posterior_s_plus_2(new_sample)))/(process_posterior_s_plus_2(new_sample)*(1-posterior_bad_prior_low_pre_like_s_plus(new_sample))))+surprise_bad_prior_low_pre_like(new_sample))

# turn into numpy arrays
samples_good_prior_good_likelihood = np.array(samples_good_prior_good_likelihood)
samples_bad_prior_good_likelihood = np.array(samples_bad_prior_good_likelihood)
samples_good_prior_bad_likelihood = np.array(samples_good_prior_bad_likelihood)
samples_bad_prior_bad_likelihood = np.array(samples_bad_prior_bad_likelihood)

# plot the resulting samples, evaluated by their surprise for the respective generative model
#plt.hist(samples_good_prior_good_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_good_prior_good_likelihood))

#plt.hist(samples_bad_prior_good_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_bad_prior_good_likelihood))

#plt.hist(samples_good_prior_bad_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_good_prior_bad_likelihood))


#plt.hist(samples_bad_prior_bad_likelihood)
#plt.xlim((2.2,3.3))
#plt.show()

print(np.mean(samples_bad_prior_bad_likelihood))

# alternative option:  all histograms in one plot for better visual comparison, however densities below may be better for visual inspection
#plt.hist(samples_good_prior_good_likelihood, color = "green")
#plt.hist(samples_bad_prior_good_likelihood, color = "cyan")
#plt.hist(samples_good_prior_bad_likelihood, color = "orange")
#plt.hist(samples_bad_prior_bad_likelihood, color = "red")
#plt.show()

# now all densities in one plot for better visual comparison
sns.kdeplot(samples_good_prior_good_likelihood, color = "green", fill = True, label = "Generative Model 1")
sns.kdeplot(samples_bad_prior_good_likelihood, color = "cyan", fill = True, label = "Generative Model 2")
sns.kdeplot(samples_good_prior_bad_likelihood,color = "orange", fill = True, label = "Generative Model 3")
sns.kdeplot(samples_bad_prior_bad_likelihood, color = "red", fill = True, label = "Generative Model 4")
plt.xlabel("Surprise of observation")
plt.legend(loc = "upper right")
plt.ylim((0,4.0))
plt.show()


# turn into numpy arrays
samples_free_energy_good_prior_high_pre_likelihood = np.array(samples_free_energy_good_prior_high_pre_likelihood)
samples_free_energy_bad_prior_high_pre_likelihood = np.array(samples_free_energy_bad_prior_high_pre_likelihood)
samples_free_energy_good_prior_low_pre_likelihood = np.array(samples_free_energy_good_prior_low_pre_likelihood)
samples_free_energy_bad_prior_low_pre_likelihood = np.array(samples_free_energy_bad_prior_low_pre_likelihood)



# plot the resulting samples, evaluated by their surprise for the respective generative model
#plt.hist(samples_free_energy_good_prior_high_pre_likelihood)
#plt.show()

print(np.mean(samples_free_energy_good_prior_high_pre_likelihood))

#plt.hist(samples_free_energy_bad_prior_high_pre_likelihood)
#plt.show()

print(np.mean(samples_free_energy_bad_prior_high_pre_likelihood))

#plt.hist(samples_free_energy_good_prior_low_pre_likelihood)
#plt.show()

print(np.mean(samples_free_energy_good_prior_low_pre_likelihood))


#plt.hist(samples_free_energy_bad_prior_low_pre_likelihood)
#plt.show()

print(np.mean(samples_free_energy_bad_prior_low_pre_likelihood))

# now all histograms in one plot for better visual comparison
#plt.hist(samples_free_energy_good_prior_high_pre_likelihood, color = "green")
#plt.hist(samples_free_energy_bad_prior_high_pre_likelihood, color = "cyan")
#plt.hist(samples_free_energy_good_prior_low_pre_likelihood, color = "orange")
#plt.hist(samples_free_energy_bad_prior_low_pre_likelihood, color = "red")
#plt.show()

# now all densities in one plot for better visual comparison
sns.kdeplot(samples_free_energy_good_prior_high_pre_likelihood, color = "green", fill = True, label = "Generative Model 1")
sns.kdeplot(samples_free_energy_bad_prior_high_pre_likelihood, color = "cyan", fill = True, label = "Generative Model 2")
sns.kdeplot(samples_free_energy_good_prior_low_pre_likelihood, color = "orange", fill = True, label = "Generative Model 3")
sns.kdeplot(samples_free_energy_bad_prior_low_pre_likelihood, color = "red", fill = True, label = "Generative Model 4")
plt.xlabel("Variational Free Energy of observation")
plt.legend(loc = "upper right")
plt.ylim((0,4.0))
plt.show()