####
#### Easy ice cream problem for Hidden Markov Model training
#### Forward - backward algorithm
#### Implementation detail : http://www.cs.tut.fi/~sgn24006/PDF/L08-HMMs.pdf
#### Find parameters (transition/observation probabilities) when unspecified that maximizes P(O|theta)
import numpy as np

## Forward backward algorithm can help us compute the transition probability and observation probability from an observation sequence,
## Even if latents space is hidden
## Transition probability can be estimated by following formula
## a_ij = expected number of transitions from state i to state j / expected number of transitions from state i


def random_generator(time):
	# We have two discrete latent variables, "Hot" and "Cold" weawther
	# We have no idea how many ice creams we had under what weather

	# Let's say we have had some ice creams every day either 1 or 2 or 3
	states = {"H":0, "C":1}
	obsindexes = {1:0, 2:1, 3:2}
	
	observation = list(np.random.randint(low = 1, high=4, size=time))
	print(observation)
	# observation = [1,2,3]
	return observation, states, obsindexes

# random initialized initial distribution, emission distribution, transition distribution
def initialrandom_distributions(time, states, obsindexes):
	## initial probability
	statesNum = len(states)
	iceCreamNum = len(obsindexes)
	pi = np.random.rand(statesNum)
	pi = pi/sum(pi)
	# pi = np.array([0.5, 0.5])
	# transition probability matrix A, row indicates given state, column indicates inference state
	A = np.zeros([statesNum, statesNum])
	for state in range(statesNum):
		A[state] = np.random.rand(statesNum)
		A[state] = A[state]/sum(A[state])

	# Emission probability matrix B, row indicates given state, column indicates emitted number of ice cream
	B = np.zeros([statesNum, iceCreamNum])
	for state in range(statesNum):
		B[state] = np.random.rand(iceCreamNum)
		B[state] = B[state]/sum(B[state])
	return pi, A, B


# 'forward algorithm computes likelihood of observation sequence O, give Model'
# For example, joint probability P(State_t=j, 3, 2, 3 | Lamgda), 
# This means when model knows about the parameters, we can calculate marginal distribution of observations
def forward(observation, obsindexes, pi, A, B, time, states):
	statesNum = len(states)
	alpha = np.zeros([statesNum, time])

	## initialization steps
	## Here, alpha_0, 1
	for state in range(statesNum):
		# Here 0 is first observation
		#				  # P(State) * P(O1|State)
		alpha[state, 0] = pi[state]*B[state, obsindexes[observation[0]] ]

	## For recursion step
	## Alpha_t(j), t indicates observation index, j indicates latent state
	## b_j(o_t), j indicates state, o_t indicate observation at t
	## a_ij, indicate transition i to j
	## sum(alpha_(t-1)(i))
	# Time first... so that state iterate over first..
	for t in range(1, time):
	# for state in range(statesNum):
		for state in range(statesNum):
		# for t in range(1, time):
			alpha[state, t] = sum(alpha[all_state,t-1]*A[all_state,state] for all_state in range(statesNum) ) * \
								B[state, obsindexes[observation[t]] ]
	
	return alpha
# backward calculates p(x_T, x_T-1, ... x_t|State_t=j)
# Hence, emitted probability -> p(x, state_t=j|M) = alpha*beta
def backward(observation, obsindexes, A, B, time, states):
	statesNum = len(states)
	beta = np.zeros([statesNum, time])

	# Initialization
	for state in range(statesNum):
		beta[state, time-1] = 1.0

	# Recursion
	for t in range(time-2, -1, -1):
	#for state in range(statesNum):
		#for t in range(time-2, -1, -1):
		for state in range(statesNum):
			### Note, here transition probability index is different,
			### We are computing on a_ij, j->i as opposed to forward
			beta[state, t] = sum( [A[state , all_state] * B[all_state, obsindexes[observation[t+1]]] * beta[all_state, t+1] for all_state in range(statesNum)] )

	return beta

#### E-step, fix parameters, find expected state assignment
def expectation_step(observation, obsindexes, A, B, alpha, beta, states, time, debug):
	statesNum = len(states)
	# For gamma, we can compute all cases with one variable
	# Gamma is probability of being in state i at time t (conditional probability given observation sequences)
	gamma = np.zeros([statesNum, time])
	for t in range(time):
		# denominator is P(O)
		denominator = sum( [ alpha[s2, t]*beta[s2, t] for s2 in range(statesNum) ] )
		for state in range(statesNum):
			gamma[state, t] = alpha[state, t]*beta[state, t] / denominator
		if debug:
			print("gamma - Should be 1 :", sum(gamma[:,t]))

	# Xi expected number of transitions from state i to state j at t
	# Hence, summing over all sequence is expected number of transtions from state i to state j in O
	#xi, we have to calculate all xi for all observation sequences
	xi_allsequences = []
	# denominator = sum( [ alpha[s2, time-1] for s2 in range(statesNum) ] )
	for t in range(time-1):
		## joint probability of P(prev_state = i, next_state = t | Model)
		xi_ij_t = np.zeros([statesNum, statesNum])
		denominator = sum( [ alpha[s2, t]*beta[s2, t] for s2 in range(statesNum) ] )
		## denominator should be fine....
		for prevState in range(statesNum):
			for nextState in range(statesNum):
				# print(prevState, nextState)
				xi_ij_t[prevState, nextState] = alpha[prevState, t] * \
												A[prevState, nextState] * \
												B[nextState, obsindexes[observation[t+1]] ] * \
												beta[nextState, t+1] \
												/ denominator
		if debug:
			print("Xi - Should be 1 :", sum(sum(xi_ij_t)))
		xi_allsequences.append(xi_ij_t)


	# gamma = np.zeros([statesNum, time])
	# for t in range(time-1):
	# 	for state in range(statesNum):
	# 		gamma[state, t] = sum( [xi_allsequences[t][state][state_j] for state_j in range(statesNum)] )
	
	# for state in range(statesNum):	
	# 	gamma[state, time-1] = sum( [xi_allsequences[t][state][state_j] for state_j in range(statesNum)] )

		# if debug:
		#print("gamma - Should be 1 :", sum(gamma[:,t]))
	return gamma, xi_allsequences

#### M-step
def maximization_step(observation, obsindexes, gamma, xi_allsequences, pi, A, B, time, states):
	statesNum = len(states)
	# Initial state occupancy probability is the
	# Expected number of times in state i at time t hence, pi = gamma_1(i)
	pi = np.zeros([statesNum])
	for state in range(statesNum):
		pi[state] = gamma[state,0]

	## Maximize transition probability
	## a_ij = expected number of transitions from state i to state j over sequence / expected number of transitions from state t
	## hence
	nominator = []
	denominator = []
	for state in range(statesNum):
		denominator = sum([gamma[state, t] for t in range(time-1) ])
		for nextState in range(statesNum):
			nominator = sum( [xi_allsequences[t][state, nextState] for t in range(time-1)] )
			A[state, nextState] = nominator / denominator
	# Estimate the observation probabilities – the number of
	# times being in state j and observing k between the
	# number of times in state j.
	for state in range(statesNum):
		denominator = sum([gamma[state, t] for t in range(time) ])
		for ice_creamNum, indx in obsindexes.items():
			nominator = sum([gamma[state, t] for t in range(time) if observation[t] == ice_creamNum])
			B[state, indx] = nominator/denominator
	# print(pi, sum(pi), A)
	# print(A, B)
	return pi, A, B

def main():
	time  = 5
	# generate observations
	observation, states, obsindexes= random_generator(time)
	initial_prob, trans_prob, emiss_prob = initialrandom_distributions(time, states, obsindexes)
	print("Initial transition : ", trans_prob)
	print("Initial emission : ", emiss_prob, np.sum(emiss_prob, axis=1))
	for it in range(20):
		alpha = forward(observation, obsindexes, initial_prob, trans_prob, emiss_prob, time, states)
		beta = backward(observation, obsindexes, trans_prob, emiss_prob, time, states)
		gamma, xi_allsequences = expectation_step(observation, obsindexes, trans_prob, emiss_prob, alpha, beta, states, time, False)
		# print(gamma)
		# print(beta)
		initial_prob, trans_prob, emiss_prob = maximization_step(observation, obsindexes, gamma, xi_allsequences, initial_prob, trans_prob, emiss_prob, time, states)
		if it % 2 == 0:
			# print(emiss_prob, np.sum(emiss_prob, axis=1))
			print(trans_prob, np.sum(emiss_prob, axis=1))
			# print(alpha)
main()










