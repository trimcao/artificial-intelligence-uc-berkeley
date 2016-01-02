# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

## test util.Counter()
#a = util.Counter()
#a = {}
#a[(0,0)] = util.Counter()
#print a
#print a[(0,0)]['north']
#a['south'] = 3
#print a.keys()
#print a['test']
#print a['test']['test2']

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # qValues is a dict of util.Counter()
        self.qValues = {}
        self.visitCount = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 cvif we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state in self.qValues):
            return self.qValues[state][action]
        else:
            return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if (len(actions) == 0):
            return 0.0
        maxVal = -float('inf')        
        for eachAction in actions:
            currentVal = self.getQValue(state, eachAction)
            if (currentVal > maxVal):
                maxVal = currentVal
        return maxVal


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if (len(actions) == 0):
            return None
        # if we have not seen the current state, then flip coin over the
        # actions
        #if not (state in self.qValues):
        #    return random.choice(actions)  
        maxVal = -float('inf')        
        bestActions = []
        for eachAction in actions:
            currentVal = self.getQValue(state, eachAction)
            if (currentVal > maxVal):
                bestActions = []
                maxVal = currentVal
                bestActions.append(eachAction)
            elif (currentVal == maxVal):
                bestActions.append(eachAction)
        # flip coin over actions        
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if (len(legalActions) == 0):
            return None
        bestAction = self.computeActionFromQValues(state)
        randomNumber = random.random()
        if (randomNumber < self.epsilon):
            return random.choice(legalActions)
        else:
            return bestAction

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # learning rate: self.alpha
        # discount: self.discount
        # first, add the state to self.qValues if we have never seen it
        if not (state in self.qValues):
            self.qValues[state] = util.Counter()
            self.visitCount[state] = util.Counter()
        # update the q value
        exploit = (1 - self.alpha) * self.qValues[state][action]
        maxVal = self.computeValueFromQValues(nextState)
        explore = self.alpha * (reward + self.discount * maxVal)
        self.qValues[state][action] = exploit + explore
        self.visitCount[state][action] += 1

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        qValue = 0
        for eachFeat in features:
            qValue += features[eachFeat] * self.weights[eachFeat]
        return qValue


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        maxQVal = self.computeValueFromQValues(nextState) 
        difference = (reward + self.discount * maxQVal) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for eachFeat in features:
            self.weights[eachFeat] += self.alpha * difference * features[eachFeat]
    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
