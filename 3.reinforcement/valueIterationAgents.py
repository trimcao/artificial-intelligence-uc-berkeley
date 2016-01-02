# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.oldValues = util.Counter()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #states = mdp.getStates()
        #print states
        #state = (3, 2)
        #state = 'TERMINAL_STATE'
        #print mdp.getPossibleActions(state)
        #print mdp.getTransitionStatesAndProbs(state, 'exit')
        #print mdp.getReward(state, 'exit', 'TERMINAL_STATE')
        #print mdp.isTerminal((3, 2))
        
        states = mdp.getStates()
        # maybe we don't need to remove TERMINAL_STATE, for later
        #states.remove('TERMINAL_STATE')
        #print states
        # do value iteration for self.iterations times
        for iteration in range(self.iterations):
            # update value for each state
            for eachState in states:
                # get available of the current state
                # later: what if a state has no action?
                actions = mdp.getPossibleActions(eachState)
                # write a helper method here to find the sum
                possibleVals = []
                for eachAction in actions:
                    possibleVals.append(self.sumOverActions(eachState, eachAction))
                # update the self.values dictionary
                if (len(actions) > 0):
                    self.values[eachState] = max(possibleVals)
                else:
                    self.values[eachState] = 0
            # update the self.oldValues 
            for key in self.values:
                self.oldValues[key] = self.values[key]


    def sumOverActions(self, state, action):
        """
        Helper method to compute the sum part in Bellman's equation.
        """
        total = 0
        #print mdp.getTransitionStatesAndProbs(state, 'exit')
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for each in transitions:
            nextState = each[0]
            prob = each[1]
            reward = self.mdp.getReward(state, action, nextState)
            total += prob * (reward + self.discount * self.oldValues[nextState])
        return total

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = self.sumOverActions(state, action)
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # case with state that has no legal actions
        actions = self.mdp.getPossibleActions(state)
        if (len(actions) == 0):
            return None
        else:
            # later: tie-breaking?
            maxAction = None
            maxValue = -float('inf')
            # extract the best action from values
            for eachAction in actions:
                value = self.computeQValueFromValues(state, eachAction)
                if (value > maxValue):
                    maxValue = value
                    maxAction = eachAction
            return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
