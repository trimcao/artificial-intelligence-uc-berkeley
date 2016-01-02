# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # evaluate the score
        basicScore = successorGameState.getScore()
        numFood = float(newFood.count()) + 1
        basicScore += (1 / numFood)
        foodList = newFood.asList()
        nearestFood = float(minDistance(newPos, foodList)) + 1
        basicScore += (1 / nearestFood)
        for eachGhost in newGhostStates:
            dist = float(manhattanDistance(newPos, eachGhost.getPosition())) + 1
            scaredTimer = eachGhost.scaredTimer
            if (dist < scaredTimer):
                basicScore += (3 / dist)
            else:
                basicScore += (-3 / dist)
        return basicScore

def minDistance(pos, locList):
    """
    Find the manhattan distance to the nearest location from the current 
    position.
    """
    minDist = float('inf')
    for each in locList:
        dist = manhattanDistance(pos, each)
        if (dist < minDist):
            minDist = dist
    return minDist

def maxDistance(pos, locList):
    """
    Find the manhattan distance to the furthest location from the current 
    position.
    """
    maxDist = -float('inf')
    for each in locList:
        dist = manhattanDistance(pos, each)
        if (dist > maxDist):
            maxDist = dist
    return maxDist

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        currentDepth = 1
        numAgents = gameState.getNumAgents()        
        # find the minimax score and action
        actions = gameState.getLegalActions(agentIndex)
        finalAct = None
        finalVal = -float('inf')
        for eachAction in actions:
            currentSuc = gameState.generateSuccessor(agentIndex, eachAction)
            currentVal = self.value(currentSuc, agentIndex, currentDepth, numAgents)
            if (currentVal > finalVal):
                finalAct = eachAction
                finalVal = currentVal
        #print "Pacman Action's Value: ", str(finalVal)
        return finalAct

    def value(self, gameState, agentIndex, currentDepth, numAgents):
        """
        Helper method to determine the value of current state
        Output: score of the state, and action?
        """
        newDepth = currentDepth
        #print 
        #print 'Value call from Index: ', agentIndex
        if (agentIndex == numAgents - 1):
            # if current agent is the last one, then we may reach the base case
            # otherwise, increase the currentDepth
            if (currentDepth == self.depth):
                #print self.evaluationFunction(gameState)
                return self.evaluationFunction(gameState)
            else:
                newDepth = currentDepth + 1
        #print 'New Depth: ', newDepth
        # find minimax for the next agent 
        nextAgent = (agentIndex + 1) % numAgents
        if (nextAgent == 0):
            return self.maxValue(gameState, nextAgent, newDepth, numAgents)
        else:
            return self.minValue(gameState, nextAgent, newDepth, numAgents)

    def maxValue(self, gameState, agentIndex, currentDepth, numAgents):
        """
        Find the max value of the current game state in a minimax game tree.
        """
        finalVal = -float('inf')
        actions = gameState.getLegalActions(agentIndex)
        # in case we don't have any legal action left
        if (len(actions) == 0):
            #finalVal = self.value(gameState, agentIndex, currentDepth, numAgents) 
            finalVal = self.evaluationFunction(gameState)
        for eachAction in actions:
            eachSuc = gameState.generateSuccessor(agentIndex, eachAction)
            finalVal = max((finalVal, self.value(eachSuc, agentIndex, currentDepth, numAgents))) 
        #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
        return finalVal
    
    def minValue(self, gameState, agentIndex, currentDepth, numAgents):
        """
        Find the min value of the current game state in a minimax game tree.
        """
        finalVal = float('inf')
        actions = gameState.getLegalActions(agentIndex)
        #print "Number of legal actions: ", str(len(actions))
        # in case we don't have any legal action left
        if (len(actions) == 0):
            #finalVal = self.value(gameState, agentIndex, currentDepth, numAgents) 
            finalVal = self.evaluationFunction(gameState)
        for eachAction in actions:
            eachSuc = gameState.generateSuccessor(agentIndex, eachAction)
            finalVal = min((finalVal, self.value(eachSuc, agentIndex, currentDepth, numAgents))) 
        #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
        return finalVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
          Note: alpha = MAX's best option on path to root
                beta  = MIN's best option on path to root
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        currentDepth = 1
        numAgents = gameState.getNumAgents()        
        # find the minimax score and action
        actions = gameState.getLegalActions(agentIndex)
        finalAct = None
        finalVal = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for eachAction in actions:
            currentSuc = gameState.generateSuccessor(agentIndex, eachAction)
            currentVal, newAlpha, newBeta = self.value(currentSuc, agentIndex, 
                                        currentDepth, numAgents, alpha, beta)
            if (currentVal > finalVal):
                finalAct = eachAction
                finalVal = currentVal
            if (currentVal > alpha):
                alpha = currentVal
        #print "Pacman Action's Value: ", str(finalVal)
        return finalAct

    def value(self, gameState, agentIndex, currentDepth, numAgents, alpha, beta):
        """
        Helper method to determine the value of current state
        Output: score of the state, alpha and beta
        """
        newDepth = currentDepth
        #print 
        #print 'Value call from Index: ', agentIndex
        if (agentIndex == numAgents - 1):
            # if current agent is the last one, then we may reach the base case
            # otherwise, increase the currentDepth
            if (currentDepth == self.depth):
                #print self.evaluationFunction(gameState)
                # we return a tuple: evaluation score, alpha, beta
                return (self.evaluationFunction(gameState), alpha, beta)
            else:
                newDepth = currentDepth + 1
        #print 'New Depth: ', newDepth
        # find minimax for the next agent 
        nextAgent = (agentIndex + 1) % numAgents
        if (nextAgent == 0):
            return self.maxValue(gameState, nextAgent, newDepth, numAgents, alpha, beta)
        else:
            return self.minValue(gameState, nextAgent, newDepth, numAgents, alpha, beta)

    def maxValue(self, gameState, agentIndex, currentDepth, numAgents, alpha, beta):
        """
        Find the max value of the current game state in a minimax game tree.
        """
        finalVal = -float('inf')
        nextAlpha = alpha
        actions = gameState.getLegalActions(agentIndex)
        # in case we don't have any legal action left
        if (len(actions) == 0):
            finalVal = self.evaluationFunction(gameState)
        for eachAction in actions:
            eachSuc = gameState.generateSuccessor(agentIndex, eachAction)
            finalVal = max((finalVal, self.value(eachSuc, agentIndex, 
                                    currentDepth, numAgents, nextAlpha, beta)[0])) 
            if (finalVal > beta):
                #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
                #print 'Return early'
                # what should we return here
                return (finalVal, nextAlpha, beta)
            # update alpha
            nextAlpha = max(nextAlpha, finalVal) 
        #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
        return (finalVal, nextAlpha, beta)
    
    def minValue(self, gameState, agentIndex, currentDepth, numAgents, alpha, beta):
        """
        Find the min value of the current game state in a minimax game tree.
        """
        finalVal = float('inf')
        nextBeta = beta
        actions = gameState.getLegalActions(agentIndex)
        #print "Number of legal actions: ", str(len(actions))
        # in case we don't have any legal action left
        if (len(actions) == 0):
            finalVal = self.evaluationFunction(gameState)
        for eachAction in actions:
            #print "Number of successors: ", str(len(successors))
            eachSuc = gameState.generateSuccessor(agentIndex, eachAction)
            finalVal = min((finalVal, self.value(eachSuc, agentIndex, 
                            currentDepth, numAgents, alpha, nextBeta)[0])) 
            if (finalVal < alpha):
                #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
                #print 'Return early'
                return (finalVal, alpha, nextBeta)
            # update beta
            nextBeta = min(nextBeta, finalVal)
        #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
        return (finalVal, alpha, nextBeta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        agentIndex = 0
        currentDepth = 1
        numAgents = gameState.getNumAgents()        
        # find the minimax score and action
        actions = gameState.getLegalActions(agentIndex)
        finalAct = None
        finalVal = -float('inf')
        for eachAction in actions:
            currentSuc = gameState.generateSuccessor(agentIndex, eachAction)
            currentVal = self.value(currentSuc, agentIndex, currentDepth, numAgents)
            if (currentVal > finalVal):
                finalAct = eachAction
                finalVal = currentVal
        #print "Pacman Action's Value: ", str(finalVal)
        return finalAct

    def value(self, gameState, agentIndex, currentDepth, numAgents):
        """
        Helper method to determine the value of current state
        Output: score of the state, and action?
        """
        newDepth = currentDepth
        #print 
        #print 'Value call from Index: ', agentIndex
        if (agentIndex == numAgents - 1):
            # if current agent is the last one, then we may reach the base case
            # otherwise, increase the currentDepth
            if (currentDepth == self.depth):
                #print self.evaluationFunction(gameState)
                return self.evaluationFunction(gameState)
            else:
                newDepth = currentDepth + 1
        #print 'New Depth: ', newDepth
        # find minimax for the next agent 
        nextAgent = (agentIndex + 1) % numAgents
        if (nextAgent == 0):
            return self.maxValue(gameState, nextAgent, newDepth, numAgents)
        else:
            return self.expValue(gameState, nextAgent, newDepth, numAgents)

    def maxValue(self, gameState, agentIndex, currentDepth, numAgents):
        """
        Find the max value of the current game state in a minimax game tree.
        """
        finalVal = -float('inf')
        actions = gameState.getLegalActions(agentIndex)
        # in case we don't have any legal action left
        if (len(actions) == 0):
            #finalVal = self.value(gameState, agentIndex, currentDepth, numAgents) 
            finalVal = self.evaluationFunction(gameState)
        for eachAction in actions:
            eachSuc = gameState.generateSuccessor(agentIndex, eachAction)
            finalVal = max((finalVal, self.value(eachSuc, agentIndex, currentDepth, numAgents))) 
        #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
        return finalVal
    
    def expValue(self, gameState, agentIndex, currentDepth, numAgents):
        """
        Find the expected value of the current game state in a minimax game tree.
        """
        finalVal = 0
        actions = gameState.getLegalActions(agentIndex)
        #print "Number of legal actions: ", str(len(actions))
        numActions = float(len(actions))
        # in case we don't have any legal action left
        if (numActions == 0):
            #finalVal = self.value(gameState, agentIndex, currentDepth, numAgents) 
            finalVal = self.evaluationFunction(gameState)
        else:    
            # assume probability of choosing an action is uniformly at random    
            prob = 1 / numActions
        for eachAction in actions:
            eachSuc = gameState.generateSuccessor(agentIndex, eachAction)
            finalVal += prob * self.value(eachSuc, agentIndex, currentDepth, numAgents) 
        #print 'AgentID: ' + str(agentIndex) + ' - Score: ' + str(finalVal) + ' - Depth: ' + str(currentDepth)
        return finalVal
    
    def actionProb(self):
        """
        Find the probability of an action in the expValue function.
        """
        pass

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # get power pellet
    powerPellets = currentGameState.getCapsules()
    # evaluate the score
    basicScore = currentGameState.getScore()
    numFood = float(newFood.count()) + 1
    basicScore += (1 / numFood)
    foodList = newFood.asList()
    # nearest Food
    #basicScore += (1 / (foodHeuristic((newPos, newFood)) + 1))
    nearestFood = float(minDistance(newPos, foodList)) + 1
    basicScore += (1 / nearestFood)
    # distance to nearest power pellet
    nearestPill = float(minDistance(newPos, powerPellets)) + 1
    basicScore += 2 * (1 / nearestPill)
    for eachGhost in newGhostStates:
        dist = float(manhattanDistance(newPos, eachGhost.getPosition())) + 1
        scaredTimer = eachGhost.scaredTimer
        if (dist < scaredTimer):
            basicScore += 3 * (1 / dist)
        else:
            basicScore += 3 * (-1 / dist)
    return basicScore

    
def foodHeuristic(state):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    # find the food pellets in each quarter
    foodQuarters = [[] for dummy in range(4)]
    for eachFood in foodGrid.asList():
        xPos = eachFood[0]
        yPos = eachFood[1]
        if (xPos < position[0]) and (yPos > position[1]):
            foodQuarters[0].append(eachFood)
        elif (xPos >= position[0]) and (yPos > position[1]):
            foodQuarters[1].append(eachFood)
        elif (xPos < position[0]) and (yPos <= position[1]):
            foodQuarters[3].append(eachFood)
        else:
            foodQuarters[2].append(eachFood)
    # find the highest Manhanttan distance in each quarter
    # and in the whole grid
    highManhattan = []
    maxManhattan = -1
    for idx in range(4):
        currentMax, maxFood = findMaxManhattan(position, foodQuarters[idx])
        if (maxFood != None):
            if (currentMax > maxManhattan):
                maxManhattan = currentMax
            highManhattan.append(maxFood)
    # find the shortest path through these points
    manhattanSum = 0
    # find the path through all points in highManhattan from each point
    distanceList = []
    for eachFood in highManhattan:
        # create a copy of highManhattan
        highManhattanCopy = list(highManhattan)
        # remove current food pellet from the copy
        highManhattanCopy.remove(eachFood)
        currentPos = eachFood
        # find the shortest path based on manhattan distance
        currentSum = 0
        while(len(highManhattanCopy) > 0):
            nextCost, nextTarget = findMinManhattan(currentPos, highManhattanCopy)
            currentSum += nextCost
            currentPos = nextTarget
            highManhattanCopy.remove(nextTarget)
        distanceList.append(currentSum)
    # manhattan sum is the min of all different paths in distanceList    
    if (len(distanceList) > 0):
        manhattanSum = min(distanceList)
    return max(manhattanSum, maxManhattan)

def findMinManhattan(position, points):
    """
    Helper method to find the min manhattan distance from position
    to a list of points
    """
    if (len(points) == 0):
        return (0, None)
    minManhattan = float('inf')
    minPoint = None
    for eachPoint in points:
        dist = util.manhattanDistance(position, eachPoint)
        if (dist < minManhattan):
            minPoint = eachPoint
            minManhattan = dist
    return (minManhattan, minPoint)

def findMaxManhattan(position, points):
    """
    Helper method to find the max manhattan distance from position
    to a list of points
    """
    if (len(points) == 0):
        return (0, None)
    maxManhattan = -1
    maxPoint = None
    for eachPoint in points:
        dist = util.manhattanDistance(position, eachPoint)
        if (dist > maxManhattan):
            maxPoint = eachPoint
            maxManhattan = dist
    return (maxManhattan, maxPoint)

# Abbreviation
better = betterEvaluationFunction

