# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    
    actions = []
    start = problem.getStartState()
    # visited stores the location
    visited = set()
    stack = util.Stack()
    # add the start node to the stack and mark it as visited
    startNode = (start, [])
    stack.push(startNode)
    while (not stack.isEmpty()):
        # pop from the stack
        currentNode = stack.pop()
        currentState = currentNode[0]
        lastActions = currentNode[1]
        # check if it is the goal
        if (problem.isGoalState(currentState)):
            actions = lastActions
            break
        else:
            if (not currentState in visited):
                visited.add(currentState)
            successors = problem.getSuccessors(currentState)
            for each in successors:
                if (not each[0] in visited):
                    nextActions = list(lastActions)
                    nextActions.append(each[1])
                    nextNode = (each[0], nextActions) 
                    stack.push(nextNode)
    return actions



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actions = []
    start = problem.getStartState()
    # visited stores the location
    visited = set()
    queue = util.Queue()
    # add the start node to the stack and mark it as visited
    # note: for BFS we need to mark it visited before put it into a queue
    startNode = (start, [])
    visited.add(start)
    queue.push(startNode)
    while (not queue.isEmpty()):
        # pop from the stack
        currentNode = queue.pop()
        currentState = currentNode[0]
        lastActions = currentNode[1]
        # check if it is the goal
        if (problem.isGoalState(currentState)):
            actions = lastActions
            break
        else:
            successors = problem.getSuccessors(currentState)
            for each in successors:
                if (not each[0] in visited):
                    visited.add(each[0])
                    nextActions = list(lastActions)
                    nextActions.append(each[1])
                    nextNode = (each[0], nextActions) 
                    queue.push(nextNode)
    return actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    actions = []
    start = problem.getStartState()
    # visited stores the location
    visited = set()
    queue = util.PriorityQueue()
    # add the start node to the stack and mark it as visited
    # startNode is a tuple (current loc, list of actions, total cost)
    startNode = (start, [], 0)
    #visited.add(start)
    # for uniform Cost Search, we should only mark it visited when we process
    # the node (make sure that the path to that node is minimum)
    queue.push(startNode, 0)
    while (not queue.isEmpty()):
        # pop from the stack (need to check for visited)
        popSuccess = False
        currentNode = None
        while (not popSuccess) and (not queue.isEmpty()):
            currentNode = queue.pop()
            if (not currentNode[0] in visited):
                popSuccess = True
        if (currentNode == None):
            break
        currentState = currentNode[0]
        lastActions = currentNode[1]
        currentCost = currentNode[2]
        # check if it is the goal
        if (problem.isGoalState(currentState)):
            actions = lastActions
            break
        else:
            if (not currentState in visited):
                visited.add(currentState)
            successors = problem.getSuccessors(currentState)
            for each in successors:
                # a node is in the visited only after we pop it from the
                # priority queue
                if (not each[0] in visited):
                    nextActions = list(lastActions)
                    nextActions.append(each[1])
                    nextCost = currentCost + each[2] 
                    nextNode = (each[0], nextActions, nextCost) 
                    queue.push(nextNode, nextCost)
                
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    actions = []
    start = problem.getStartState()
    # visited stores the location
    visited = set()
    queue = util.PriorityQueue()
    # add the start node to the stack and mark it as visited
    # startNode is a tuple (current loc, list of actions, total cost)
    startNode = (start, [], 0)
    # for aStar, we should only mark it visited when we process
    # the node (make sure that the path to that node is minimum)
    queue.push(startNode, 0)
    while (not queue.isEmpty()):
        # pop from the stack (need to check for visited)
        popSuccess = False
        currentNode = None
        while (not popSuccess) and (not queue.isEmpty()):
            currentNode = queue.pop()
            if (not currentNode[0] in visited):
                popSuccess = True
        if (currentNode == None):
            break
        currentState = currentNode[0]
        lastActions = currentNode[1]
        currentCost = currentNode[2]
        # check if it is the goal
        if (problem.isGoalState(currentState)):
            actions = lastActions
            break
        else:
            if (not currentState in visited):
                visited.add(currentState)
            successors = problem.getSuccessors(currentState)
            for each in successors:
                # a node is in the visited only after we pop it from the
                # priority queue
                if (not each[0] in visited):
                    nextActions = list(lastActions)
                    nextActions.append(each[1])
                    nextCost = currentCost + each[2]
                    heuristicCompare = heuristic(each[0], problem) + nextCost
                    nextNode = (each[0], nextActions, nextCost) 
                    # push the node to the queue based on the heuristicCompare
                    queue.push(nextNode, heuristicCompare)
                
    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
