"""
THIS  CODE  WAS MY OWN WORK,
IT WAS  WRITTEN  WITHOUT  CONSULTING  ANY SOURCES  OUTSIDE  OF  THOSE  APPROVED  BY THE  INSTRUCTOR.
Anthony Wong
"""

# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)

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
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.

        The two main things of concern is how much food is on the board and
        proximity to ghosts. This function should have two parts
        1. find the distance to the ghosts
        2. find if the position moves the ghost closer to the furthest food piece

        return the distance from the ghosts + the num of food left on the board8
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor)
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        """
        Greedy algorithm which minimizes distance to the closest food pellet
        Doesn't really care about ghosts unless it is about to be eaten
        """

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()
        ghostStates = successorGameState.getGhostPositions()

        #finds the minimum food distance which is the closest one
        closestFoodDistance = float("inf")
        for food in newFood:
            distanceFromFood = util.manhattanDistance(newPos, food)
            if (distanceFromFood < closestFoodDistance):
                closestFoodDistance = distanceFromFood

        closestFood = 1/float(closestFoodDistance)

        #finds the closest ghost distance
        #uses "inf" to prevent it from getting eaten by ghosts
        ghostDistances = 1
        for locations in ghostStates:
            distanceToGhost = util.manhattanDistance(newPos, locations)
            if distanceToGhost <= 1:
                ghostDistances = ghostDistances + float("inf")
            else:
                ghostDistances = ghostDistances + (1/distanceToGhost)

        #print(successorGameState.getScore())
        return successorGameState.getScore() + closestFood - ghostDistances

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

        Tried to replicate the pseudocode from the textbook.
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

        return self.miniMaxDecision(gameState, 0, 0)



    def miniMaxDecision(self, gameState, agent, depth): #is the decision state which will make recursive max/min calls
        if agent >= gameState.getNumAgents(): #resets to pacman once all ghosts have been visisted
            agent = 0
            depth = depth + 1

        if depth == self.depth: #finishes at the pre-determined depth
            return self.evaluationFunction(gameState)

        if gameState.isLose() or gameState.isWin(): #exits once game is won or lost
            return self.evaluationFunction(gameState)

        if agent == 0: #for the case of pacman
            return self.maxValue(gameState, agent, depth)

        else: #for the case of ghosts
            return self.minValue(gameState, agent, depth)

        #function MAX-VALUE(state) returns a utility value
    def maxValue(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin(): #equivilant of if TERMINAL-TEST(state) then return UTILITY(state) from textbook
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if depth != 0: #non-base case. Still needs to go down the tree so don't return an action
            v = float("-inf")

            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                #v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                nextState = gameState.generateSuccessor(agent, actions)
                if self.miniMaxDecision(nextState, agent + 1, depth) > v:
                    v = self.miniMaxDecision(nextState, agent + 1, depth)

            return v
        else: #base case. Need to return an action here
            bestAction = Directions.STOP
            v = float("-inf")

            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                # v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                #finds the max value and stores the related action
                nextState = gameState.generateSuccessor(agent, actions)
                if self.miniMaxDecision(nextState, agent + 1, depth) > v:
                    v = self.miniMaxDecision(nextState, agent + 1, depth)
                    bestAction = actions

            return bestAction #action that is returned

    def minValue(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        v = float("inf")

        #finds the min value
        for actions in gameState.getLegalActions(agent):
            if actions == Directions.STOP:
                continue

            #stores the min value
            nextState = gameState.generateSuccessor(agent, actions)
            if self.miniMaxDecision(nextState, agent + 1, depth) < v:
                v = self.miniMaxDecision(nextState, agent + 1, depth)

        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          #Returns the expectimax action using self.depth and self.evaluationFunction

          #All ghosts should be modeled as choosing uniformly at random from their
          #legal moves.

          The same as the code above, only distinction is adding in the alpha and the beta
          Tried to replicate from textbook
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        #use inf as default values because it is auto the biggest and smallest values
        return self.alphaBeta(gameState, 0, 0, alpha, beta)



    def alphaBeta(self, gameState, agent, depth, alpha, beta): #is the decision state which will make recursive max/min calls
        if agent >= gameState.getNumAgents(): #resets to pacman once all ghosts have been visisted
            agent = 0
            depth = depth + 1

        if depth == self.depth: #finishes at the pre-determined depth
            return self.evaluationFunction(gameState)

        if gameState.isLose() or gameState.isWin(): #exits once game is won or lost
            return self.evaluationFunction(gameState)

        if agent == self.index: #for the case of pacman
            return self.maxValue(gameState, agent, depth, alpha, beta)

        else: #for the case of ghosts
            return self.minValue(gameState, agent, depth, alpha, beta)

        #function MAX-VALUE(state) returns a utility value
    def maxValue(self, gameState, agent, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin(): #equivilant of if TERMINAL-TEST(state) then return UTILITY(state) from textbook
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if depth != 0: #non-base case. Still needs to go down the tree so don't return an action
            v = float("-inf")

            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                #v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                nextState = gameState.generateSuccessor(agent, actions)
                if self.alphaBeta(nextState, agent + 1, depth, alpha, beta) > v:
                    v = self.alphaBeta(nextState, agent + 1, depth, alpha, beta)
                    """
                    if v >= beta then return v
                    alpha <- max(alpha, v)
                    """
                    if v > beta: #prunes the alpha and beta values here. value is maxed out so just return it
                        return v

                    if v > alpha:
                        alpha = v

            return v

        else: #base case. Need to return an action here
            v = float("-inf")
            bestMove = Directions.STOP

            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                #v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                nextState = gameState.generateSuccessor(agent, actions)
                if self.alphaBeta(nextState, agent + 1, depth, alpha, beta) > v:
                    v = self.alphaBeta(nextState, agent + 1, depth, alpha, beta)
                    bestMove = actions
                    """
                    if v >= beta then return v
                    alpha <- max(alpha, v)
                    """
                    if v > beta: #prunes the alpha and beta values here. value is maxed out so just return it
                        return v

                    if v > alpha:
                        alpha = v

            return bestMove

    def minValue(self, gameState, agent, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        v = float("inf")

        for actions in gameState.getLegalActions(agent):
            if actions == Directions.STOP:
                continue

            nextState = gameState.generateSuccessor(agent, actions)
            if self.alphaBeta(nextState, agent + 1, depth, alpha, beta) < v:
                v = self.alphaBeta(nextState, agent + 1, depth, alpha, beta)

            """
            if v <= alpha then return v
            beta <- mina(beta, v)
            """
            if v < alpha: #prunes the alpha and beta values here. value is minned out so just return it
                return v

            if v < beta:
                beta = v

        return v



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
      Code is the same as the miniMax function except the min returns the average of the values
    """

    def expectiMax(self, gameState, agent, depth): # is the decision state which will make recursive max/min calls
        if agent >= gameState.getNumAgents(): # resets to pacman once all ghosts have been visisted
            agent = 0
            depth = depth + 1

        if depth == self.depth:  # finishes at the pre-determined depth
            return self.evaluationFunction(gameState)

        if gameState.isLose() or gameState.isWin():  # exits once game is won or lost
            return self.evaluationFunction(gameState)

        if agent == 0:  # for the case of pacman
            return self.maxValue(gameState, agent, depth)

        else:  # for the case of ghosts
            return self.minValue(gameState, agent, depth)

        # function MAX-VALUE(state) returns a utility value
        # max value is the same as minimax because the pacman still needs to act optimally

    def maxValue(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin():  # equivilant of if TERMINAL-TEST(state) then return UTILITY(state) from textbook
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if depth != 0: #non-base case. Still needs to go down the tree so don't return an action
            v = float("-inf")
            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                # v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                nextState = gameState.generateSuccessor(agent, actions)
                if self.expectiMax(nextState, agent + 1, depth) > v:
                    v = self.expectiMax(nextState, agent + 1, depth)
                    bestMove = actions
            return v
        else: # base case. need to return an action here
            bestMove = Directions.STOP
            v = float("-inf")

            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                # v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                nextState = gameState.generateSuccessor(agent, actions)
                if self.expectiMax(nextState, agent + 1, depth) > v:
                    v = self.expectiMax(nextState, agent + 1, depth)
                    bestMove = actions

            return bestMove

    #same as minimax but returns average of the values
    def minValue(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        sum = 0
        numMoves = 0 #have numMoves because don't know how many legal moves there are
        for actions in gameState.getLegalActions(agent):
            if actions == Directions.STOP:
                continue

            numMoves = numMoves + 1
            nextState = gameState.generateSuccessor(agent, actions)
            sum = sum + self.expectiMax(nextState, agent + 1, depth) #sums up the values in order to find avg

        average = sum/numMoves #gets the avg of the values

        return average

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.expectiMax(gameState, 0, 0)

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

        currentFood = currentGameState.getFood() #food available from current state
        #print('currentFood', currentFood) prints food location in form of coordinates if using .asList(), otherwise tuples

        ghostStates = currentGameState.getGhostPositions() #where ghosts are currently


        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        #print('currentCapsules', currentCapsules) #prints how many capsules theinere are as a list


      DESCRIPTION: Similar to the first function but add in capsules
      1. minimize distance to the nearest food pellet
      2. Don't be paranoid about ghosts until they're about to eat you
      3. Subtract the number of capsules left. If there's fewer capsules then that's better because pacman has been
      invulnerable for longer

    """
    "*** YOUR CODE HERE ***"

    currentPacmanState = currentGameState.getPacmanPosition() #where pacman is currently
    ghostPosition = currentGameState.getGhostPositions() #where ghosts are currently
    currentFood = currentGameState.getFood()  # food available from current state
    currentCapsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghosts.scaredTimer for ghosts in ghostStates]


    closestFoodDistance = float("inf")
    for food in currentFood:
        distanceFromFood = util.manhattanDistance(currentPacmanState, food)
        if (distanceFromFood < closestFoodDistance):
            closestFoodDistance = distanceFromFood

    closestFood = 1 / float(closestFoodDistance)

    ghostDistances = float("inf")
    for locations in ghostPosition:
        distanceToGhost = util.manhattanDistance(currentPacmanState, locations)
        if distanceToGhost <= 1:
            ghostDistances = ghostDistances + float("-inf")
        elif distanceToGhost < ghostDistances:
            ghostDistances = (1 / distanceToGhost)

    numCapsulesLeft = len(currentCapsules)


    """
    capsuleScore = 0
    if numCapsulesLeft != 0:
        capsuleScore = 1/(float(numCapsulesLeft))


    if currentPacmanState in currentCapsules:
        capsuleScore = capsuleScore * -1

    for times in scaredTimes:
        if times > 0:
            ghostDistances = 2

    for ghosts in ghostStates:
        if ghosts.scaredTimer > 0 and currentPacmanState in ghostPosition:
            ghostDistances = ghostDistances + 1
    """


    #thrashing can't decide what to do because all options look the same identify WHERE there is thrashing and how it relates
    #to the defined function. nudge in a direction. equidistant from two food pellets then what would pacman do?
    #too careful around ghosts. check distance from ghosts

    # print(successorGameState.getScore())
    return currentGameState.getScore() + ghostDistances + closestFood - numCapsulesLeft


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
      combine alpha beta with the better evaluation function
    """

    def getAction(self, gameState):
        """
          #Returns the expectimax action using self.depth and self.evaluationFunction

          #All ghosts should be modeled as choosing uniformly at random from their
          #legal moves.
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        return self.alphaBeta(gameState, 0, 0, alpha, beta)



    def alphaBeta(self, gameState, agent, depth, alpha, beta): #is the decision state which will make recursive max/min calls
        if agent >= gameState.getNumAgents(): #resets to pacman once all ghosts have been visisted
            agent = 0
            depth = depth + 1

        if depth == self.depth: #finishes at the pre-determined depth
            return self.evaluationFunction(gameState)

        if gameState.isLose() or gameState.isWin(): #exits once game is won or lost
            return self.evaluationFunction(gameState)

        if agent == self.index: #for the case of pacman
            return self.maxValue(gameState, agent, depth, alpha, beta)

        else: #for the case of ghosts
            return self.minValue(gameState, agent, depth, alpha, beta)

        #function MAX-VALUE(state) returns a utility value
    def maxValue(self, gameState, agent, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin(): #equivilant of if TERMINAL-TEST(state) then return UTILITY(state) from textbook
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if depth != 0:
            v = float("-inf")


            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                #v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                nextState = gameState.generateSuccessor(agent, actions)
                if self.alphaBeta(nextState, agent + 1, depth, alpha, beta) > v:
                    v = self.alphaBeta(nextState, agent + 1, depth, alpha, beta)
                    """
                    if v >= beta then return v
                    alpha <- max(alpha, v)
                    """
                    if v > beta: #and depth != 0:
                        return v

                    if v > alpha:
                        alpha = v

            return v

        else:
            v = float("-inf")
            bestMove = Directions.STOP

            for actions in gameState.getLegalActions(agent):
                if actions == Directions.STOP:
                    continue
                #v <- MAX(v, MIN-VALUE(RESULT(s, a)))
                nextState = gameState.generateSuccessor(agent, actions)
                if self.alphaBeta(nextState, agent + 1, depth, alpha, beta) > v:
                    v = self.alphaBeta(nextState, agent + 1, depth, alpha, beta)
                    bestMove = actions
                    """
                    if v >= beta then return v
                    alpha <- max(alpha, v)
                    """
                    if v > beta: #and depth != 0:
                        return v

                    if v > alpha:
                        alpha = v

            return bestMove

    def minValue(self, gameState, agent, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        v = float("inf")

        for actions in gameState.getLegalActions(agent):
            if actions == Directions.STOP:
                continue

            nextState = gameState.generateSuccessor(agent, actions)
            if self.alphaBeta(nextState, agent + 1, depth, alpha, beta) < v:
                v = self.alphaBeta(nextState, agent + 1, depth, alpha, beta)


            """
            if v <= alpha then return v
            beta <- mina(beta, v)
            """

            if v < alpha:
                return v

            if v < beta:
                beta = v

        return v
