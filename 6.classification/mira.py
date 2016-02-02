# mira.py
# -------
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


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        # print things out to see
        #print Cgrid
        #print self.max_iterations
        # store the results of validating the weights
        #results = [0 for x in range(len(Cgrid))]
        #print 'test'
        #feature = trainingData[0]
        #print feature * feature
        #feature2 = trainingData[1]
        #print (feature - feature2) * feature
        #return
        # store all the weights in a list
        weightsList = []
        for eachC in Cgrid:
            print "Test with C = ", eachC
            # initialize new weights
            currentWeights = {}
            for label in self.legalLabels:
                currentWeights[label] = util.Counter() # this is the data-structure you should use
            # start training
            for iteration in range(self.max_iterations):
                print "Iteration ", iteration
                for i in range(len(trainingData)):
                    # find the max score
                    maxScore = -float("inf")
                    maxLabel = -1
                    for j in range(len(currentWeights)):
                        score = currentWeights[j] * trainingData[i]
                        if (score > maxScore):
                            maxScore = score
                            maxLabel = j
                    # update the weights accordingly
                    correctLabel = trainingLabels[i]
                    if (maxLabel != correctLabel):
                        weightDiff = currentWeights[maxLabel] - currentWeights[correctLabel]
                        feature = trainingData[i]
                        thao = ((weightDiff * feature) + 1.0) / (2 * (feature * feature)) # careful with arithmetic
                        step = min(eachC, thao)
                        # modify weight vectors
                        change = self.scaleVector(feature, step)
                        currentWeights[maxLabel] -= change
                        currentWeights[correctLabel] += change
            # append currentWeights to weightsList
            weightsList.append(currentWeights)
        # find bestC
        bestCidx = -1
        correctGuesses = -float("inf")
        for i in range(len(weightsList)):
            # generate guesses
            weights = weightsList[i]
            guesses = self.classifyWithWeights(weights, validationData)
            numCorrect = 0
            for idx in range(len(guesses)):
                if (guesses[idx] == validationLabels[idx]):
                     numCorrect += 1
            # update bestC
            if (numCorrect > correctGuesses):
                correctGuesses = numCorrect
                bestCidx = i
            elif (numCorrect == correctGuesses):
                if (Cgrid[i] < Cgrid[bestCidx]):
                    bestCidx = i
        # update self.weights
        self.weights = weightsList[i]

    def scaleVector(self, feature, step):
        newFeature = util.Counter()
        for each in feature:
            newFeature[each] = feature[each] * step
        return newFeature

    def classifyWithWeights(self, weights, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
