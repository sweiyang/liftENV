import numpy as np
import copy

class Person:
    def __init__(self, initialTime, dest, post):
        self.initialTime = initialTime
        self.dest = dest
        self.position = post

    def __repr__(self):
        return "Person" + "==" + str(self.dest) + "=="

    def __str__(self):
        return "Person"


class Lift:
    def __init__(self, startingPos, NUM_FLOORS):
        self.position = startingPos
        self.prevPos = -1
        self.carryingPerson = []
        self.topfloor = NUM_FLOORS


class Building:
    def __init__(self, buildingArr, NUM_LIFTS, NUM_FLOORS, startingPos):
        self.buildingArr = buildingArr
        self.lifts = np.ndarray(shape=NUM_LIFTS, dtype=Lift)
        for x in range(NUM_LIFTS):
            self.lifts[x] = Lift(startingPos, NUM_FLOORS)


class TD0FFA:
    def __init__(self, NUM_LIFTS, NUM_FLOORS):
        self.NOT_TAKEN_ACTION = True
        self.WAITING_NEXT_ACTION = True
        self.LEARNING_RATE = 0.0005
        self.DISCOUNT_RATE = 0.05
        self.EPSILON = 1
        self.EPSILON_DECAY = 0.999
        self.CURR_STATE_ACTION = None
        self.NEXT_REWARD = None
        self.NEXT_STATE_ACTION = None
        self.NUM_FLOORS = NUM_FLOORS
        self.NUM_LIFTS = NUM_LIFTS
        self.ACTION_SPACE = pow(3, NUM_LIFTS)
        self.featureVector = np.zeros((2 * NUM_LIFTS), dtype=float)
        self.weightVector = np.full((2 * NUM_LIFTS), -0.5, dtype=float)


    def calculateStateActionValue(self):
        return np.dot(self.featureVector, self.weightVector)

    def decayEpsilon(self):
        self.EPSILON = self.EPSILON * self.EPSILON_DECAY

    def updateFeatureVector(self, observation_space):
        # print("Before feature vector update")
        # print(self.featureVector)
        # Update floor activations
        # for floor in range(self.NUM_FLOORS):
        #     self.featureVector[floor] = observation_space[floor][0]

        # Update lift positions and carrying capacity
        for lift in range(self.NUM_LIFTS):
            featureVectorIndex =  (2 * lift)
            liftLocation = 0
            for floor in range(self.NUM_FLOORS):
                if observation_space[floor][lift + 1] != 0:
                    liftLocation = floor
                    break

            # print("Lift Location: {}".format(liftLocation))
            # print("Average destination: {}".format(observation_space[liftLocation][lift + 1]))
            # Position, normalized to [0, 1]
            self.featureVector[featureVectorIndex] = liftLocation
            # Average destination of passeners, normalized to [0, 1]
            self.featureVector[featureVectorIndex + 1] = observation_space[liftLocation][lift + 1]

        # print("After Updated vector")
        # print(self.featureVector)

    def storeFirstStateActionValue(self, currentReward):
        # print("Before q(S,A,w): {}\n".format(self.CURR_STATE_ACTION))
        self.NEXT_REWARD = currentReward
        self.CURR_STATE_ACTION = self.calculateStateActionValue()
        self.WAITING_NEXT_ACTION = False
        # print("After q(S,A,w): {}\n".format(self.CURR_STATE_ACTION))

    def updateWeightVector(self, currentReward):
        # print("Before weightVector: {}\n".format(self.weightVector))
        self.NEXT_STATE_ACTION = self.calculateStateActionValue()

        gradient = self.LEARNING_RATE * (self.NEXT_REWARD
                                         + (self.DISCOUNT_RATE * self.NEXT_STATE_ACTION)
                                         - self.CURR_STATE_ACTION)
        # print("[CURR_STATE_ACTION: {}][NEXT_STATE_ACTION: {}][Gradient: {}]".format(self.CURR_STATE_ACTION, self.NEXT_STATE_ACTION, gradient))

        deltaW = copy.deepcopy(self.featureVector)
        for i in range(self.featureVector.size):
            deltaW[i] = deltaW[i] * min(gradient, 50)

        # print("DeltaW")
        # print(deltaW)
        for i in range(self.weightVector.size):
            self.weightVector[i] += deltaW[i]
        # print("After weightVector: {}\n".format(self.weightVector))

        # Update for next iteration
        self.CURR_STATE_ACTION = self.NEXT_STATE_ACTION
        self.NEXT_REWARD = currentReward
