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
        self.LEARNING_RATE = 0.000000005
        self.DISCOUNT_RATE = 0.000005
        self.EPSILON = 1
        self.EPSILON_DECAY = 0.999
        self.CURR_STATE_ACTION = None
        self.NEXT_REWARD = None
        self.NEXT_STATE_ACTION = None
        self.NUM_FLOORS = NUM_FLOORS
        self.NUM_LIFTS = NUM_LIFTS
        self.ACTION_SPACE = pow(3, NUM_LIFTS)
        self.featureVector = np.zeros((2 * NUM_LIFTS) + NUM_FLOORS + 1, dtype=float)
        self.weightVector = np.full((2 * NUM_LIFTS) + NUM_FLOORS + 1, -0.5, dtype=float)


    def calculateStateActionValue(self):
        return np.dot(self.featureVector, self.weightVector)

    def decayEpsilon(self):
        self.EPSILON = self.EPSILON * self.EPSILON_DECAY

    def updateFeatureVector(self, clonedBuilding):
        # print("Before feature vector update")
        # print(self.featureVector)

        """
        Features per elevator:
            1. Number of people going up vs going down
            2. Geometric distance to destination of passengers

        Feature per floor:
            1. Minimum distance to request per floor
                -> Lift distance to request = (distance * number of passengers dest in opposite direction)

        Global features:
            1. Average distance between lifts
        """

        requestArray = clonedBuilding.buildingArr
        liftArray = clonedBuilding.lifts

        averageDistBetweenLift = 0
        pairCounter = 0
        for lift in range(self.NUM_LIFTS):
            favouredDirection = 0
            geometricDist = 0
            liftPosition = liftArray[lift].position
            passengers = liftArray[lift].carryingPerson
            for p in range(len(passengers)):
                if passengers[p] is not None:
                    pDest = passengers[p].dest
                    if pDest > liftPosition:
                        favouredDirection += 1
                    elif pDest < liftPosition:
                        favouredDirection -= 1

                    euclideanDist = int(abs(liftPosition - pDest))
                    if euclideanDist > 0:
                        geometricDist += (1 - pow(0.5, euclideanDist)) * 2

            featureIndex = (2 * lift)
            self.featureVector[featureIndex] = favouredDirection
            self.featureVector[featureIndex + 1] = geometricDist

            # Calculate average distance pair
            for nextLift in range(lift + 1, self.NUM_LIFTS):
                currDist = abs(liftPosition - liftArray[nextLift].position)
                averageDistBetweenLift = ((pairCounter * averageDistBetweenLift) + currDist)/(pairCounter + 1)
                pairCounter += 1

        self.featureVector[(2 * self.NUM_LIFTS) + self.NUM_FLOORS] = averageDistBetweenLift

        # Minimum distance to request
        for floor in range(self.NUM_FLOORS):
            hasRequest = False
            for r in range(requestArray[0].size):
                if requestArray[floor][r] is not None:
                    hasRequest = True
                    break

            if hasRequest:
                minDist = 1000000
                for lift in range(self.NUM_LIFTS):
                    liftPosition = liftArray[lift].position
                    passengers = liftArray[lift].carryingPerson
                    oppositePassengers = 1
                    for p in range(len(passengers)):
                        if passengers[p] is not None:
                            pDest = passengers[p].dest
                            if (pDest > liftPosition > floor) or (pDest < liftPosition < floor):
                                oppositePassengers += 1

                    currLiftDist = oppositePassengers * abs(floor - liftPosition)
                    if currLiftDist < minDist:
                        minDist = currLiftDist

                self.featureVector[(2 * self.NUM_LIFTS) + floor] = minDist
            else:
                self.featureVector[(2 * self.NUM_LIFTS) + floor] = 0




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
            deltaW[i] = deltaW[i] * gradient

        # print("DeltaW")
        # print(deltaW)
        for i in range(self.weightVector.size):
            self.weightVector[i] += deltaW[i]
        # print("After weightVector: {}\n".format(self.weightVector))

        # Update for next iteration
        self.CURR_STATE_ACTION = self.NEXT_STATE_ACTION
        self.NEXT_REWARD = currentReward
