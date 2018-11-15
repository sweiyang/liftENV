import copy
import numpy as np
import math
import sys
import random

# initializing hyper parameters
from Obj import *

# Global constants [UPPER CASE]
REWARD_SCALAR = 0.000000000001
NUM_FLOORS = 12
NUM_LIFTS = 2
ACTION_SPACE_n = pow(3, NUM_LIFTS)
LIFT_STARTING_FLOOR = 0
DAY_HOURS = 24
DAY_MINUTES = 1440
NUM_EPISODES = 1000
MAX_PERSON = 10

# Global variables [Camel Case]
timestepMinutes = 0
timestepHours = 0

# initializing action space
action_space = np.zeros(shape=(ACTION_SPACE_n, NUM_LIFTS), dtype=int)

# Pre-compute power array
pow_array = np.zeros(NUM_LIFTS, dtype=int)
for index in range(pow_array.size):
    pow_array[index] = int(pow(3, NUM_LIFTS - index - 1))

# Initialize elevator action space
for action_index in range(ACTION_SPACE_n):
    cumulativeSum = action_index
    for lift_index in range(NUM_LIFTS):
        lift_index_pow = pow_array[lift_index]
        if cumulativeSum >= lift_index_pow * 2:
            action_space[action_index][lift_index] = 2
            cumulativeSum -= lift_index_pow * 2
        elif cumulativeSum >= lift_index_pow:
            action_space[action_index][lift_index] = 1
            cumulativeSum -= lift_index_pow

print(action_space)

# initializing observational space
'''
rows = number of floors
1st column determins the button of the floor that is pressed
subsequent columns are the num of lifts. 0 denoting that the lift is not at that floor and 1 denoting that the lift is at that floor
'''
def printObsSpace():
    for n in range(NUM_FLOORS):
        print(observation_space[NUM_FLOORS - n - 1])


observation_space = np.zeros(shape=(NUM_FLOORS, 1 + NUM_LIFTS), dtype=float)
for lift in range(NUM_LIFTS):
    observation_space[LIFT_STARTING_FLOOR][lift + 1] = 1

# creating a 2-D array
buildingsDATA = np.ndarray(shape=(NUM_FLOORS, MAX_PERSON), dtype=Person)
comingBackArray = np.ndarray(shape=(NUM_FLOORS, DAY_HOURS, MAX_PERSON), dtype=Person)
building = Building(buildingsDATA, NUM_LIFTS, NUM_FLOORS, LIFT_STARTING_FLOOR)


def countDestinations(carryingPerson):
    total = 0
    cumulativeDestination = 0
    for i in range(len(carryingPerson)):
        if carryingPerson[i] is not None:
            total += 1
            cumulativeDestination += int(carryingPerson[i].dest)

    if total == 0:
        return 0

    return cumulativeDestination/total


def generatingData():
    comingBackArray = np.ndarray(shape=(NUM_FLOORS, DAY_HOURS, MAX_PERSON), dtype=Person)
    timestepArray = building.buildingArr
    # go out of the house
    data = 5
    for iter in range(data):
        positionHome = np.random.randint(1, NUM_FLOORS)
        # print(positionHome, timestepHours)
        # if there are < MAXPERSON
        x = 0
        while x < MAX_PERSON and timestepArray[positionHome][x] is not None:
            x = x + 1

        if x < MAX_PERSON:
            for n in range(NUM_FLOORS):
                for j in range(10):
                    # including those that are coming home at that timestepHours
                    if comingBackArray[0][timestepHours][j] is not None:
                        person = comingBackArray[n][timestepHours][j]
                        timestepArray[person.dest][j] = person
            if x != MAX_PERSON:
                timestepArray[positionHome][x] = Person(timestepHours, 0, positionHome)
            # print(positionHome, timestepRand, x)
            # print(0, timestepBack, y)

    # including those that are coming home at that timestepHours
    building.buildingArr = timestepArray


def updateObsSpace(building, obs):
    liftArr = building.lifts
    buildingArr = building.buildingArr
    for liftIndex in range(NUM_LIFTS):
        obs[liftArr[liftIndex].prevPos][liftIndex + 1] = 0
        obs[liftArr[liftIndex].position][liftIndex + 1] \
            = countDestinations(liftArr[liftIndex].carryingPerson) + 1
    for x in range(NUM_FLOORS):
        for y in range(MAX_PERSON):
            if buildingArr[x][y] is not None:
                obs[x][0] = 1
                break
            obs[x][0] = 0


def chooseActions(actionIndex, index, building):
    lift = building.lifts
    if (actionIndex == 0):  # going up
        # print('==== Lift ', index, 'Action Taken: Up ====')
        # check whether is the lift position at the top floor
        if lift[index].position != lift[index].topfloor - 1:
            lift[index].prevPos = lift[index].position
            lift[index].position = lift[index].position + 1
            # negative reward for each move of the lift
            carryPassenger(lift[index].position, index, building)
            return -2 + dropPassenger(index, building) + calculateCost(building)
        else:
            return -10

    if (actionIndex == 1):  # going down
        # print('==== Lift ', index, 'Action Taken: Down ====')
        # check whether is the lift at level 1
        if (lift[index].position != 0):
            lift[index].prevPos = lift[index].position
            lift[index].position = lift[index].position - 1
            # negative reward for each move of the lift
            carryPassenger(lift[index].position, index, building)
            return -2 + dropPassenger(index, building) + calculateCost(building)

        else:
            return -10

    if (actionIndex == 2):  # the lift staying put
        # print('==== Lift ', index, 'Action Taken: Stay ====')
        lift[index].prevPos = lift[index].position
        lift[index].position = lift[index].position
        carryPassenger(lift[index].position, index, building)
        return calculateCost(building)


# check whether does that floor has people
def checkFloor(building):
    liftArr = building.lifts
    # print('====Additional Humans Going into the lift suddenly====')
    for x in range(len(liftArr)):
        carryPassenger(liftArr[x].position, x, building)


def carryPassenger(position, liftIndex, building):
    lift = building.lifts
    building_array = building.buildingArr
    personAtThatFloor = building_array[position]
    # print('People at floor ', position, ':', personAtThatFloor)
    # copy the array that the lift is going to carrying into the lift carrying array

    # initialise the building array into none
    for x in range(len(personAtThatFloor)):
        if personAtThatFloor[x] is not None:
            lift[liftIndex].carryingPerson.append(personAtThatFloor[x])
        personAtThatFloor[x] = None
    # print('Passenger in the lift ', index, ': ', lift[index].carryingPerson)
    # print('Resulting People at that floor: ', building_array[position][timestepHours])


def dropPassenger(liftIndex, building):
    lift = building.lifts[liftIndex]
    carrying = lift.carryingPerson
    reward = 0
    groundFloor = False
    if lift.position == 0:
        groundFloor = True
    x = 0
    while x != len(carrying):
        if carrying[x] is not None:
            # reward for completing the trip and also negative reward for the amount of waiting time occured
            # initialtime is in hours, converting it to minutes
            reward = reward + (timestepMinutes - lift.carryingPerson[x].initialTime * 60) * -1
            if carrying[x].dest == lift.position:
                if groundFloor:
                    # generate coming back data
                    if timestepHours < DAY_HOURS:
                        timestepHoursBack = np.random.randint(timestepHours, DAY_HOURS)
                    else:
                        timestepHoursBack = 0
                    j = 0
                    while j < MAX_PERSON and comingBackArray[0][timestepHoursBack][j] is not None:
                        j = j + 1
                    if j < MAX_PERSON:
                        comingBackArray[0][timestepHoursBack][j] = Person(timestepHours, carrying[x].position, 0)
                        # print('=== Person coming back at', timestepHoursBack, '====')
                del carrying[x]
                x = x - 1
        x += 1
    # print("Cost of Waiting time in the lift: ", reward)
    return reward * REWARD_SCALAR


def calculateCost(building):
    buildingArr = building.buildingArr
    reward = 0
    for n in range(NUM_FLOORS):
        personAtThatFloor = buildingArr[n]
        for x in range(len(personAtThatFloor)):
            if personAtThatFloor[x] is not None:
                reward = reward + (timestepMinutes - personAtThatFloor[x].initialTime * 60) * -1

    # print("Cost of waiting time in the building", reward)
    return reward * REWARD_SCALAR


def simulateAction(Agent, action):

    buildingCopy = copy.deepcopy(building)
    observationCopy = copy.deepcopy(observation_space)

    for liftIndex in range(NUM_LIFTS):
        liftAction = action_space[action][liftIndex]
        chooseActions(liftAction, liftIndex, buildingCopy)

    updateObsSpace(buildingCopy, observationCopy)
    Agent.updateFeatureVector(observationCopy)
    return Agent.calculateStateActionValue()



def iterateState(action):
    r = 0
    global timestepMinutes, timestepHours
    timestepMinutes += 1
    timestepHours = math.floor(timestepMinutes / 60)
    # os.system('cls')
    for liftIndex in range(NUM_LIFTS):
        a = action_space[action][liftIndex]
        r = r + chooseActions(a, liftIndex, building)

    updateObsSpace(building, observation_space)
    return r


# Initialize Agent and helper variables
Agent = TD0FFA(NUM_LIFTS, NUM_FLOORS)
WARMUP_TIME = 5000
currentReward = 0
episodeReward = 0
bestReward = -1 * (sys.maxsize - 1)
iterationNum = 0
print("Num episodes: {}".format(NUM_EPISODES))
for e in range(NUM_EPISODES):
    timestepHours = 0
    timestepMinutes = 0
    episodeReward = 0
    while timestepHours < DAY_HOURS:
        # print('==== episode', e, '====')
        # print('====timestep', timestep, '====')
        # print('==== timestepHours', timestepHours, '====')
        if timestepMinutes % 60 == 0:
            # print('=== Generating more HOOMANs ===')
            generatingData()
            updateObsSpace(building, observation_space)
            checkFloor(building)

        # print('Reward received at this timestep: ', r)
        # # print('Accumulative Reward: ', r_All)
        # for liftIndex in range(NUM_LIFTS):
        #     print(liftIndex, ":", "Carrying", building.lifts[liftIndex].carryingPerson)
        # printObsSpace()

        # replace this part with the agent
        '''
        MODEL FREE TD(0) with feature vector function approximation
        '''
        # printObsSpace()

        action = 0

        # Epsilon Greedy Policy Improvement
        if (random.uniform(0, 1) <= Agent.EPSILON):
            action = random.randint(0, ACTION_SPACE_n - 1)
        else:
            bestStateActionValue = -1 * (sys.maxsize - 1)
            bestActionIndex = 0
            for possibleAction in range(ACTION_SPACE_n):
                currentStateActionValue = simulateAction(Agent, possibleAction)
                if (currentStateActionValue > bestStateActionValue):
                    bestStateActionValue = currentStateActionValue
                    bestActionIndex = possibleAction
            action = bestActionIndex

        currentReward = iterateState(action)
        # print("Current Reward: {}".format(currentReward))
        episodeReward += currentReward

        # Update Feature vector
        Agent.updateFeatureVector(observation_space)
        iterationNum += 1
        # Store first state action value and reward
        if Agent.NOT_TAKEN_ACTION:
            # print("Continuing")
            Agent.NOT_TAKEN_ACTION = False
            continue
        elif Agent.WAITING_NEXT_ACTION:
            # print("First Wait")
            Agent.storeFirstStateActionValue(currentReward)
        else:
            # print("Updating")
            Agent.updateWeightVector(currentReward)

        '''
        End of A.I class
        '''

    # Check reward
    if episodeReward > bestReward:
        bestReward = episodeReward

    # Decay Epsilon
    Agent.decayEpsilon()
    print("Finished episode {}: [Total iterations: {}][Episode Reward: {}][Epsilon: {}]"
          .format(e, iterationNum, episodeReward, Agent.EPSILON))

    if e % 100 == 0:
        print("Feature Vector: {}".format(len(Agent.featureVector)))
        print(Agent.featureVector)
        print("Weight Vector: {}".format(len(Agent.weightVector)))
        print(Agent.weightVector)



print("Total iterations: {}".format(iterationNum))
print("Best reward: {}".format(bestReward))
print("Final Weight Vector:")
print(Agent.weightVector)