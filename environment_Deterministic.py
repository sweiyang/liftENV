import random

import numpy as np
import math
import os
import sys
import time

# initializing hyper parameters
from Obj import Person, Building

NUM_FLOORS = 12
NUM_LIFTS = 2
ACTION_SPACE_n = pow(3, NUM_LIFTS)
timestep = 0
timestepHours = 0
MAX_PERSON = 10
NUM_EPISODES = 1000


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

# initializing observational space
'''
rows = number of floors
1st column determins the button of the floor that is pressed
subsequent columns are the num of lifts. 0 denoting that the lift is not at that floor and 1 denoting that the lift is at that floor
'''
observation_space = np.zeros(shape=(NUM_FLOORS, 1 + NUM_LIFTS), dtype=int)
for x in range(NUM_LIFTS):
    observation_space[0][x] = 1

# creating a 2-D array
buildingsDATA = np.ndarray(shape=(NUM_FLOORS, MAX_PERSON), dtype=Person)
comingBackArray = np.ndarray(shape=(NUM_FLOORS, 24, MAX_PERSON), dtype=Person)
building = Building(buildingsDATA, NUM_LIFTS, NUM_FLOORS, 0)


def generatingData():
    comingBackArray = np.ndarray(shape=(NUM_FLOORS, 24, MAX_PERSON), dtype=Person)
    timestepArray = building.buildingArr
    personAdded = np.zeros(NUM_FLOORS)
    # go out of the house
    data = 0
    while (data != 5):
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
                if observation_space[positionHome][0] != 1:
                    personAdded[positionHome] = 1
                timestepArray[positionHome][x] = Person(timestepHours, 0, positionHome)

            # print(positionHome, timestepRand, x)
            # print(0, timestepBack, y)
        data += 1

    # including those that are coming home at that timestepHours
    building.buildingArr = timestepArray
    return personAdded


def createObsSpace(timestepHours, buildingObj):
    liftArr = buildingObj.lifts
    buildingArr = buildingObj.buildingArr
    for liftIndex in range(NUM_LIFTS):
        observation_space[liftArr[liftIndex].prevPos][liftIndex + 1] = 0
        observation_space[liftArr[liftIndex].position][liftIndex + 1] = 1
    for x in range(NUM_FLOORS):
        for y in range(MAX_PERSON):
            if buildingArr[x][y] is not None and observation_space[x][0] == 0:
                observation_space[x][0] = 1
                break


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
    dest1 = 0
    dest2 = 0
    # print('====Additional Humans Going into the lift suddenly====')
    for x in range(len(liftArr)):
        carryPassenger(liftArr[x].position, x, building)


def carryPassenger(position, index, building):
    lift = building.lifts
    building_array = building.buildingArr
    personAtThatFloor = building_array[position]
    # print('People at floor ', position, ':', personAtThatFloor)
    # copy the array that the lift is going to carrying into the lift carrying array

    # initialise the building array into none
    for x in range(len(personAtThatFloor)):
        if personAtThatFloor[x] is not None:
            lift[index].carryingPerson.append(personAtThatFloor[x])
        personAtThatFloor[x] = None
    observation_space[position][0] = 0
    # print('Passenger in the lift ', index, ': ', lift[index].carryingPerson)
    # print('Resulting People at that floor: ', building_array[position][timestepHours])


def dropPassenger(index, building):
    lift = building.lifts[index]
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
            reward = reward + (timestep - lift.carryingPerson[x].initialTime * 60) * -1
            if carrying[x].dest == lift.position:
                if groundFloor:
                    # generate coming back data
                    if timestepHours != 24:
                        timestepHoursBack = np.random.randint(timestepHours, 24)
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
    return reward


def calculateCost(building):
    buildingArr = building.buildingArr
    reward = 0
    for n in range(NUM_FLOORS):
        personAtThatFloor = buildingArr[n]
        for x in range(len(personAtThatFloor)):
            if personAtThatFloor[x] is not None:
                reward = reward + (timestep - personAtThatFloor[x].initialTime * 60) * -1

    # print("Cost of waiting time in the building", reward)

    return reward


def printObsSpace():
    for n in range(NUM_FLOORS):
        print(observation_space[NUM_FLOORS - n - 1])


r_All = 0
os.system('cls')
'''
Deterministic Lift variables
'''


def checkButtons(s, lift_1_dest, lift_2_dest, destSector1, destSector2, isComingDown1, isComingDown2):
    for n in range(NUM_FLOORS):
        lift1pos = lift1.position
        lift2pos = lift2.position
        # check from the top if there is a person there
        if s[NUM_FLOORS - 1 - n] == 1:
            dest = NUM_FLOORS - 1 - n
            if dest > 5:
                destSector = 2
            else:
                destSector = 1
                # if both are at ground floor
            if not isComingDown1 and not isComingDown2:
                if lift1pos == 0 and lift2pos == 0:
                    if lift_1_dest < dest:
                        lift_1_dest = dest
                        destSector1 = destSector
                # if only one is at ground floor
                elif lift1pos == 0:
                    if destSector2 != destSector:
                        if lift_1_dest < dest:
                            lift_1_dest = dest
                            destSector1 = destSector
                    else:
                        if dest > lift_2_dest:
                            lift_2_dest = dest
                elif lift2pos == 0:
                    if destSector1 != destSector:
                        if lift_2_dest < dest:
                            lift_2_dest = dest
                            destSector2 = destSector
                    else:
                        if dest > lift_1_dest:
                            lift_1_dest = dest
                # if both of them are moving
                else:
                    if destSector == destSector2:
                        if dest < lift_2_dest:
                            lift_2_dest = dest
                    elif destSector == destSector1:
                        if dest < lift_1_dest:
                            lift_1_dest = dest

                    # if destSector2 == destSector1 and dest == destSector2:
                    #     if lift_1_dest < lift_2_dest < dest:
                    #         lift_2_dest = dest
                    #     elif lift_2_dest < lift_1_dest < dest:
                    #         lift_1_dest = dest
                    # elif dest == destSector2:
                    #     if lift_2_dest < dest:
                    #         lift_2_dest = dest
                    # elif dest == destSector1:
                    #     if lift_1_dest < dest:
                    #         lift_1_dest = dest
                    # # destination sector is not in both lift sector
                    # else:
                    #     if lift_1_dest < lift_2_dest < dest:
                    #         lift_2_dest = dest
                    #         destSector2 = destSector
                    #     elif lift_2_dest < lift_1_dest < dest:
                    #         lift_1_dest = dest
                    #         destSector1 = dest
            elif isComingDown2:

                # if only one is at ground floor
                if lift2pos < dest:
                    if lift_1_dest < dest:
                        lift_1_dest = dest
                        destSector1 = dest
            elif isComingDown1:
                if lift1pos < dest:
                    if lift_2_dest < dest:
                        lift_2_dest = dest
                        destSector2 = dest

    return lift_1_dest, lift_2_dest, destSector1, destSector2, isComingDown1, isComingDown2

r_array = []
for e in range(NUM_EPISODES):
    observation_space = np.zeros(shape=(NUM_FLOORS, 1 + NUM_LIFTS), dtype=int)
    for x in range(NUM_LIFTS):
        observation_space[0][x] = 1

    # creating a 2-D array
    buildingsDATA = np.ndarray(shape=(NUM_FLOORS, MAX_PERSON), dtype=Person)
    comingBackArray = np.ndarray(shape=(NUM_FLOORS, 24, MAX_PERSON), dtype=Person)
    building = Building(buildingsDATA, NUM_LIFTS, NUM_FLOORS, 0)
    timestep = 0
    timestepHours = 0
    r_All = 0
    isComingDown1 = False
    isComingDown2 = False
    lift_1_dest = -1
    lift_2_dest = -1
    destSector1 = 0
    destSector2 = 0

    liftsArr = building.lifts
    lift1 = liftsArr[0]
    lift2 = liftsArr[1]
    while timestepHours != 24:
        dest1 = 0
        r = 0

        # print('==== episode', e, '====')
        # print('====timestep', timestep, '====')
        # print('==== timestepHours', timestepHours, '====')
        if timestep % 10 == 0:
            # print('=== Generating more HOOMANs ===')
            s = generatingData()
            createObsSpace(timestepHours, building)
            checkFloor(building)

            if lift1.position == 0 or (lift2.position == 0 and lift1.position == 0):
                for n in range(len(lift1.carryingPerson)):
                    if dest1 < lift1.carryingPerson[n].positionHome:
                        dest1 = lift1.carryingPerson[n].positionHome
            elif lift2.position == 0:
                for n in range(len(lift2.carryingPerson)):
                    if dest1 < lift2.carryingPerson[n].positionHome:
                        dest1 = lift2.carryingPerson[n].positionHome
            lift_1_dest, lift_2_dest, destSector1, destSector2, isComingDown1, isComingDown2 = checkButtons(s,
                                                                                                            lift_1_dest,
                                                                                                            lift_2_dest,
                                                                                                            destSector1,
                                                                                                            destSector2,
                                                                                                            isComingDown1,
                                                                                                            isComingDown2)


        # # print('Reward received at this timestep: ', r)
        # # print('Accumulative Reward: ', r_All)
        # for liftIndex in range(NUM_LIFTS):
        #     print(liftIndex, ":", "Carrying", building.lifts[liftIndex].carryingPerson)
        # printObsSpace()
        # replace this part with the agent
        '''
        implement your A.I here
        '''
        # print('lift 1 Destination:', lift_1_dest)
        # print('lift 2 Destination:', lift_2_dest)

        if lift1.position < lift_1_dest:

            if lift2.position < lift_2_dest:
                chooseActions(0, 1, building)
                if lift2.position == lift_2_dest:
                    lift_2_dest = 0
                    isComingDown2 = True
            elif lift2.position > lift_2_dest:
                chooseActions(1, 1, building)
                if lift2.position == lift_2_dest:
                    lift_2_dest = 0
                    isComingDown2 = True
            r = r + chooseActions(0, 0, building)
            if lift1.position == lift_1_dest:
                lift_1_dest = 0
                isComingDown1 = True

        elif lift1.position > lift_1_dest:

            if lift2.position < lift_2_dest:
                chooseActions(0, 1, building)
                if lift2.position == lift_2_dest:
                    lift_2_dest = 0
                    isComingDown2 = True
            elif lift2.position > lift_2_dest:
                chooseActions(1, 1, building)
                if lift2.position == lift_2_dest:
                    lift_2_dest = 0
                    isComingDown2 = True
            r = r + chooseActions(1, 0, building)
            if lift1.position == lift_1_dest:
                lift_1_dest = 0
                isComingDown1 = True
        else:
            if lift2.position < lift_2_dest:
                r = r + chooseActions(0, 1, building)
                if lift2.position == lift_2_dest:
                    lift_2_dest = 0
                    isComingDown2 = True
            elif lift2.position > lift_2_dest:
                r = r + chooseActions(1, 1, building)
                if lift2.position == lift_2_dest:
                    lift_2_dest = 0
                    isComingDown2 = True

        if lift1.position == 0:
            isComingDown1 = False
        if lift2.position == 0:
            isComingDown2 = False
        '''
        End of A.I class
        '''

        timestep = timestep + 1
        timestepHours = math.floor(timestep / 60)

        createObsSpace(timestepHours, building)
        r_All = r_All + r
        sys.stdout.write("\r{0}".format("=" * timestepHours, e) + "[episode " + str(e) + "]")
    r_array.append(r_All)
    sys.stdout.write("[reward:" + str(r_All) + "]\n")

np.savetxt("data.csv", r_array, delimiter=",")
