import numpy as np
import random
import math
import os
# initializing hyper parameters
from Obj import Person, Building

NUM_FLOORS = 12
NUM_LIFTS = 2
ACTION_SPACE_n = pow(3, NUM_LIFTS)
timestep = 0
timestepHours = 0
MAX_PERSON = 10
NUM_EPISODES = 1

action_space = np.ndarray(shape=(ACTION_SPACE_n, NUM_LIFTS), dtype=int)
# initializing action space
x = 0
for i in range(3):
    for j in range(3):
        action_space[x][0] = i
        action_space[x][1] = j
        x = x + 1

# print(action_space)

# initializing observational space
'''
rows = number of floors
1st column determins the button of the floor that is pressed
subsequent columns are the num of lifts. 0 denoting that the lift is not at that floor and 1 denoting that the lift is at that floor
'''
observation_space = np.zeros(shape=(NUM_FLOORS, 1 + NUM_LIFTS), dtype=int)
observation_space[0][1] = 1
observation_space[0][2] = 1

# creating a 2-D array
buildingsDATA = np.ndarray(shape=(NUM_FLOORS, MAX_PERSON), dtype=Person)
comingBackArray = np.ndarray(shape=(NUM_FLOORS, 24, MAX_PERSON), dtype=Person)
building = Building(buildingsDATA, NUM_LIFTS, NUM_FLOORS, 0)


def generatingData():
    timestepArray = building.buildingArr
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
                        person = comingBackArray[0][timestepHours][j]
                        timestepArray[person.dest][j] = person
            if x != MAX_PERSON:
                timestepArray[positionHome][x] = Person(timestepHours, 0, positionHome)
            # print(positionHome, timestepRand, x)
            # print(0, timestepBack, y)
        data += 1

    #including those that are coming home at that timestepHours
    building.buildingArr = timestepArray


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
    print('====Additional Humans Going into the lift suddenly====')
    for x in range(len(liftArr)):
        carryPassenger(liftArr[x].position, x, building)


def carryPassenger(position, index, building):
    lift = building.lifts
    building_array = building.buildingArr
    personAtThatFloor = building_array[position]
    print('People at floor ', position , ':', personAtThatFloor)
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
                        print('=== Person coming back at', timestepHoursBack,  '====')
                del carrying[x]
                x = x - 1
        x += 1
    print("Cost of Waiting time in the lift: ", reward)
    return reward


def calculateCost(building):
    buildingArr = building.buildingArr
    reward = 0
    for n in range(NUM_FLOORS):
        personAtThatFloor = buildingArr[n]
        for x in range(len(personAtThatFloor)):
            if personAtThatFloor[x] is not None:
                reward = reward + (timestep - personAtThatFloor[x].initialTime * 60) * -1

    print("Cost of waiting time in the building", reward)

    return reward

def printObsSpace():
    for n in range(NUM_FLOORS):
        print(observation_space[NUM_FLOORS - n - 1])
        
os.system('cls')
for e in range(NUM_EPISODES):
    r_All = 0   
    # reintialize you can remove the top part if you want 
    buildingsDATA = np.ndarray(shape=(NUM_FLOORS, MAX_PERSON), dtype=Person)
    comingBackArray = np.ndarray(shape=(NUM_FLOORS, 24, MAX_PERSON), dtype=Person)
    building = Building(buildingsDATA, NUM_LIFTS, NUM_FLOORS, 0)
    while timestepHours != 24:


        r = 0
        print('==== episode', e, '====')
        print('====timestep', timestep, '====')
        print('==== timestepHours', timestepHours, '====')
        if timestep % 60 == 0:
            #print('=== Generating more HOOMANs ===')
            generatingData()
            createObsSpace(timestepHours, building)
            checkFloor(building)

        print('Reward received at this timestep: ', r)
        print('Accumulative Reward: ', r_All)
        for liftIndex in range(NUM_LIFTS):
            print(liftIndex, ":", "Carrying", building.lifts[liftIndex].carryingPerson)
        printObsSpace()

        # replace this part with the agent
        '''
        include your agent class here
        '''
        action = int(input())
        timestep = timestep + 1
        timestepHours = math.floor(timestep / 60)
        os.system('cls')
        once = True
        for liftIndex in range(NUM_LIFTS):
            a = action_space[action][liftIndex]
            if once:
                r = r + chooseActions(a, liftIndex, building)
            else:
                chooseActions(a, liftIndex, building)
            once = False
            createObsSpace(timestepHours, building)

        createObsSpace(timestepHours, building)
        r_All = r_All + r
        # print(building.buildingArr)
