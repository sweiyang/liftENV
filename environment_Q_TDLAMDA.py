import random
import time
import numpy as np
import math
import os
import sys

# initializing hyper parameters
from Obj import Person, Building
from deterministic_fx import DetAgent

NUM_FLOORS = 20
NUM_LIFTS = 3
ACTION_SPACE_n = pow(3, NUM_LIFTS)
timestep = 0
timestepHours = 0
MAX_PERSON = 20
NUM_EPISODES = 2000
REWARD_SCALE = 1
WARMUP = 90
LAMDA = 5

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
observation_space = np.zeros(shape=(NUM_FLOORS, 1 + (NUM_LIFTS * 2)), dtype=int)
for x in range(NUM_LIFTS):
    observation_space[0][2 * x + 1] = 1

# creating a 2-D array
buildingsDATA = np.ndarray(shape=(NUM_FLOORS, MAX_PERSON), dtype=Person)
comingBackArray = np.ndarray(shape=(NUM_FLOORS, 24, MAX_PERSON), dtype=Person)
building = Building(buildingsDATA, NUM_LIFTS, NUM_FLOORS, 0)

'''
A.I 
'''
# Q_table initialization
Q = np.zeros(shape=(pow(2, 2 * 10), ACTION_SPACE_n), dtype=float)
episode_reward = 0
array_episode_reward = []  # storing rewards per episodes
array_det_reward = []
discount = 0.5
learning_rate = 0.05
epsilon = 1
epsilon_decay = 0.995
epsilon_minimum = 0
r_All = 0
e_array = []  # how has the epsilon decay over the number of episodes

'''
A.I
'''


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

    # including those that are coming home at that timestepHours
    building.buildingArr = timestepArray


def createObsSpace(timestepHours, buildingObj):
    liftArr = buildingObj.lifts
    buildingArr = buildingObj.buildingArr
    for liftIndex in range(NUM_LIFTS):
        observation_space[liftArr[liftIndex].prevPos][2 * liftIndex + 1] = 0
        observation_space[liftArr[liftIndex].position][2 * liftIndex + 1] = 1
        carryingPerson = liftArr[liftIndex].carryingPerson
        for person in range(len(carryingPerson)):
            observation_space[carryingPerson[person].dest][2 * (liftIndex + 1)] = 1
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
                observation_space[lift.position][2 * (index + 1)] = 0
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


def binaryToIntConversion(number, length):
    bin = np.zeros(length, dtype=int)
    num = number
    i = 0
    while num > 0:
        rem = num % 2
        bin[length - 1 - i] = rem
        num = num // 2
        i += 1

    return bin


def lengthOfBinary(number):
    bin = []
    while number > 0:
        rem = number % 2
        bin.append(rem)
        number = number // 2

    return len(bin)


def intToBinaryConversion(binary):
    number = 0
    for i in range(len(binary)):
        if binary[i] == 1:
            number += pow(2, i)

    return number


def convertObsSpaceToInt(obsSpace, building):
    binary = np.zeros(shape=(NUM_FLOORS), dtype=int)
    for floors in range(NUM_FLOORS):
        binary[floors] = obsSpace[floors][0]
    liftArr = building.lifts
    MAX_BINARYLENGTH = lengthOfBinary(NUM_FLOORS)
    for n in range(NUM_LIFTS):
        np.append(binary, binaryToIntConversion(liftArr[n].position, MAX_BINARYLENGTH))
        destArrBin = np.zeros(shape=(NUM_FLOORS), dtype=int)
        for b in range(NUM_FLOORS):
            destArrBin[b] = obsSpace[b][2 * (n + 1)]
        np.append(binary, destArrBin)

    return intToBinaryConversion(binary)

os.system('cls')

for e in range(NUM_EPISODES):
    r_All = 0
    r_Det = 0
    timestep = 0
    timestepHours = 0
    buildingsDATA = np.ndarray(shape=(NUM_FLOORS, MAX_PERSON), dtype=Person)
    comingBackArray = np.ndarray(shape=(NUM_FLOORS, 24, MAX_PERSON), dtype=Person)
    building = Building(buildingsDATA, NUM_LIFTS, NUM_FLOORS, 0)
    observation_space = np.zeros(shape=(NUM_FLOORS, 1 + (NUM_LIFTS * 2)), dtype=int)
    timestepArr_SA = np.ndarray(shape=(1440, 2), dtype=int)
    timestepArr_reward = np.zeros(shape=1440, dtype=int)

    for x in range(NUM_LIFTS):
        observation_space[0][2 * x + 1] = 1
    generatingData()
    createObsSpace(timestepHours, building)
    # det = DetAgent(building, REWARD_SCALE, action_space, observation_space)
    while timestepHours != 24:

        r = 0

        if timestep % 20 == 0 and timestep != 0:
            # print('=== Generating more HOOMANs ===')
            generatingData()
            createObsSpace(timestepHours, building)
            checkFloor(building)
        '''
        print('Reward received at this timestep: ', r)
        print('Accumulative Reward: ', r_All)
        
        for liftIndex in range(NUM_LIFTS):
            print(liftIndex, ":", "Carrying", building.lifts[liftIndex].carryingPerson)
        printObsSpace()
        '''
        # replace this part with the agent
        '''
        implement your A.I here
        '''
        # Do conversion of observation space to an interger
        s = np.copy(observation_space)
        s_index = convertObsSpaceToInt(s, building)

        # choose actions
        if epsilon > random.random():
            action = np.random.randint(0, len(action_space))
        else:
            action = np.argmax(Q[s_index])

        '''
        End of A.I class
        '''
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
        '''
        implement your A.I here
        '''
        s_prime = np.copy(observation_space)
        sprime_index = convertObsSpaceToInt(s_prime, building)
        # updating Q table lamda
        timestepArr_SA[timestep] = [s_index, action]
        timestepArr_reward[timestep] = r * REWARD_SCALE
        t = timestep - LAMDA + 1
        if t >= 0:
            dest = min(t + LAMDA, 1440)
            G = 0
            for i in range(t + 1,dest):
                G += pow(discount, i - t - 1) * timestepArr_reward[i]
            if t + LAMDA < 1440:
                index = timestepArr_SA[timestep][0]
                a = timestepArr_SA[timestep][1]
                G += G + pow(discount, LAMDA) * Q[index][a]
            Q[index][a] = Q[index][a] + learning_rate * (
                    G - Q[index][a])
        sys.stdout.write("\r{0}".format("=" * timestepHours, e) + "[episode " + str(e) + "]")
        s = np.copy(s_prime)
        s_index = sprime_index
        # r_Det += det.run(timestep, timestepHours)
        timestep = timestep + 1
        timestepHours = math.floor(timestep / 60)

        sys.stdout.flush()

    array_episode_reward.append(r_All)
    # array_det_reward.append(r_Det)
    sys.stdout.write("[reward:" + str(r_All) + "] [epsilon: " + str(epsilon) + "]" + "Action:" + str(action) + "\n")
    # epsilon decay
    if e > WARMUP:
        epsilon = epsilon_decay * epsilon
        e_array.append(epsilon)
    # np.savetxt("Q.csv", Q, delimiter=",")

    '''
    End of A.I class
    '''

'''
download data 
'''

np.savetxt("dataTD.csv", array_episode_reward, delimiter=",")
np.savetxt("epsilon.csv", e_array, delimiter=",")
# outfile = open("Qstate.txt", "a")
# for x in range(len(Q_s)):
#     np.savetxt(outfile, Q_s[x], fmt='%-7.2f')
# outfile.close()
# "] [Det Reward:" + str(r_Det) +
