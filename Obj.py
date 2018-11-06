import numpy as np

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
        self.lifts = np.ndarray(shape=2, dtype=Lift)
        for x in range(NUM_LIFTS):
            self.lifts[x] = Lift(startingPos, NUM_FLOORS)
