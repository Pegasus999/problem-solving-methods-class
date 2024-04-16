from __future__ import annotations
import random
from typing import Optional
import time


class Reverter:
    """This class represents an array to be sorted. It formally encodes the states of the problem"""

    def __init__(self, size: int, init=True) -> None:
        """The class only sorts an array containing numbers 1..size. The constructor shuffles the array
        in order to create an unsorted array.

        Args:
            size (int): the size of the array
            init (bool, optional): if True, the array is initialized with value 1..size, the shuffled, else, the array
            remains empty (it is used to clone the array). Defaults to True.
        """
        if init:
            self.table = list(range(1, size + 1))
            random.shuffle(self.table)
            self.hash()
            self.parent = None
            self.g = 0
            self.h = self.heuristic()
            self.f = self.g + self.h
        else:
            self.table = []

    def __str__(self) -> str:
        """returns a string representation of the object Reverter

        Returns:
            str: the string representation
        """
        return str(self.table)

    def hash(self):
        """Compute a hashcode of the array. Since it is not possible to hash a list, this one is first
        converted to a tuple
        """
        self.__hash__ = hash(tuple(self.table))

    def __eq__(self, __value: Reverter) -> bool:
        """Tests whether the current object if equals to another object (Reverter). The comparison is made by comparing the hashcodes

        Args:
            __value (Reverter): _description_

        Returns:
            bool: True if self==__value, else it is False
        """
        return self.__hash__ == __value.__hash__

    def is_the_goal(self) -> bool:
        """Tests whether the table is already sorted (so that the search is stopped)

        Returns:
            bool: True if the table is sorted, else it is False.
        """
        for i in range(1, len(self.table)):
            if self.table[i - 1] > self.table[i]:
                return False
        return True

    def clone(self) -> Reverter:
        """This methods create a copy of the current object

        Returns:
            Reverter: the copy to be created
        """
        res = Reverter(len(self.table), False)
        res.table = [*self.table]
        res.parent = self
        return res

    def actions(self) -> list[Reverter]:
        """This class builds a list of possible actions. The returned list contains a set of tables depending of possible
        reverting of the current table

        Returns:
            list[Reverter]: the list of tables obtained after applying the possible reverting
        """
        res = []
        sz = len(self.table)
        for i in range(sz):
            r = self.clone()
            v = self.table[i:]
            v.reverse()
            r.table = self.table[:i] + v
            r.hash()
            res.append(r)
        return res

    def printSteps(current_state, parent_map):
        path = [current_state]
        parent_state = parent_map[current_state.hash]
        while parent_state is not None:
            path.insert(0, parent_state)
            parent_state = parent_map[parent_state.hash]
        for state in path:
            print(state)

    def solveBreadth(self) -> Optional[Reverter]:
        """This method implements breadth first search

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []

        while len(OPEN) > 0:
            current_node = OPEN.pop(0)
            CLOSED.append(current_node)
            if current_node.is_the_goal():
                return current_node

            possible_lists = current_node.actions()
            for node in possible_lists:
                if node not in CLOSED and node not in OPEN:
                    OPEN.append(node)

        # raise NotImplementedError("This method is not yet implemented")

    def solveDepth(self) -> Optional[Reverter]:
        """This method implements depth first search

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []

        while len(OPEN) > 0:
            current_node = OPEN.pop()
            CLOSED.append(current_node)
            if current_node.is_the_goal():
                return current_node

            possible_lists = current_node.actions()
            for node in possible_lists:
                if node not in CLOSED and node not in OPEN:
                    OPEN.insert(0, node)

        # raise NotImplementedError("This method is not yet implemented")

    def solveRandom(self) -> Optional[Reverter]:
        """This method implements random search

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []

        while len(OPEN) > 0:
            n = random.choice(OPEN)
            CLOSED.append(n)
            if n.is_the_goal():
                return n

            m = n.actions()
            for node in m:
                if node not in CLOSED and node not in OPEN:
                    OPEN = OPEN + m

    def heuristic(self):
        h = 0
        for i in range(len(self.table)):
            left_sum = sum(1 for j in range(i) if self.table[j] > self.table[i])
            right_sum = sum(
                1
                for j in range(i + 1, len(self.table))
                if self.table[j] < self.table[i]
            )
            h += left_sum + right_sum
        return h

    def solveHeuristic1(self) -> Optional[Reverter]:
        """This method implements heuristic search (heuristic n° 1)

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []
        while len(OPEN) > 0:
            n = min(OPEN, key=lambda x: x.f, default=None)

            OPEN.remove(n)
            CLOSED.append(n)
            if n.is_the_goal():
                return n

            m = n.actions()
            for node in m:
                if node not in OPEN and node not in CLOSED:
                    OPEN.append(node)
                    node.g = 0
                    node.h = node.heuristic()
                    node.f = node.g + node.h

    def solveHeuristic2(self) -> Optional[Reverter]:
        """This method implements heuristic search (heuristic n° 2)

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []
        while len(OPEN) > 0:
            n = min(OPEN, key=lambda x: x.f, default=None)

            OPEN.remove(n)
            CLOSED.append(n)
            if n.is_the_goal():
                return n

            m = n.actions()
            for node in m:
                if node not in OPEN and node not in CLOSED:
                    OPEN.append(node)
                    node.g = n.g + 1
                    node.h = node.heuristic()
                    node.f = node.g + node.h

    def heuristic_value(self) -> int:
        """Calculate the heuristic value for the current state.

        Returns:
            int: The heuristic value indicating the estimated distance to the goal state.
        """
        count_correct = sum(
            1 for i in range(1, len(self.table)) if self.table[i] > self.table[i - 1]
        )
        return len(self.table) - count_correct

    def solveHeuristic3(self) -> Optional[Reverter]:
        """This method implements heuristic search (heuristic n° 2).

        Returns:
            Optional[Reverter]: The sorted table if possible, otherwise None.
        """
        OPEN = [self]
        CLOSED = []

        while OPEN:
            current_state = min(OPEN, key=lambda x: x.heuristic_value())
            OPEN.remove(current_state)

            if current_state.is_the_goal():
                return current_state

            CLOSED.append(current_state)

            successor_states = current_state.actions()

            for successor_state in successor_states:
                if successor_state not in CLOSED:
                    if successor_state not in OPEN:
                        OPEN.append(successor_state)
        return None


def calculate_time(name, func):
    start_time = time.time() * 1000
    r = func()
    end_time = time.time() * 1000
    print(name, "->", int(end_time - start_time), "ms")


size = 10  # 8,...,15,...
rev = Reverter(size, True)
print(rev.table)
# calculate_time("Breadth first", rev.solveBreadth)
# calculate_time("Depth first ", rev.solveDepth)
# calculate_time("random", rev.solveRandom)
calculate_time("Heuristic 1", rev.solveHeuristic1)
calculate_time("Heuristic 2", rev.solveHeuristic2)
# calculate_time("Heuristic 3", rev.solveHeuristic3)
