from __future__ import annotations
import random
import time
from queue import Queue
from typing import Optional


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

    def solveBreadth(self) -> Optional[Reverter]:
        """This method implements breadth first search

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        queue = Queue()
        visited = set()
        parent_map = {self.hash: None}

        # add initial table to the queue and mark it as visited
        queue.put(self)
        visited.add(self.hash)

        while not queue.empty():
            # get next element
            current_state = queue.get()

            if current_state.is_the_goal():
                # get to this state if it turns out to be the solution
                path = [current_state]
                parent_state = parent_map[current_state.hash]
                while parent_state is not None:
                    path.insert(0, parent_state)
                    parent_state = parent_map[parent_state.hash]
                for state in path:
                    print(state)
                return current_state

            # develop the possible states out of the current state
            possible_actions = current_state.actions()
            for action in possible_actions:
                # check if the action has been visited before
                if action.hash not in visited:
                    # mark the action as visited, add it to the queue, and update parent map
                    visited.add(action.hash)
                    queue.put(action)
                    parent_map[action.hash] = current_state

        print("No solution found.")
        return None

    def solveDepth(self) -> Optional[Reverter]:
        """This method implements depth first search

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []
        parent_map = {self.hash: None}

        while len(OPEN) > 0:
            n = OPEN.pop(0)
            CLOSED.append(n)
            if n.is_the_goal():
                # Trace back the path from the goal state to the initial state and print it
                path = [n]
                parent_state = parent_map[n.hash]
                while parent_state is not None:
                    path.insert(0, parent_state)
                    parent_state = parent_map[parent_state.hash]
                for state in path:
                    print(state)
                return n
            m = n.actions()
            for node in m:
                if node not in CLOSED and node not in OPEN:
                    OPEN.insert(0, node)
                    parent_map[node.hash] = n

        print("No solution found.")
        return None

    def solveRandom(self) -> Optional[Reverter]:
        """This method implements random search

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []
        parent_map = {self.hash: None}

        while len(OPEN) > 0:
            n = OPEN.pop(random.randint(0, len(OPEN) - 1))
            CLOSED.append(n)
            if n.is_the_goal():
                # Trace back the path from the goal state to the initial state and print it
                path = [n]
                parent_state = parent_map[n.hash]
                while parent_state is not None:
                    path.insert(0, parent_state)
                    parent_state = parent_map[parent_state.hash]
                for state in path:
                    print(state)
                return n
            m = n.actions()
            for node in m:
                if node not in CLOSED and node not in OPEN:
                    OPEN.append(node)
                    parent_map[node.hash] = n

        print("No solution found.")
        return None

    def solveHeuristic1(self) -> Optional[Reverter]:
        """This method implements heuristic search (heuristic n° 1)

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []
        parent_map = {self.hash: None}

        while len(OPEN) > 0:
            n = min(OPEN, key=lambda x: x.f, default=None)

            OPEN.remove(n)
            CLOSED.append(n)
            if n.is_the_goal():
                path = [n]
                parent_state = parent_map[n.hash]
                while parent_state is not None:
                    path.insert(0, parent_state)
                    parent_state = parent_map[parent_state.hash]
                for state in path:
                    print(state)

                return n

            m = n.actions()
            for node in m:
                if node not in OPEN and node not in CLOSED:
                    OPEN.append(node)
                    node.g = 0
                    node.h = node.heuristic()
                    node.f = node.g + node.h
                    parent_map[node.hash] = n

    def solveHeuristic2(self) -> Optional[Reverter]:
        """This method implements heuristic search (heuristic n° 2)

        Returns:
            Optional[Reverter]: the sorted table is possible
        """
        OPEN = [self]
        CLOSED = []
        parent_map = {self.hash: None}
        while len(OPEN) > 0:
            n = min(OPEN, key=lambda x: x.f, default=None)

            OPEN.remove(n)
            CLOSED.append(n)
            if n.is_the_goal():
                path = [n]
                parent_state = parent_map[n.hash]
                while parent_state is not None:
                    path.insert(0, parent_state)
                    parent_state = parent_map[parent_state.hash]
                for state in path:
                    print(state)

                return n

            m = n.actions()
            for node in m:
                if node not in OPEN and node not in CLOSED:
                    OPEN.append(node)
                    node.g = n.g + 1
                    node.h = node.heuristic()
                    node.f = node.g + node.h
                    parent_map[node.hash] = n

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
        OPEN = [self]  # Priority queue sorted by the heuristic value
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
                    else:
                        existing_state = OPEN[OPEN.index(successor_state)]
                        if (
                            successor_state.heuristic_value()
                            < existing_state.heuristic_value()
                        ):
                            OPEN[OPEN.index(successor_state)] = successor_state

        return None


rev = Reverter(8, True)


print("Breadth search:")
start_time_5 = time.time() * 1000
rev.solveBreadth()
end_time_5 = time.time() * 1000
breadth = int(end_time_5 - start_time_5)


print("\nDepth search:")
start_time_5 = time.time() * 1000
rev.solveDepth()
end_time_5 = time.time() * 1000
depth = int(end_time_5 - start_time_5)


print("\nRandom search:")
start_time_5 = time.time() * 1000
rev.solveRandom()
end_time_5 = time.time() * 1000
randomi = int(end_time_5 - start_time_5)


print("\nH1 search:")
start_time_5 = time.time() * 1000
rev.solveHeuristic2()
end_time_5 = time.time() * 1000
h1 = int(end_time_5 - start_time_5)


print("\nH2 search:")
start_time_5 = time.time() * 1000
rev.solveHeuristic2()
end_time_5 = time.time() * 1000
h2 = int(end_time_5 - start_time_5)

print("\nH3 search:")
start_time_5 = time.time() * 1000
r = rev.solveHeuristic3()
print(r)
end_time_5 = time.time() * 1000
h3 = int(end_time_5 - start_time_5)

print(" breadth : ", breadth)
print(" depth : ", depth)
print(" random : ", randomi)
print(" h1 : ", h1)
print(" h2 : ", h2)
print(" h3 : ", h3)
