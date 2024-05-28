import math
import random


from matplotlib import pyplot as plt


class Point:
    """This class represents a point to be drawn on the grid
    """
    
    def __init__(self,x,y) -> None:
        """Builds a Point

        Args:
            x (int): the abscissa
            y (int): the ordinate
        """
        self.x=x
        self.y=y
    
    def __eq__(self, value) -> bool:
        """Test if two points are equals

        Args:
            value (Point): the point to be compared to self

        Returns:
            bool: True if self==value, else False
        """
        return self.x==value.x and self.y==value.y
    
    def copy(self):
        """Create a copy of this point

        Returns:
            Point: a copy of the point
        """
        return Point(self.x,self.y)
    
    def __str__(self) -> str:
        """Returns a string representation of the point

        Returns:
            str: the string representation
        """
        return f"({self.x},{self.y})"
    
    def __repr__(self) -> str:
        """Does the same thing as str

        Returns:
            str: the same string returned by str
        """
        return str(self)

class Solution:
    graph=None
    
    def __init__(self,m,n,init=True):
        """If init is True, the constructor creates a random assignment of vertices and grid points. It assures that no two
        vertices will be at the same position. If init is False, the constructor simply initializes the grid with no vertices present.
        
        Hill climbing and simulated annealing should use the first version of the constructor (init=True). The second version
        should be used for backtracking resolution, and brand and bound resolution.
        
        Note that self.grid[row][col]==0 if the point at (row,col) is empty, otherwise it contains the identity of the vertex that lie
        within

        Args:
            m (_type_): rows' number in the grid
            n (_type_): columns' number in the grid
            init (bool, optional): True if the vertices are to be placed randomly in the grid, False if the only the grid is created
        """
        self.m=m
        self.n=n
        self.positions=[]#these are the positions of the vertices. This is a list of Point.
        self.grid=[[0]*n for _ in range(m)]
        if init:
            for node in range(len(Solution.graph["nodes"])):
                while True:
                    row=random.randint(0,m-1)
                    col=random.randint(0,n-1)
                    if self.grid[row][col]==0:break
                self.grid[row][col]=node+1
                self.positions.append(Point(col,row))
    
    def copy(self):
        """Create a copy of the current solution

        Returns:
            Solution: the create copy
        """
        c=Solution(self.m,self.n,False)
        c.positions=[p.copy() for p in self.positions]
        return c
    
    def __eq__(self, value) -> bool:
        """Tests if two solutions are equal

        Args:
            value (Solution): the solution to be compared to

        Returns:
            bool: True if self==value, False otherwise
        """
        for i in range(len(self.positions)):
            if self.positions[i]!=value.positions[i]:return False
        return True
    
    def __lt__(self, other) -> bool:
        """This methods compares two solution to decide which one is better. Let s and t be two solutions, s is better than t if:
        - s.nb<t.nb (there are less edge intersections in the solution s than in t)
        - s.fitness<t.fitness if s.nb=t.nb

        Args:
            other (Solution): the solution to be compared to

        Returns:
            bool: True if self < other, False otherwise
        """
        return True if self.nb<other.nb else False if self.nb>other.nb else self.fitness<other.fitness
    
    def compute_fitness(self):
        """This method computes the fitness of the current solution by:
        - computing of edge intersections, let it be nb
        - computing the minimal distance between vertices and edges, let it be pen
        - fitness=3*nb-nb
        - fitness should be minimized (either with hill climbing or simulated annealing)

        Returns:
            float: the value of the fitness
        """
        self.nb=compute_intersections(self.positions,Solution.graph)
        self.pen=compute_minimal_distance(self.positions,Solution.graph)
        self.fitness=3*self.nb-self.pen
        return self.fitness
    
    def plot(self,filename):
        """This method draw the graph by using the positions of the vertices (based on matplotlib modules)
        """

        plt.clf()
        
        plt.scatter([p.x for p in self.positions],[p.y for p in self.positions],c="r")
        for node in range(len(Solution.graph["nodes"])):
            plt.text(self.positions[node].x-0.1,self.positions[node].y+0.2,f"{node}",fontsize=10)
        for source,target in Solution.graph["edges"]:
            plt.plot([self.positions[source].x,self.positions[target].x],[self.positions[source].y,self.positions[target].y],c="k")
        
        plt.savefig(filename, format='png')
        
    def mutate(self,extent):
        """The method introduces a random modification to the current solution. Technically,
        it relocates certain vertices from occupied points to vacant ones.

        Args:
            extent (int): the maximum number of points to displace

        Returns:
            Solution: the current solution after modification
        """
        mdf=random.randint(1,extent)
        cands=list(range(len(self.positions)))
        to_modify={}
        for _ in range(mdf):
            i=random.choice(cands)
            to_modify[i]=None
            cands.remove(i)
        modif={}
        for i in to_modify:
            while True:
                p=Point(random.randint(0,self.m-1),random.randint(0,self.n-1))
                if p not in self.positions and p not in modif.values():
                    modif[i]=p
                    break

        for i in modif:
            self.positions[i]=modif[i]
        return self

    def place_vertex(self,row,column,vertex):
        """This method tries to place a vertex at a given position. If this is possible, the vertex is placed and the method
        returns True, otherwise it return False

        Args:
            row (int): the abscissa of the point
            column (int): the ordinate of the point
            vertex (int): the identity of the vertex

        Returns:
            bool: True if the point is empty, False otherwise
        """
        if self.grid[row][column]==0:
            self.grid[row][column]=vertex+1
            self.positions.append(Point(column,row))
            return True
        else:
            return False
    
    def remove_last(self):
        # This metho undoes the last assignement
        p=self.positions.pop()
        self.grid[p.y][p.x]=0
    
    
    
    
    def simulated_annealing(self,neigh_size=10,extent=3,T=10,iter_max=1000):
        current=self
        current.compute_fitness()
        best=current
        T_init=T
        for i in range(iter_max):
            T=T_init*math.exp(math.log(1e-5/T_init)*i/iter_max)
            # print(i,best.fitness,best.pen,f"  >>>{best.nb}")
            next_hope=current.copy().mutate(extent)

            next_hope.compute_fitness()
            for _ in range(1,neigh_size):
                candidate=current.copy().mutate(extent)
                candidate.compute_fitness()
                if candidate<next_hope:
                    next_hope=candidate
            if next_hope<current:
                current=next_hope
            else:
                t=math.exp((current.fitness - next_hope.fitness) / T)
                if random.random() < t:
                    current=next_hope
            if current<best:
                best=current
        return best
    
    #This is actually a second chance hill climbing
    def hill_climbing(self,neigh_size=10,extent=3,iter_max=1000):
        current=self
        current.compute_fitness()
        for i in range(iter_max):
            print(i,current.fitness,current.nb,current.pen)
            next_hope=current.copy().mutate(extent)
            next_hope.compute_fitness()
            for _ in range(1,neigh_size):
                candidate=current.copy().mutate(extent)
                candidate.compute_fitness()
                if candidate<next_hope:
                    next_hope=candidate
            if next_hope<current:
                current=next_hope
        return current
    
    def backtracking(self):
        """You should implement this method to solve the problem by backtracking
        """
        nodes = self.graph['nodes'] # Liste des noeuds dans le graphe
        self.bestNb = float('inf') # Initialise le meilleur nombre d'intersections à l'infini
        self.best = None # Initialise la meilleure solution à None

        def backtrack(vertex):
            if vertex == len(nodes): # tous les noeuds ont été placés
                self.compute_fitness() # Calcule la fitness de la solution actuelle
                
                if self.nb < self.bestNb:
                    # mettre à jour self.bestNb et self.best
                    print(self.positions,self.fitness,self.nb,self.pen)
                    self.bestNb = self.nb
                    self.best = self
                    print('new best',self.best.nb)
                return 
            # Boucle de placement des vertex
            for row in range(self.m):
                
                for col in range(self.n):
                    if self.place_vertex(row, col, vertex):
                        backtrack(vertex + 1)
                        self.remove_last() # Enlève le dernier vertex placé pour revenir à l'état précédent et essayer une nouvelle position
        backtrack(0)
        return self.best # Retourne la meilleure solution trouvée
    
    def branch_and_bound(self):
        """you should implement this method to solve the problem by brand and bound
        """
        nodes = self.graph['nodes']
        self.best = None
        self.bestNb = float('inf')
        # self.bestNb = 3

        def backtrack(vertex):
            if vertex == len(nodes):
                self.compute_fitness()
                
                if self.nb < self.bestNb:
                    print(self.positions,self.fitness,self.nb,self.pen)
                    self.bestNb = self.nb
                    self.best = self

                    print('new best',self.best.nb)
                return
            

            self.nb = compute_intersections(self.positions,Solution.graph)

            if self.nb > self.bestNb:
                print(f'rejected at depth {vertex} because nb is bigger already')
                return
            
            if vertex == 0:
                for row in range(math.floor(self.m/2)):
                    for col in range(math.floor(self.n/2)):
                        if self.place_vertex(row, col, vertex):
                            backtrack(vertex + 1)
                            self.remove_last()
            else:
                for row in range(self.m):
                    for col in range(self.n):
                        if self.place_vertex(row, col, vertex):
                            backtrack(vertex + 1)
                            self.remove_last()
        backtrack(0)
        return self.best

#Utility functions to be used for computing useful functions

def compute_intersections(points,graph):
    """You should implement this function that computes the number of intersections of the edges (using the function doIntersect)
    """
    intersections = 0
    edges = graph['edges']
    for i in range(len(edges)):
        if edges[i][0] >= len(points) or edges[i][1] >= len(points):
                continue
        for j in range(i + 1, len(edges)):
            if edges[j][0] >= len(points) or edges[j][1] >= len(points):
                continue

            if edges[i][0] in edges[j] or edges[i][1] in edges[j]:
                continue

            if doIntersect(points[edges[i][0]], points[edges[i][1]], points[edges[j][0]], points[edges[j][1]]):
                intersections += 1
    return intersections

def compute_minimal_distance(points,graph):
    """You should implement this function that computes the minimal distance between a vertice and every edge in the graph (you should not
    consider the distance between a Point A and a segment [AB])
    """
    min_distance = float('inf')
    edges = graph['edges']
    for i, point in enumerate(points):
        for edge in edges:
            if i not in edge:  # Ne pas considérer la distance entre un point et une arête à laquelle il appartient
                distance = minDistance(points[edge[0]], points[edge[1]], point)
                if distance < min_distance:
                    min_distance = distance
    return min_distance if min_distance != float('inf') else 0

def minDistance(A, B, E) :
    """This function computes the euclidean distance between the point E and the segment [AB].
    This code has been adapted from: https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/

    Args:
        A (Point): The first endpoint of the segment
        B (Point): The second endpoint of the segment
        E (Point): The point for which the distance is computed

    Returns:
        float: the distance between E and the segement [AB]
    """
    # vector AB 
    AB = [None, None] 
    AB[0] = B.x - A.x 
    AB[1] = B.y - A.y
    
    # vector BE
    BE = [None, None]
    BE[0] = E.x - B.x
    BE[1] = E.y - B.y

    # vector AE 
    AE = [None, None]
    AE[0] = E.x - A.x
    AE[1] = E.y - A.y

    # Variables to store dot product 
    # Calculating the dot product 
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1] 
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1] 

    # Minimum distance from 
    # point E to the line segment 
    reqAns = 0 

    # Case 1 
    if (AB_BE > 0) :
        # Finding the magnitude 
        y = E.y - B.y
        x = E.x - B.x 
        reqAns = math.sqrt(x * x + y * y) 
        
    # Case 2 
    elif (AB_AE < 0) :
        y = E.y - A.y
        x = E.x - A.x
        reqAns = math.sqrt(x * x + y * y) 

    # Case 3 
    else:
        # Finding the perpendicular distance 
        x1 = AB[0] 
        y1 = AB[1] 
        x2 = AE[0] 
        y2 = AE[1] 
        mod = math.sqrt(x1 * x1 + y1 * y1) 
        reqAns = abs(x1 * y2 - y1 * x2) / mod 
    
    return reqAns
    
def doIntersect(p1,q1,p2,q2):
    """This function test if the segment [p1q1] intersect with the segment [p2q2].
    This code has been adapted from: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    Args:
        p1 (Point): the first endpoint of the first segment
        q1 (Point): the second endpoint of the first segment
        p2 (Point): the first endpoint of the second segment
        q2 (Point): the second endpoint of the second segment
    
    Returns:
        bool: True if the segment [p1q1] intersect with the segment [p2q2], False otherwise
    """
    def onSegment(p, q, r): 
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
            return True
        return False
    
    def orientation(p, q, r): 
        # to find the orientation of an ordered triplet (p,q,r) 
        # function returns the following values: 
        # 0 : Collinear points 
        # 1 : Clockwise points 
        # 2 : Counterclockwise 
        
        val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y)) 
        if (val > 0): 
            # Clockwise orientation 
            return 1
        elif (val < 0): 
            # Counterclockwise orientation 
            return 2
        else: 
            # Collinear orientation 
            return 0
    
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 

    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True

    # Special Cases 

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True

    # If none of the cases 
    return False


if __name__=="__main__":
    #A graph is represented by a dictionary with two keys: nodes and edges. The "nodes" value gives the adjacent vertices of each vertex,
    # while the "edges" value gives the graph's edges (each edge is represented as a tuple). This is a redundancy, yet it is used to simplify the implementation.
    benchmarks = []
    benchmarks.append({'nodes': [[4, 3, 6, 1], [5, 8, 3, 0], [5, 3, 8], [5, 4, 2, 1, 6, 0], [0, 3, 5], [1, 2, 3, 4, 6, 9], [5, 7, 3, 0, 8], [9, 6], [1, 9, 2, 6], [7, 5, 8]], 'edges': [(0, 4), (1, 5), (2, 5), (3, 5), (3, 4), (4, 5), (5, 6), (7, 9), (1, 8), (5, 9), (6, 7), (2, 3), (8, 9), (1, 3), (3, 6), (0, 3), (0, 6), (0, 1), (2, 8), (6, 8)]})
    benchmarks.append({'nodes': [[1, 2], [0, 2, 5, 9, 8, 6], [1, 3, 4, 9, 6, 7, 0], [2, 9, 5], [2, 7], [1, 8, 7, 3], [7, 8, 2, 1], [6, 9, 4, 5, 2], [6, 5, 1], [3, 7, 2, 1]], 'edges': [(0, 1), (1, 2), (2, 3), (3, 9), (2, 4), (1, 5), (6, 7), (7, 9), (6, 8), (2, 9), (5, 8), (2, 6), (4, 7), (5, 7), (2, 7), (1, 9), (1, 8), (0, 2), (3, 5), (1, 6)]})
    benchmarks.append({'nodes': [[8, 6, 9, 5], [8, 5, 9, 6], [7, 3], [2, 6], [5, 6, 8, 9], [4, 1, 8, 7, 0], [3, 9, 4, 0, 8, 1], [2, 9, 5], [0, 1, 5, 4, 6], [7, 6, 0, 1, 4]], 'edges': [(0, 8), (1, 8), (2, 7), (2, 3), (4, 5), (1, 5), (3, 6), (7, 9), (5, 8), (6, 9), (4, 6), (0, 6), (4, 8), (6, 8), (5, 7), (0, 9), (1, 9), (4, 9), (0, 5), (1, 6)]})
    benchmarks.append({'nodes': [[8, 5, 7], [8, 4, 9], [6, 4, 5, 9], [7, 6], [1, 8, 9, 2, 5], [0, 7, 9, 4, 6, 2], [2, 3, 9, 5, 8], [3, 0, 5], [0, 1, 4, 6], [6, 4, 5, 1, 2]], 'edges': [(0, 8), (1, 8), (2, 6), (3, 7), (1, 4), (0, 5), (3, 6), (0, 7), (4, 8), (6, 9), (5, 7), (4, 9), (5, 9), (2, 4), (4, 5), (1, 9), (5, 6), (2, 5), (6, 8), (2, 9)]})
    benchmarks.append({'nodes': [[1,3, 6, 4], [0,8, 9, 2, 4,5], [7, 9, 8, 1, 4], [0, 4, 6], [3, 0, 1, 2], [7,1,8], [0, 3, 9], [2, 5, 8], [1, 7, 9, 2,5], [8, 1, 2, 6]], 'edges': [(0,1),(5,8), (0, 3), (1, 8), (2, 7), (3, 4), (5, 7), (0, 6), (7, 8), (8, 9), (1, 9), (3, 6), (2, 9), (0, 4), (2, 8), (1, 2), (1, 4), (6, 9), (2, 4),(1,5)]})
    benchmarks.append({'nodes': [[8, 6, 3, 2], [8, 4, 7, 5, 3], [7, 0], [5, 7, 1, 6, 0], [1, 7], [3, 9, 1], [0, 9, 3], [2, 1, 4, 3, 8], [0, 1, 9, 7], [5, 8, 6]], 'edges': [(0, 8), (1, 8), (2, 7), (3, 5), (1, 4), (5, 9), (0, 6), (1, 7), (8, 9), (6, 9), (4, 7), (1, 5), (3, 7), (1, 3), (7, 8), (3, 6), (0, 3), (0, 2)]})
    benchmarks.append({'nodes': [[3, 7, 4, 5], [5, 3, 8, 7], [3, 7, 5], [0, 2, 1, 7, 9, 4], [9, 0, 3, 7], [1, 7, 2, 0], [8, 9], [5, 3, 2, 0, 4, 1], [6, 1, 9], [4, 3, 6, 8]], 'edges': [(0, 3), (1, 5), (2, 3), (1, 3), (4, 9), (5, 7), (6, 8), (3, 7), (1, 8), (3, 9), (2, 7), (6, 9), (2, 5), (8, 9), (0, 7), (0, 4), (3, 4), (4, 7), (0, 5), (1, 7)]})
    benchmarks.append({'nodes': [[3, 6, 8, 5], [6, 4, 5], [6, 8, 3], [0, 8, 2, 5], [1, 9, 6, 5], [6, 7, 1, 3, 4, 0], [1, 2, 5, 0, 9, 4, 8], [5, 9], [3, 2, 0, 6], [4, 7, 6]], 'edges': [(0, 3), (1, 6), (2, 6), (3, 8), (1, 4), (5, 6), (0, 6), (5, 7), (2, 8), (4, 9), (1, 5), (0, 8), (7, 9), (6, 9), (4, 6), (2, 3), (3, 5), (4, 5), (6, 8), (0, 5)]})
    benchmarks.append({'nodes': [[2, 4, 1, 9, 8], [3, 7, 8, 0, 9, 5, 6], [0, 3, 9, 7], [1, 2, 6, 9, 7, 8], [0, 5, 9], [4, 9, 6, 1], [3, 8, 5, 1], [1, 3, 9, 8, 2], [6, 1, 7, 0, 3], [3, 4, 7, 5, 2, 0, 1]], 'edges': [(0, 2), (1, 3), (2, 3), (3, 6), (0, 4), (4, 5), (6, 8), (1, 7), (1, 8), (3, 9), (3, 7), (4, 9), (7, 9), (0, 1), (7, 8), (5, 9), (2, 9), (0, 9), (1, 9), (2, 7), (5, 6), (0, 8), (1, 5), (3, 8), (1, 6)]})
    
    
    
    for i in range(len(benchmarks)):

        Solution.graph=benchmarks[i]

        sol=Solution(12,12,False)
        # a=sol.backtracking()
        # a.plot()

        b=sol.branch_and_bound()
        b.plot()

        solmh=Solution(12,12,True)

        
        # print(f'using hill climbing on benchmark {i+1}...')
        # c=solmh.hill_climbing()
        # c.plot(f'./benchmark{i+1}_hill_climbing.png')
        # print('done')


        # print(f'using simulated annealing on benchmark {i+1}...')
        # d=solmh.simulated_annealing()
        # d.plot(f'./benchmark{i+1}_simulated_annealing.png')
        # print('done')

        
