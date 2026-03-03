class RectangularRoom(object):
    """
    A RectangularRoom represents a rectangular region containing clean or dirty
    tiles.

    A room has a width and a height and contains (width * height) tiles. At any
    particular time, each of these tiles is either clean or dirty.
    """
    def __init__(self, width, height):
        """
        Initializes a rectangular room with the specified width and height.

        Initially, no tiles in the room have been cleaned.

        width: an integer > 0
        height: an integer > 0
        cleaned: a list of tuples representing cleaned titles
        """
        self.width = width
        self.height = height
        self.cleaned=[]
    
    def cleanTileAtPosition(self, pos):
        """
        Mark the tile under the position POS as cleaned.

        Assumes that POS represents a valid position inside this room.

        pos: a Position
        """
        x=math.floor(Position.getX(pos))
        y=math.floor(Position.getY(pos))
        title=(x,y)
        if title not in self.cleaned:
            self.cleaned.append(title)

    def isTileCleaned(self, m, n):
        """
        Return True if the tile (m, n) has been cleaned.

        Assumes that (m, n) represents a valid tile inside the room.

        m: an integer
        n: an integer
        returns: True if (m, n) is cleaned, False otherwise
        """
        title=(m,n)
        return title in self.cleaned
    
    def getNumTiles(self):
        """
        Return the total number of tiles in the room.

        returns: an integer
        """
        return self.width*self.height

    def getNumCleanedTiles(self):
        """
        Return the total number of clean tiles in the room.

        returns: an integer
        """
        return len(self.cleaned)

    def getRandomPosition(self):
        """
        Return a random position inside the room.

        returns: a Position object.
        """
        w=float(self.width)
        h=float(self.height)
        x=random.uniform(0.00,w-0.01)
        y=random.uniform(0.00,h-0.01)
        pos=Position(x,y)
        return pos

    def isPositionInRoom(self, pos):
        """
        Return True if pos is inside the room.

        pos: a Position object.
        returns: True if pos is in the room, False otherwise.
        """
        x=Position.getX(pos)
        y=Position.getY(pos)
        return x>=0 and y>=0 and x<self.width and y<self.height
  class RectangularRoom(object):
    """
    A RectangularRoom represents a rectangular region containing clean or dirty
    tiles.

    A room has a width and a height and contains (width * height) tiles. At any
    particular time, each of these tiles is either clean or dirty.
    """
    def __init__(self, width, height):
        """
        Initializes a rectangular room with the specified width and height.

        Initially, no tiles in the room have been cleaned.

        width: an integer > 0
        height: an integer > 0
        cleaned: a list of tuples representing cleaned titles
        """
        self.width = width
        self.height = height
        self.cleaned=[]
    
    def cleanTileAtPosition(self, pos):
        """
        Mark the tile under the position POS as cleaned.

        Assumes that POS represents a valid position inside this room.

        pos: a Position
        """
        x=math.floor(Position.getX(pos))
        y=math.floor(Position.getY(pos))
        title=(x,y)
        if title not in self.cleaned:
            self.cleaned.append(title)

    def isTileCleaned(self, m, n):
        """
        Return True if the tile (m, n) has been cleaned.

        Assumes that (m, n) represents a valid tile inside the room.

        m: an integer
        n: an integer
        returns: True if (m, n) is cleaned, False otherwise
        """
        title=(m,n)
        return title in self.cleaned
    
    def getNumTiles(self):
        """
        Return the total number of tiles in the room.

        returns: an integer
        """
        return self.width*self.height

    def getNumCleanedTiles(self):
        """
        Return the total number of clean tiles in the room.

        returns: an integer
        """
        return len(self.cleaned)

    def getRandomPosition(self):
        """
        Return a random position inside the room.

        returns: a Position object.
        """
        w=float(self.width)
        h=float(self.height)
        x=random.uniform(0.00,w-0.01)
        y=random.uniform(0.00,h-0.01)
        pos=Position(x,y)
        return pos

    def isPositionInRoom(self, pos):
        """
        Return True if pos is inside the room.

        pos: a Position object.
        returns: True if pos is in the room, False otherwise.
        """
        x=Position.getX(pos)
        y=Position.getY(pos)
        return x>=0 and y>=0 and x<self.width and y<self.height
class Robot(object):
    """
    Represents a robot cleaning a particular room.

    At all times the robot has a particular position and direction in the room.
    The robot also has a fixed speed.

    Subclasses of Robot should provide movement strategies by implementing
    updatePositionAndClean(), which simulates a single time-step.
    """
    def __init__(self, room, speed):
        """
        Initializes a Robot with the given speed in the specified room. The
        robot initially has a random direction and a random position in the
        room. The robot cleans the tile it is on.

        room:  a RectangularRoom object.
        speed: a float (speed > 0)
        """
        self.room=room
        self.speed=speed
        self.direction=random.randint(0,359)
        self.position=RectangularRoom.getRandomPosition(self.room)
        RectangularRoom.cleanTileAtPosition(self.room, self.position)

    def getRobotPosition(self):
        """
        Return the position of the robot.

        returns: a Position object giving the robot's position.
        """
        return self.position
    
    def getRobotDirection(self):
        """
        Return the direction of the robot.

        returns: an integer d giving the direction of the robot as an angle in
        degrees, 0 <= d < 360.
        """
        return self.direction

    def setRobotPosition(self, position):
        """
        Set the position of the robot to POSITION.

        position: a Position object.
        """
        self.position=position

    def setRobotDirection(self, direction):
        """
        Set the direction of the robot to DIRECTION.

        direction: integer representing an angle in degrees
        """
        self.direction=direction

    def updatePositionAndClean(self):
        """
        Simulate the passage of a single time-step.

        Move the robot to a new position and mark the tile it is on as having
        been cleaned.
        """
        raise NotImplementedError # don't change this!
class StandardRobot(Robot):
    """
    A StandardRobot is a Robot with the standard movement strategy.

    At each time-step, a StandardRobot attempts to move in its current
    direction; when it would hit a wall, it *instead* chooses a new direction
    randomly.
    """
    def updatePositionAndClean(self):
        """
        Simulate the passage of a single time-step.

        Move the robot to a new position and mark the tile it is on as having
        been cleaned.
        """
        pos=Position.getNewPosition(self.position,self.direction,self.speed)
        if RectangularRoom.isPositionInRoom(self.room, pos):
            self.position=pos
            RectangularRoom.cleanTileAtPosition(self.room, self.position)
        else:
            self.direction=random.randint(0,359)
class RandomWalkRobot(Robot):
    """
    A RandomWalkRobot is a robot with the "random walk" movement strategy: it
    chooses a new direction at random at the end of each time-step.
    """
    def updatePositionAndClean(self):
        """
        Simulate the passage of a single time-step.

        Move the robot to a new position and mark the tile it is on as having
        been cleaned.
        """
        pos=Position.getNewPosition(self.position,self.direction,self.speed)
        if RectangularRoom.isPositionInRoom(self.room, pos):
            self.position=pos
            RectangularRoom.cleanTileAtPosition(self.room, self.position)
        self.direction=random.randint(0,359)
def runSimulation(num_robots, speed, width, height, min_coverage, num_trials,
                  robot_type):
    """
    Runs NUM_TRIALS trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction MIN_COVERAGE of the room.

    The simulation is run with NUM_ROBOTS robots of type ROBOT_TYPE, each with
    speed SPEED, in a room of dimensions WIDTH x HEIGHT.

    num_robots: an int (num_robots > 0)
    speed: a float (speed > 0)
    width: an int (width > 0)
    height: an int (height > 0)
    min_coverage: a float (0 <= min_coverage <= 1.0)
    num_trials: an int (num_trials > 0)
    robot_type: class of robot to be instantiated (e.g. StandardRobot or
                RandomWalkRobot)
    """
    steps=0
    for i in range(num_trials):
        robots=[]
        step=0
        room=RectangularRoom(width,height)
        for i in range(num_robots):
            robot=robot_type(room,speed)
            robots.append(robot)
        while RectangularRoom.getNumCleanedTiles(room)<min_coverage*RectangularRoom.getNumTiles(room):
            for robot in robots:
                robot_type.updatePositionAndClean(robot)
            step+=1
        steps+=step
    return steps/num_trials
