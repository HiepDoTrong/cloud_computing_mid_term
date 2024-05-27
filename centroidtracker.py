# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import queue

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = (centroid)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # inputCarID = np.zeros(len(rects), dtype="int")


        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects, take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects
    

class Person(CentroidTracker):
    def __init__(self, maxDisappeared=50):
        # Call the constructor of the parent class
        super().__init__(maxDisappeared)


    def register(self, centroid, rect, carID=0):
        # Call the parent's register method
        super().register(centroid)

        # Modify the value in self.objects for this object to be a tuple of centroid and carID
        self.objects[self.nextObjectID - 1] = [centroid, rect, carID]
    
    def update(self, rects, car_objects, overlines):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        carCentroids = []

        for ((objectID, centroid_rect), (_, overline))  in zip(car_objects.items(), overlines.items()):
            centroid, rect = centroid_rect
            if overline==1: 
                carCentroids.append(centroid)    
            else:
                carCentroids.append((10000, 10000))
 
        carCentroids = np.array(carCentroids)
        # print(len(carCentroids))

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)


        person_car_dist = dist.cdist(np.array(carCentroids), np.array(inputCentroids))
        closest_carCentroid_indices = np.argmin(person_car_dist, axis=0)
        # closest_carCentroids = [carCentroids[i] for i in closest_carCentroid_indices]

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                
                self.register(inputCentroids[i], rects[i], closest_carCentroid_indices[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [centroid for centroid, rect, carID in self.objects.values()]

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID][0] = inputCentroids[col]
                self.objects[objectID][1] = rects[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col], closest_carCentroid_indices[col])
        return self.objects, self.disappeared


class Car(CentroidTracker):
    def __init__(self, maxDisappeared=50):
        super().__init__(maxDisappeared)

        self.overline = OrderedDict()
        self.moving = OrderedDict()

    def register(self, centroid, rect):
        super().register(centroid)

        self.objects[self.nextObjectID - 1] = [centroid, rect]
        self.overline[self.nextObjectID - 1] = -1
        # self.moving[self.nextObjectID - 1] = queue.Queue()
        # for _ in range(3):
        #     self.moving.put(0)

    def deregister(self, objectID):
        super().deregister(objectID)
        del self.overline[objectID]
        # del self.moving[objectID]
    
    def same_side(self, line_start, line_end, A, B):
        a = line_start[1] - line_end[1]
        b = line_end[0] - line_start[0]
        c = - (line_start[1] - line_end[1])*line_start[0] - (line_end[0] - line_start[0]) * line_start[1]
        
        check = (a * A[0] + b * A[1] + c) * (a * B[0] + b * B[1] + c)
        if check < 0:
            return False
        else:
            return True
    
    def move(self, A, B):
        distance = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
        if distance > 5:
            return True
        else:
            return False

    # def update(self, rects, line_start, line_end):
    def update(self, rects):

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
            # for objectID in self.objects.keys():
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects, self.overline, self.disappeared
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())                                 
            objectCentroids = [centroid for centroid, rect in self.objects.values()]
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                # if self.same_side(line_start, line_end, self.objects[objectID][0], inputCentroids[col])==False:
                    # print(objectID)
                self.overline[objectID] = self.overline[objectID]*-1
                # if self.move(self.objects[objectID][0], inputCentroids[col]):
                #     self.moving[objectID] += 1
                # else:
                #     self.moving[objectID] -= 1
                self.objects[objectID][0] = inputCentroids[col]
                self.objects[objectID][1] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        # print(self.objects, self.overline, self.disappeared)
        return self.objects, self.overline, self.disappeared
