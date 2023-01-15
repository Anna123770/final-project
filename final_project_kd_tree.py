import math
import time


class Node():
    def __init__(self, lchild, rchild, value, split_dim):
        self.lchild = lchild  # Left subtree of node
        self.rchild = rchild  # Right subtree of node
        self.value = value  # Value of node
        self.split_dim = split_dim  # Dimension used for division




class Rectangle():
    def __init__(self,lower_point, up_point):
        self.lower = lower_point
        self.upper = up_point

    def is_contains(self, point):
        # Return true if the point is inside the rectangle otherwise false
        return self.lower[0] <= point[0] <= self.upper[0] and self.lower[1] <= point[1] <= self.upper[1]




class KD_Tree():
    def __init__(self, data):
        self.dims = len(data[0])  # Total point
        self.nearest_point = None
        self.nearest_dist = math.inf  # Init value with INF

    def insert(self, current_data, split_dim):
        # Set recursive exit: exit when all samples are divided
        if len(current_data) == 0:
            return None

        mid = self.calculate_current_medium_value(current_data)  # Calculate the subscript of the median
        data_sorted = sorted(current_data, key=lambda x: x[split_dim])  # Sort by sharding dimension from small to large

        # The following three codes are essentially the post-order traversal of binary trees
        lchild = self.insert(data_sorted[0:mid], self.calucate_split_dim(split_dim))  # Recursively construct left subtree
        rchild = self.insert(data_sorted[mid + 1:], self.calucate_split_dim(split_dim))  # Recursively construct right subtree
        return Node(lchild, rchild, data_sorted[mid], split_dim)  # Connect the left and right subtrees starting from the root node and return

    # Calculate the next partition dimension
    def calucate_split_dim(self, split_dim):
        return (split_dim + 1) % self.dims

    # Calculate the subscript of the current dimension's median
    def calculate_current_medium_value(self, current_data):
        return len(current_data) // 2

    # Calculate the Euclidean distance between two points
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Pass in the root node of the kd tree and the point element to be searched, and search the nearest neighbor of the element
    def neighbor_query(self, root, point):
        if root is None:
            return 
        # Calculate the distance between the target node on the current partition dimension and the single dimension of the current node
        dist = root.value[root.split_dim] - point[root.split_dim]
        # Forward search
        if dist > 0:  # The current node is on the top or left of the target node (in 2D space)
            self.neighbor_query(root.lchild, point)  # Search the left subtree by using recursion
        else:  # Otherwise, the current node is below or to the right of the target node (in 2D space)
            self.neighbor_query(root.rchild, point)  # Search the right subtree by using recursion
        # Calculate the Euclidean distance between the target node and the current node
        current_dist = self.calculate_distance(root.value, point)
        # Update the nearest neighbor node
        if current_dist < self.nearest_dist:
            self.nearest_dist = current_dist
            self.nearest_point = root
                # print(self.nearest_point.value)

        # Compare whether the "nearest distance" exceeds the "distance between the target node and the current node in the current division dimension".
        # If it exceeds the distance, it indicates that there may be a closer point in the subtree on the other side of the current node,
        # so you need to search in the subtree on the other side of the current node
        if self.nearest_dist > abs(dist):
            # Because the search is performed in the subtree on the other side of the current node,
            # it is the opposite of the previous forward search
            if dist > 0:
                self.neighbor_query(root.rchild, point)
            else:
                self.neighbor_query(root.lchild, point)

    def get_nearest_point(self, root, element):
        self.neighbor_query(root, element)
        return self.nearest_point.value, self.nearest_dist

    def query(self, node, point):
        '''
        Judge whether the value of a node in the kd tree is equal to the sample point
        :param k:
        :return:
        '''
        while node != None and node.value != point:
            # Calculate the distance between the target node on the current partition dimension and the single dimension of the current node
            dist = node.value[node.split_dim] - point[node.split_dim]
            # Forward search
            if dist > 0:  # The current node is on the top or left of the target node (in 2D space)
                node = node.lchild
            else:  # Otherwise, the current node is below or to the right of the target node (in 2D space)
                node = node.rchild  # Search right subtree recursively
        
        if node != None:
            return True 
        else:
            return False

    def range(self,rectangle,root):
        list = []
        # the function of range
        def ran(rec=Rectangle, root=Node, split_dim=0):
            if not root:
                return
            # calculate the axis
            axis = split_dim % 2
            # x axis
            if axis == 0:
                # The point is in rectangle
                if rec.lower[0] <= root.value[0] <= rec.upper[0] and rec.lower[1] <= root.value[1] <= rec.upper[1]:
                    list.append(root.value)
                # if root value is smaller than lower point, then search the left tree
                if rec.lower[0] > root.value[0]:
                    ran(rec, root.rchild, split_dim + 1)
                # if root value is bigger than upper point, then search the right tree
                if rec.upper[0] < root.value[0]:
                    ran(rec, root.lchild, split_dim + 1)
                    # if root value is bigger than lower point and smaller than upper point, then search the right tree and finally search the left tree
                if rec.lower[0] <= root.value[0] <= rec.upper[0]:
                    ran(rec, root.rchild, split_dim + 1)
                    ran(rec, root.lchild, split_dim + 1)
            # y axis
            else:
                if rec.lower[0] <= root.value[0] <= rec.upper[0] and rec.lower[1] <= root.value[1] <= rec.upper[1]:
                    list.append(root.value)
                    # if root value is smaller than lower point, then search the right tree
                if rec.lower[1] > root.value[1]:
                    ran(rec, root.rchild, split_dim + 1)
                    # if root value is bigger than root node, then search the left tree
                if rec.upper[1] < root.value[1]:
                    ran(rec, root.lchild, split_dim + 1)
                # if root value is bigger than lower point and smaller than upper point, then search the right tree and finally search the left tree
                if rec.lower[1] <= root.value[1] <= rec.upper[1]:
                    ran(rec, root.rchild, split_dim + 1)
                    ran(rec, root.lchild, split_dim + 1)

        ran(rectangle, root, split_dim=0)
        return list




# Naive method
class Naive_Method():
    def __init__(self,points):
        self.points = points

    def query(self, current_point):
        flag = False
        for point in self.points:
            if current_point ==point:
                flag = True
                break
        return flag

    def range(self,rectangle):
        result1 = [p for p in self.points if rectangle.is_contains(p)]
        return result1




def performance_test():
    points_choose_list = [[x, y] for x in range(1000) for y in range(1000)]

    lower = [500, 500]
    upper = [504, 504]
    rectangle = Rectangle(lower, upper)
    naive_method = Naive_Method(points_choose_list)
    
    # naive method
    start = int(round(time.time() * 1000))
    result1 = naive_method.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')

    # k-d tree
    kd = KD_Tree(points_choose_list)
    root = kd.insert(points_choose_list, 0)
    start = int(round(time.time() * 1000))
    result2 = kd.range(Rectangle([500, 500], [504, 504]), root)
    end = int(round(time.time() * 1000))
    print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)


if __name__ == '__main__':
    performance_test()