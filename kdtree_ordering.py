import numpy as np
class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kd_tree(points, depth=0):
    if not points:
        return None
    
    k = len(points[0])  # Assumes all points have the same dimension
    axis = depth % k
    
    points.sort(key=lambda x: x[axis])
    median = len(points) // 2
    
    return Node(
        point=points[median],
        left=build_kd_tree(points[:median], depth + 1),
        right=build_kd_tree(points[median + 1:], depth + 1)
    )

def in_order_traversal(node, result=None):
    if result is None:
        result = []
    if node:
        in_order_traversal(node.left, result)
        result.append(node.point)
        in_order_traversal(node.right, result)
    return result



def generate_kd_tree(points):
    # Generate the list of 2D points

    # Build the KD-tree
    kd_tree = build_kd_tree(points)

    # Get the KD-tree ordering via in-order traversal
    ordered_points = in_order_traversal(kd_tree)

    return np.array(ordered_points)