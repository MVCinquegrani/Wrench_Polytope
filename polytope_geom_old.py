from scipy.spatial import ConvexHull
import numpy as np

class Polytope:

    def __init__(self, vertices=None,faces=None,face_indices=None,H=None,d=None):
        """
        A constructor of the polytope object

        Args:
            vertices (np.array): vertices of the polytope (optional)
            faces (list): list of triangulated faces of the polytope (optional)
            face_indices (list): list of indices of vertices of the polytope (optional)
            
            for the intersection:
            H (np.array): half-plane representation of the polytope Hx<d - matrix H (optional)
            d (np.array): half-plane representation of the polytope Hx<d- vector d (optional)


        """
        self.vertices = vertices
        self.faces = faces
        self.face_indices = face_indices
        if H is not None and d is not None:
            self.H = H
            self.d = d.reshape(-1,1)
        else:
            self.H = None
            self.d = None




def find_face_indices(vertices):
    """
    A function calculating the face representation of the polytope from the vertex representation
    """
    # if self.vertices is None:
    #     print("No vertex representation of the polytope is available ")
        
    # # if self.face_indices is not None:
    # #     self.faces = face_index_to_vertex(self.vertices,self.face_indices)
    # else:
        # self.face_indices = vertex_to_faces(self.vertices)
        # self.faces = face_index_to_vertex(self.vertices,self.face_indices)

    face_indices, grouped_vertices = vertex_to_faces(vertices)

    print("\n")
    return face_indices, grouped_vertices
    


def vertex_to_faces(vertex):
    """
    Function grouping the vertices to faces using a ConvexHull algorithm

    Args:
        vertex(array):  list of vertices

    Returns:
        faces(array) : list of triangle faces with vertex indexes which form them
    """

    if vertex.shape[0] == 1:
        faces = [0, 1]
    else:        
        hull = ConvexHull(vertex, qhull_options='QJ')
        faces = hull.simplices
        grouped_vertices = [[vertex[idx] for idx in face] for face in faces]
    return faces, grouped_vertices

def display_polytope(grouped_vertices):
    return