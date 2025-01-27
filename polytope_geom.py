"""
Overview
---------

* `Polytope <#pycapacity\.objects\.Polytope>`_: A generic class representing a polytope with different representations (vertex, half-plane, face)
"""

# from pycapacity.algorithms import *
from scipy.spatial import ConvexHull, HalfspaceIntersection, Delaunay
import numpy as np
from cvxopt import matrix
import cvxopt.glpk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

# import cdd
# from pypoman import compute_polytope_vertices
# from sklearn.decomposition import PCA



class Polytope:
    """
    A generic class representing a polytope with different representations (vertex, half-plane, face)

    Vertex representation is a list of vertices of the polytope

    .. math::
        \mathcal{H}\!-\!rep = \{x \in \mathbb{R}^n | Hx \leq d \}

        
    Half-plane representation is a list of inequalities defining the polytope (usefull for intersection)

    .. math::
        \mathcal{V}\!-\!rep = \{x_{v1},~ x_{v2},~ \ldots, ~x_{vn} \}

    Face representation is a list of triangles forming faces of the polytope, each triangle is represented as a list of tree vertices

    .. math::
        \mathcal{F}\!-\!rep = \{ [x_{v1},~ x_{v2}, ~x_{v2}],  ~ \ldots , ~[x_{vi},~ x_{vj}, ~x_{vk}], \ldots \}


    :ivar vertices: vertex representation of the polytope
    :ivar H: half-plane representation of the polytope Hx<d - matrix H
    :ivar d: half-plane representation of the polytope Hx<d- vector d
    :ivar faces: face representation of the polytope - faces are represented as a list of triangulated vertices
    :ivar face_indices: face representation of the polytope - faces are represented as a list of indices of vertices


    Polytope object implements the following operators:

    - ``+`` : minkowski sum of two polytopes
    - ``&`` : intersection of two polytopes

    Examples:
        >>> from pycapacity.objects import *
        >>> import numpy as np
        >>> # create a polytope object
        >>> p = Polytope(vertices=np.random.rand(3,10))
        >>> # find half-plane representation of the polytope
        >>> p.find_halfplanes()
        >>> # find face representation of the polytope
        >>> p.find_faces()
        >>> # find vertex representation of the polytope
        >>> p.find_vertices()
        >>> # create another polytope object
        >>> p1 = Polytope(vertices=np.random.rand(3,10))
        >>> # minkowski sum of two polytopes
        >>> p_sum = p + p1
        >>> # intersection of two polytopes
        >>> p_int = p & p1


    Additionally for robot's force polytopes will have additional attributes:
    
    - **torque_vertices**: (if applicable) joint torques corresponding to the vertices of the polytope

    Additionally for human's force polytopes will have additional attributes:

    - **torque_vertices**: (if applicable) joint torques corresponding to the vertices of the polytope
    - **muscle_forces_vertices**: (if applicable) muscle forces corresponding to the vertices of the polytope

    Human's velocity polytopes will have additional attributes:

    - **dq_vertices**: (if applicable) joint velocities corresponding to the vertices of the polytope
    - **dl_vert**: (if applicable) muscle elongation velocities corresponding to the vertices of the polytope
    """
    
    def __init__(self, vertices=None,faces=None,face_indices=None, grouped_vertices=None,H=None,d=None):
        """
        A constructor of the polytope object

        Args:
            vertices (np.array): vertices of the polytope (optional)
            faces (list): list of triangulated faces of the polytope (optional)
            face_indices (list): list of indices of vertices of the polytope (optional)
            H (np.array): half-plane representation of the polytope Hx<d - matrix H (optional)
            d (np.array): half-plane representation of the polytope Hx<d- vector d (optional)

        """
        self.vertices = vertices
        self.faces = faces
        self.face_indices = face_indices
        self.grouped_vertices = grouped_vertices
        if H is not None and d is not None:
            self.H = H
            self.d = d.reshape(-1,1)
        else:
            self.H = None
            self.d = None
    
    def find_vertices(self):
        """
        A function calculating the vertex representation of the polytope from the half-plane representation

        """
        if self.H is not None and self.d is not None:
            # finding vertices of the polytope from the half-plane representation
            self.vertices, self.face_indices = hspace_to_vertex(self.H,self.d)
        else:
            print("No half-plane representation of the polytope is available")
            self.find_halfplanes()
            self.vertices, self.face_indices = hspace_to_vertex(self.H,self.d)


    def find_grouped_vertices(self):
        if self.vertices is None:
            self.find_vertices()
        if self.face_indices is None:
            self.face_indices = vertex_to_faces(self.vertices)

        self.grouped_vertices = [[self.vertices[:,idx] for idx in face] for face in self.face_indices]

    def find_halfplanes(self):
        """
        A function calculating the half-plane representation of the polytope from the vertex representation
        """
        if self.vertices is not None:
            self.H, self.d = vertex_to_hspace(self.vertices)
        else:
            print("No vertex representation of the polytope is available")


    def find_faces(self):
        """
        A function calculating the face representation of the polytope from the vertex representation
        """
        if self.vertices is None:
            print("No vertex representation of the polytope is available - calculating it")
            self.find_vertices()
            
        if self.face_indices is not None:
            self.faces = face_index_to_vertex(self.vertices,self.face_indices)
        else:
            self.face_indices = vertex_to_faces(self.vertices)
            self.find_grouped_vertices()
            self.faces = face_index_to_vertex(self.vertices,self.face_indices)
        return self.face_indices

    def find_from_point_cloud(self, points):
        """
        A function updating the polytope object from a point cloud it calculates the vertex and half-plane representation of the polytope. 
        
        Note:
            The polytope will be constructed as a convex hull of the point cloud

        Args:
            points (np.array): an array of points forming a point cloud 
        """
        # O riduci la dimenzione e poi la allarghi dinuovo 
        # o fai in modo che se una dimenzione e nulla, semplicemente vai avanti senza importartene
        # o ancora provi a capire teoricamente quale dim e nulla e se i due polytopi possono essere considerati separatamente
        
        
        # rank = np.linalg.matrix_rank(points)
        # pca = PCA(n_components=rank)
        # reduced_points = pca.fit_transform(np.array(points).T)
        # delaunay = Delaunay(np.array(reduced_points))

        # # delaunay = Delaunay(np.array(points).T)
        # hull_indices = np.unique(delaunay.convex_hull)
        # self.vertices = reduced_points[hull_indices]
        # print('a')

        self.H, self.d = vertex_to_hspace(points)

        # coincident_pairs = remove_coincident_halfspaces(self.H, self.d)
        self.vertices, self.face_indices = hspace_to_vertex(self.H,self.d)

    # minkowski sum of two polytopes
    def __add__(self, p):
        if self.vertices is None:
            self.find_vertices()
        if p.vertices is None:
            p.find_vertices()   
        vertices_sum = []
        for v1 in self.vertices.T:
            for v2 in p.vertices.T:
                vertices_sum.append(v1+v2)
        P_sum = Polytope()
        if True:
            points=np.array(vertices_sum).T
            centro = np.mean(points, axis=1)

            # # Trasporre la matrice per lavorare con i punti come righe
            # points_T = points.T

            # # Filtra i punti per mantenere solo quelli unici
            # unique_points = np.unique(points_T, axis=0)

            # # Ritorna la matrice filtrata alla forma originale (6xN)
            # filtered_points = unique_points.T

            # # Calcolo delle distanze tra tutti i punti unici
            # distances = np.linalg.norm(filtered_points[:, :, None] - filtered_points[:, None, :], axis=0)

            # # Imposta la diagonale a infinito per ignorare le distanze con sé stessi
            # np.fill_diagonal(distances, np.inf)

            # # Trova la minima distanza
            # min_distance = np.min(distances)

            # # Lista per mantenere solo i punti accettabili
            # remaining_points = []
            # threshold = 0.01

            # # Itera attraverso i punti e filtra quelli troppo vicini
            # for i, point in enumerate(filtered_points.T):
            #     if len(remaining_points) == 0 or not any(np.linalg.norm(point - np.array(remaining_points), axis=1) <= threshold):
            #         remaining_points.append(point)

            # # Converti la lista di punti rimanenti in una matrice
            # remaining_points = np.array(remaining_points).T
            # remaining_points.shape[1]  # Numero di punti rimanenti


        P_sum.find_from_point_cloud(points=np.array(vertices_sum).T)
        return P_sum
    
    # intersecting two polytopes
    def __and__(self, p):
        if self.H is None or self.d is None:
            self.find_halfplanes()
        if p.H is None or p.d is None:
            p.find_halfplanes()
        H_int = np.vstack((self.H,p.H))
        d_int = np.vstack((self.d,p.d))
        return Polytope(H=H_int,d=d_int)
    
    def intersection(self, poly):
        p_int = self & poly
        return p_int
    

    def display(self,color):

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['lightcoral', 'pink', 'violet', 'plum', 'darkred', 'plum', 'purple', 'darkviolet']
        legend_elements = []
        legeend_labels = []
        p_min = np.zeros(3)
        p_max = np.zeros(3)

        if self.vertices is None:
            self.find_vertices()   
        if self.faces is None:
            self.face_indices = vertex_to_faces(self.vertices)
            self.find_grouped_vertices()
            self.faces = face_index_to_vertex(self.vertices,self.face_indices)

        vertices = np.array(self.vertices)
        ax.scatter(vertices[0,:], vertices[1,:], vertices[2,:])
        for i, verts in enumerate(vertices):
            p_max[i] = np.max(verts)
            p_min[i] = np.min(verts)

        # polytope
        # grouped_vertices = [[self.vertices[idx] for idx in face] for face in self.face_indices]
        ax.add_collection3d(Poly3DCollection(self.grouped_vertices, facecolors=color, linewidths=1.0, edgecolors=(1,0,0,0.1), alpha=.25))     
        plot_patch = Patch(color=color) 
        legend_elements.append((plot_patch))
        legeend_labels.append((" force geometry"))

        ax.set_xlim([p_min[0],p_max[0]])
        ax.set_ylim([p_min[1],p_max[1]])
        ax.set_zlim3d([p_min[2],p_max[2]])

        # Set labels in LaTeX
        ax.set_xlabel(r'$ x [m] $')
        ax.set_ylabel(r'$ y [m] $')
        ax.set_zlabel(r'$ z [m] $')
                                                                                                                                                                                                     
        plt.legend(legend_elements, legeend_labels)
        plt.show()        
        print('/n')


    
class ContactPolytope(Polytope):
    """
    A class representing a Pyramid, which extends the Polytope class
    by adding attributes for contact and foot name.
    
    :ivar contact: Stores contact information associated with the Pyramid.
    :ivar foot_name: Stores the name of the foot associated with the Pyramid.
    """

    def __init__(self, contact=None, foot_name=None, polytope=None, **kwargs):
        """
        Initialize a Pyramid object with additional attributes for contact and foot name.

        Args:
            contact (optional): Contact information (default: None).
            foot_name (str, optional): Name of the foot (default: None).
            **kwargs: Additional arguments to pass to the Polytope constructor.
        """
        if polytope is not None:
            # Initialize the base class using the attributes of the provided Polytope
            super().__init__(
                vertices=polytope.vertices,
                faces=polytope.faces,
                face_indices=polytope.face_indices,
                grouped_vertices=polytope.grouped_vertices,
                H=polytope.H,
                d=polytope.d,
                **kwargs
            )
        else:
            super().__init__(**kwargs)  # Initialize an empty Polytope if no polytope is provided
        
        self.contact = contact
        self.foot_name = foot_name
        self.wrench = None


# -------------------------------------------------------------------------------------------------------------------------------------------------

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
        hull = ConvexHull(vertex.T, qhull_options='QJ')
        faces = hull.simplices
    return faces


def face_index_to_vertex(vertices, indexes):
    """
    Helping function for transforming the list of faces with indexes to the vertices

    Args:
        vertices: list of vertices
        indexes: list of vertex indexes forming faces

    Returns:
        faces: list of faces composed of vertices

    """
    dim = min(np.array(vertices).shape)
    if dim == 2:
        v = vertices[:,np.unique(indexes.flatten())]
        return v[:,order_index(v)]
    else:
        return [vertices[:,face] for face in indexes]

def vertex_to_hspace(vertex):
    """
    Function transforming vertices to half-space representation using a ConvexHull algorithm

    Args:
        vertex(array):  list of vertices

    Returns
    -------
        H(list): matrix of half-space representation `Hx<d`
        d(list): vector of half-space representation `Hx<d`
    """
    try:
        hull = ConvexHull(vertex.T, qhull_options='Q0')
    except:
        hull = ConvexHull(vertex.T, qhull_options='Qbk:0Bk:0')

    return  hull.equations[:,:-1], -hull.equations[:,-1].reshape((-1,1))


def hspace_to_vertex(H,d):
    """
    From half-space representation to the vertex representation

    Args:
        H(list):  
            matrix of half-space representation `Hx<d`
        d(list): 
            vector of half-space representation `Hx<d`
    Returns
    --------
        vertices(list)  : vertices of the polytope
        face_indexes(list) : indexes of vertices forming triangulated faces of the polytope

    """

    # H, d = remove_coincident_halfspaces(H_old, d_old)

    if len(H):
        d = d.reshape(-1,1)
    
        hd_mat = np.hstack((np.array(H),-np.array(d)))
        # calculate a feasible point inside the polytope
        feasible_point = chebyshev_center(H,d)
        # calculate the convex hull
        try:
            hd = HalfspaceIntersection(hd_mat,feasible_point)
            hull = ConvexHull(hd.intersections)
        except:
            print("H2V: Convex hull issue: using QJ option! ")
            try:
                hd = HalfspaceIntersection(hd_mat,feasible_point,qhull_options='QJ')
                hull = ConvexHull(hd.intersections)
            except:
                print("H2V: Convex hull issue: using Q0 option! ")
                hd = HalfspaceIntersection(hd_mat,feasible_point,qhull_options='Q0')
                hull = ConvexHull(hd.intersections)

                # print("ciaaaaa")
                # mat = np.hstack((d.reshape(-1, 1), -H))
                # mat = cdd.Matrix(mat.tolist(), number_type='float')
                # mat.rep_type = cdd.RepType.INEQUALITY

                # # Compute vertices using CDDLib
                # poly = cdd.Polyhedron(mat)
                # generators = poly.get_generators()

                # # Extract vertices
                # vertices = np.array([gen[1:] for gen in generators if gen[0] == 1])  # Type 1 = vertex
                # return vertices,d
            
                # # vertices = compute_polytope_vertices(H, d)
                # # return vertices,d


        # return vertices,d
        return hd.intersections.T, hull.simplices


def chebyshev_center(A,b):
    """
    Calculating chebyshev center of a polytope given the half-space representation

    https://pageperso.lis-lab.fr/~francois.denis/IAAM1/scipy-html-1.0.0/generated/scipy.spatial.HalfspaceIntersection.html

    Args:
        A(list):  
            matrix of half-space representation `Ax<b`
        b(list): 
            vector of half-space representation `Ax<b`
    Returns:
        center(array): returns a chebyshev center of the polytope
    """
    # calculate the chebyshev ball
    c, r = chebyshev_ball(A,b)
    return c


def chebyshev_ball(A,b):
    """
    Calculating chebyshev ball of a polytope given the half-space representation

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html#r9b902253b317-1

    Args:
        A(list):  
            matrix of half-space representation `Ax<b`
        b(list): 
            vector of half-space representation `Ax<b`
    Returns:
        center(array): returns a chebyshev center of the polytope
        radius(float): returns a chebyshev radius of the polytope
    """
    # calculate the vertices
    Ab_mat = np.hstack((np.array(A),-np.array(b)))

    # calculating chebyshev center
    norm_vector = np.reshape(np.linalg.norm(Ab_mat[:, :-1], axis=1), (A.shape[0], 1))
    c = np.zeros((Ab_mat.shape[1],))
    c[-1] = -1
    G = matrix(np.hstack((Ab_mat[:, :-1], norm_vector)))
    h = matrix(- Ab_mat[:, -1:])
    solvers_opt={'tm_lim': 100000, 'msg_lev': 'GLP_MSG_OFF', 'it_lim':10000}
    res = cvxopt.glpk.lp(c=c,  G=G, h=h, options=solvers_opt)
    return np.array(res[1][:-1]).reshape((-1,)), np.array(res[1][-1]).reshape((-1,))


def order_index(points):
    """
    Order clockwise 2D points

    Args:
        points:  matrix of 2D points

    Returns
        indexes(array) : ordered indexes
    """
    px = np.array(points[0,:]).ravel()
    py = np.array(points[1,:]).ravel()
    p_mean = np.array(np.mean(points,axis=1)).ravel()

    angles = np.arctan2( (py-p_mean[1]), (px-p_mean[0]))
    sort_index = np.argsort(angles)
    return sort_index


# def remove_coincident_halfspaces(H, d, tolerance=1e-6):
#     """

#     """

#     num_halfspaces = H.shape[0]
#     coincident_pairs = []
#     # Scorre tutti i semispazi (coppie di righe di H e d)
#     for i in range(num_halfspaces):
#         for j in range(i + 1, num_halfspaces):
#             h1, d1 = H[i], d[i]
#             h2, d2 = H[j], d[j]

#             # Normalizza i vettori normali
#             h1_norm = np.linalg.norm(h1)
#             h2_norm = np.linalg.norm(h2)

#             # Evita vettori nulli
#             if h1_norm < tolerance or h2_norm < tolerance:
#                 continue

#             h1_unit = h1 / h1_norm
#             h2_unit = h2 / h2_norm

#             # Verifica se i vettori normali sono paralleli
#             if not np.allclose(h1_unit, h2_unit, atol=tolerance) and \
#                not np.allclose(h1_unit, -h2_unit, atol=tolerance):
#                 continue

#             # Verifica la proporzionalità dei termini noti
#             lambda_factor = d1 / d2 if abs(d2) > tolerance else None
#             if lambda_factor is not None and np.isclose(d1, lambda_factor * d2, atol=tolerance):
#                 coincident_pairs.append((i, j))

#     components = [-1] * num_halfspaces  # -1 significa "non visitato"
#     component_id = 0

#     # 2. Identifica le componenti connesse  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
#     for i, j in coincident_pairs:
#         if components[i] == -1 and components[j] == -1:
#             # Se entrambi sono non visitati, crea una nuova componente
#             components[i] = component_id
#             components[j] = component_id
#             component_id += 1
#         elif components[i] != -1 and components[j] == -1:
#             # Se i appartiene a una componente ma j no, aggiungi j alla componente di i
#             components[j] = components[i]
#         elif components[j] != -1 and components[i] == -1:
#             # Se j appartiene a una componente ma i no, aggiungi i alla componente di j
#             components[i] = components[j]
#         elif components[i] != components[j]:
#             # Se i e j appartengono a componenti diverse, unisci le due componenti
#             old_id, new_id = components[j], components[i]
#             for k in range(num_halfspaces):
#                 if components[k] == old_id:
#                     components[k] = new_id

#     # Assegna un ID unico per i semispazi non ancora visitati
#     for i in range(num_halfspaces):
#         if components[i] == -1:
#             components[i] = component_id
#             component_id += 1

#     # 3. Seleziona un rappresentante per ogni componente
#     unique_components = {}
#     for i, comp_id in enumerate(components):
#         if comp_id not in unique_components:
#             unique_components[comp_id] = i  # Scegli il primo semispazio di ogni componente        
                

#     # 4. Filtra H e d mantenendo solo i rappresentanti
#     keep_indices = sorted(unique_components.values())  # Ordina gli indici per coerenza
#     H_reduced = H[keep_indices, :]
#     d_reduced = d[keep_indices]

#     return H_reduced, d_reduced            

#     # return coincident_pairs


def remove_coincident_halfspaces(H, d, tolerance=1e-6):
    """
    Rimuove semispazi duplicati (coincidenti) da H e d.
    Un semispazio i coincide con j se i loro vettori normali sono paralleli
    e i termini noti (d_i, d_j) soddisfano la stessa proporzionalità.

    La funzione:
    1. Identifica tutte le coppie (i, j) di semispazi coincidenti.
    2. Costruisce un grafo e raggruppa i semispazi in componenti connesse
       (tutti quelli collegati, direttamente o indirettamente, in un'unica componente).
    3. Per ogni componente, sceglie un solo rappresentante.
    4. Restituisce H e d filtrati mantenendo solo i rappresentanti.
    """

    num_halfspaces = H.shape[0]
    # -------------------------------------------------------------
    # 1) Troviamo tutte le coppie di semispazi coincidenti
    # -------------------------------------------------------------
    coincident_pairs = []
    for i in range(num_halfspaces):
        for j in range(i + 1, num_halfspaces):
            h1, d1 = H[i], d[i]
            h2, d2 = H[j], d[j]

            # Evita vettori normali (quasi) nulli
            norm1 = np.linalg.norm(h1)
            norm2 = np.linalg.norm(h2)
            if norm1 < tolerance or norm2 < tolerance:
                continue

            # Normalizza i vettori
            h1_unit = h1 / norm1
            h2_unit = h2 / norm2

            # Verifica se h1 e h2 sono paralleli/antiparalleli
            if (np.allclose(h1_unit, h2_unit, atol=tolerance) or
                np.allclose(h1_unit, -h2_unit, atol=tolerance)):
                
                # Verifica la proporzionalità dei termini noti
                if abs(d2) > tolerance:
                    lam = d1 / d2
                    # Se d1 ≈ lam * d2 e lam > 0 => i due semispazi coincidono
                    if lam > 0 and np.isclose(d1, lam*d2, atol=tolerance):
                        coincident_pairs.append((i, j))

    # -------------------------------------------------------------
    # 2) Costruiamo il grafo (liste di adiacenza) e cerchiamo le componenti connesse
    # -------------------------------------------------------------
    adjacency_list = [[] for _ in range(num_halfspaces)]
    for (i, j) in coincident_pairs:
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    visited = [False] * num_halfspaces
    components = [-1] * num_halfspaces
    comp_id = 0

    # BFS per trovare tutte le componenti
    for start_node in range(num_halfspaces):
        if not visited[start_node]:
            # Avvia una BFS dalla sorgente start_node
            queue = [start_node]
            visited[start_node] = True
            components[start_node] = comp_id

            while queue:
                current = queue.pop(0)
                for neighbor in adjacency_list[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        components[neighbor] = comp_id
                        queue.append(neighbor)
            
            # Alla fine della BFS, incrementiamo comp_id per la prossima componente
            comp_id += 1

    # -------------------------------------------------------------
    # 3) Selezioniamo un rappresentante per ogni componente
    # -------------------------------------------------------------
    # Per ogni comp_id, prendiamo il primo (o minimo) indice i che ci capita
    representatives = {}
    for i, cid in enumerate(components):
        if cid not in representatives:
            representatives[cid] = i

    # -------------------------------------------------------------
    # 4) Filtriamo H e d, tenendo solo i rappresentanti
    # -------------------------------------------------------------
    keep_indices = sorted(representatives.values())
    H_reduced = H[keep_indices, :]
    d_reduced = d[keep_indices]

    return H_reduced, d_reduced
    # return H, d
