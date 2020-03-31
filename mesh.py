import numpy as np
import matplotlib.pyplot as plt
import time

"""Module for creating and manipulating triangular meshes. 
The contents are based on original MATLAB implementations by
Antti Hannukainen (Aalto university) and Mika Juntunen (TKK)."""

class TriangularMesh:
    """Triangular mesh constructed from given triangulation.
    
    Parameters
    ----------
    nodes: numpy.array, shape (n_nodes, 2)
        Nodes in triangulation
    elements: numpy.array, shape (n_elements, 3)
        Elements in triangulation

    Attributes
    ----------
    edges: numpy.array, shape (n_edges, 2)
        Unique edges in the triangulation
    """
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = np.sort(elements, axis=1)
        self.n_nodes = nodes.shape[0]
        self.n_elements = elements.shape[0]
        self.edges, self.elem_to_edges = self.__get_edges(self.nodes, self.elements)

    def __get_edges(self, nodes, elements):
        """Returns the (unique) edges of given triangulation and map from the triangulation
        elements to the edges"""
        e_ref = np.array([[0,1], [1,2], [0,2]])
        edges = np.concatenate([elements[:, e_ref[i,:]] for i in range(e_ref.shape[0])])
        edges_unique, indices = np.unique(edges, axis=0, return_inverse=True)
        elem_to_edges = indices.reshape((3, self.n_elements)).T
        return edges_unique, elem_to_edges

    def visualize(self): 
        visualize_mesh(self)


def uniform_mesh(n):
    """Creates a uniform triangulation on the unit square
    
    Parameters
    ----------
    n: int
        Defines the grade of refinement. The resulting mesh
        has (2^n + 1)*(2^n + 1) nodes. 

    Returns
    -------
    TriangularMesh
        Uniform triangularized mesh
    """
    nodes = np.array([[0,0], [1,0], [0,1], [1,1]])
    elements = np.array([[0, 1, 3], [0, 3, 2]])
    mesh = TriangularMesh(nodes, elements)
    for i in range(n):
        mesh = refined_mesh(mesh)
    return mesh

def refined_mesh(mesh):
    """Uniformly refines triangular mesh to a finer grid
    
    Parameters
    ----------
    mesh: TriangularMesh
        original mesh to refine

    Returns
    -------
    TriangularMesh
        new refined mesh
    """
    nodes = mesh.nodes
    elements = mesh.elements
    edges = mesh.edges
    elem_to_edges = mesh.elem_to_edges
    n_nodes = nodes.shape[0]
    n_elements = elements.shape
    
    # Create nodes for refined mesh
    edge_nodes = (nodes[edges[:,0],:] + nodes[edges[:,1],:]) / 2
    nodes_refined = np.concatenate((nodes, edge_nodes))

    # Create elements for refined mesh
    elems_1 = np.array([elements[:,0],
                        elem_to_edges[:,0] + n_nodes,
                        elem_to_edges[:,2] + n_nodes]).T
    elems_2 = np.array([elements[:,1],
                        elem_to_edges[:,0] + n_nodes,
                        elem_to_edges[:,1] + n_nodes]).T
    elems_3 = np.array([elements[:,2],
                        elem_to_edges[:,2] + n_nodes,
                        elem_to_edges[:,1] + n_nodes]).T
    elems_4 = np.array([elem_to_edges[:,0] + n_nodes,
                        elem_to_edges[:,1] + n_nodes,
                        elem_to_edges[:,2] + n_nodes]).T
    elements_refined = np.concatenate((elems_1, elems_2, elems_3, elems_4))

    mesh_refined = TriangularMesh(nodes_refined, elements_refined)
    return mesh_refined                             


def visualize_mesh(mesh):
    """Function for visualizing a mesh
    
    Parameters
    ----------
    mesh:
    """
    nodes = mesh.nodes
    edges = mesh.edges
    edge_coords = nodes[edges]
    plt.figure()
    plt.scatter(nodes[:,0], nodes[:,1])
    for edge in edge_coords:
        plt.plot(edge[:,0], edge[:,1], color="orange")
    plt.show()
    

if __name__ == "__main__":
    p = np.array([[0,0], [1,0], [0.5,1]])
    t = np.array([[0, 1, 2]])
    mesh = TriangularMesh(p, t)
    refined_mesh(refined_mesh(mesh)).visualize()
    start = time.time()
    uniform_mesh(5)
    end = time.time()
    print(end-start)
    #print(t)
    #print(np.sort(t, axis=1))