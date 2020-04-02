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
        Elements in triangulation. axis 1 contains indices
        of the element nodes in `nodes`

    Attributes
    ----------
    edges: numpy.array, shape (n_edges, 2)
        Unique edges in the triangulation. Axis 1 contains indices
        of the edge nodes in `nodes`
    elem_to_edges: numpy.array, shape (n_elements, 3)
        Matrix mapping elements to their edges
    boundary_edges: numpy.array, shape (n_boundary_edges, 2)
        Boundary edges in triangulation
    """
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = np.sort(elements, axis=1)
        self.n_nodes = nodes.shape[0]
        self.n_elements = elements.shape[0]

        e_ref = np.array([[0,1], [1,2], [0,2]])
        edges = np.concatenate([self.elements[:, e_ref[i,:]] for i in range(e_ref.shape[0])])
        edges_unique, indices, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
        self.edges = edges_unique
        self.elem_to_edges = indices.reshape((3, self.n_elements)).T
        self.boundary_edges = edges_unique[counts==1]

        # Find boundary and internal nodes
        node_idx = range(self.n_nodes)
        self.boundary_node_idx = np.unique(self.boundary_edges.ravel())
        self.interior_node_idx = np.setdiff1d(node_idx, self.boundary_node_idx)
        self.boundary_nodes = self.nodes[self.boundary_node_idx]
        self.internal_nodes = self.nodes[self.interior_node_idx]


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
    
    # Calculate refined mesh
    mesh = uniform_mesh(2)
    print(mesh.nodes)
    print(mesh.boundary_edges)
    print(mesh.boundary_nodes)
    print(mesh.internal_nodes)
    # Visualize mesh
    visualize_mesh(mesh)