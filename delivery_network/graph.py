
class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"          
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output

    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 
        Parameters:
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        la complexité est en O(V!) en effet V! est l'ensemble des chemins
        possibles dans le pire des cas l'algorithme parcours tous les chemins
        possibles
        """
        self.graph[node1].append([node2, power_min, dist])
        self.graph[node2].append([node1, power_min, dist])
        self.nb_edges += 1

    def get_path_with_power(self, src, dest, power):
        

        def min_dist(trajets_possibles):
            if trajets_possibles==[]:
                return None
            distances=[]
            for i in range(len(trajets_possibles)):
                s=0
                for j in range(len(trajets_possibles[i])-1) :
                    for k in self.graph[trajets_possibles[i][j]] :
                        if k[0]==trajets_possibles[i][j+1] :
                            s+=k[2] 
                distances.append(s) #liste avec toutes les distances, indexées comme trajets_possibles, mtn on cherche l'indice de la distance la plus petite
            indice_min=[0,distances[0]]
            for i in range(1,len(distances)) :
                if indice_min[1] > distances[i] :
                    indice_min=[i,distances[i]]
            return trajets_possibles[i]


     

        trajets_possibles=[]

        def rec(start, parcourus,chemin): #to be continued
            print(chemin)
            for i in self.graph[start]:
                if i[1] > power :
                    continue #skip ceux nécessitant une trop grande puissance par rapport à notre power
                elif parcourus[i[0]] :
                    continue #skip ceux déjà parcourus
                elif i[0] == dest:
                    parcourus[start]=True
                    chemin.append(start)
                    chemin.append(dest)
                    trajets_possibles.append(chemin)    
                else:
                    parcourus[start]=True
                    chemin.append(start)
                    rec(i[0], parcourus, chemin)

        parcourus=[False]*self.nb_nodes
        parcourus[src]=True
        rec(src, parcourus, [])
        print(trajets_possibles)
        return min_dist(trajets_possibles)
        

    def connected_components(self):
        '''
        La complexité est en O(V+E) car chaques sommets et chaques arêtes sont parcourus au plus une fois
        '''
        deja_vu = []    # liste les sommets dejà vu
        connected_components = []   # liste de liste des composantes connectées

        def parcours_graphe(q):
            '''
            fonction récursive qui parcours une composante connectée du graphe
            '''
            if q==[] :
                return
            
            s = q.pop()[0]
            if s in deja_vu:
                parcours_graphe(q)
            else:
                deja_vu.append(s)
                connected_components[-1].append(s)
                q += self.graph[s]
                parcours_graphe(q)

        for i in self.nodes:
            if i in deja_vu:
                continue
            else:
                connected_components.append([])
                parcours_graphe(self.graph[i])
        return connected_components

    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))

    def min_power(self, src, dest):
        """
        Should return path, min_power.
        """
        raise NotImplementedError


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.
    The file should have the following format:
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min'
        (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.
    Parameters:
    -----------
    filename: str
        The name of the file
    Outputs:
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    files = open(filename, "r")
    for ligne in files:
        ligne = ligne.split(" ")
        if len(ligne) == 2:
            graph = Graph([i+1 for i in range(int(ligne[0]))])
        else:
            if len(ligne) == 4:
                graph.add_edge(int(ligne[0]), int(ligne[1]), int(ligne[2]), int(ligne[3]))
            else:
                graph.add_edge(int(ligne[0]), int(ligne[1]), int(ligne[2]))
    return graph


