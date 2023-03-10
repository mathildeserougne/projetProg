class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.arret = True #initialisation d'une variable globale au sein de la classe pour stopper les ramifications de recherche du bon chemin
        self.chemin = None 


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

    #question 3.
    #elle passe les tests mais ne marche pas :)
    # #on cherche un moyen de stopper toutes les ramifications une fois qu'on a trouvé la solution.
    def get_path_with_power(self, src, dest, power):    
        dejà_vu = [False]*(self.nb_nodes+1)
        self.arret=True

        def rec(s,dejà_vu,parcourus): 
            if self.arret : 
                adj = self.graph[s]
                for i in adj:
                    if i[1] > power or dejà_vu[i[0]]:
                        continue
                    elif i[0] == dest:
                        self.arret=False
                        parcourus += [s,i[0]]                 
                        self.chemin=parcourus 
                    else: 
                        if self.arret : 
                            dejà_vu[i[0]] = True
                            parcourus.append(s)
                            rec(i[0],dejà_vu,parcourus)

        rec(src, dejà_vu, [])
        return self.chemin
        

#fonction de la question 2
    def connected_components(self):
        '''
        La complexité est en O(V+E) car chaques sommets et chaques arêtes sont parcourus au plus une fois
        '''
        deja_vu = [False]*self.nb_nodes    # liste les sommets dejà vu
        connected_components = []   # liste de liste des composantes connectées

        def parcours_graphe(q):
            '''
            fonction récursive qui parcours une composante connectée du graphe
            '''
            if q==[] :
                return
            
            s = q.pop()[0]
            if deja_vu[s-1]:
                parcours_graphe(q)
            else:
                deja_vu[s-1] = True
                connected_components[-1].append(s)
                q += self.graph[s]
                parcours_graphe(q)

        for i in self.nodes:
            if deja_vu[i-1]:
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

#question 6 (renvoie chemin ET puissance minimale)
    def min_power(self, src, dest):
        """
        Should return path, min_power.
        """
        def récupération_puissance() : 
            puissance=[] 
            for i in range(self.nb_nodes) :
                for j in self.graph[i+1] :
                    puissance.append(j[1])
            puissance.sort()
            puissance=set(puissance) #retirer les doublons
            print(puissance)
            return puissance
        
        puissance=récupération_puissance()

        for i in puissance : 
            chemin=self.get_path_with_power(src, dest, i)
            print(chemin)
            if chemin != None : 
                
                return chemin, i
        return None





       

        

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

'''
Séance 2
'''

def kruskal(g):
    # Créer un ensemble de tous les sommets du graphe
    vertices = set()
    for u in range(g.nb_nodes):
        vertices.add(u+1)

    # Trier toutes les arêtes par ordre croissant de poids
    edges = []
    for u in range(g.nb_nodes):
        for v, p, w in g.graph[u+1]:
            edges.append((w, u, v, p))
    edges.sort()

    # Créer une structure de données pour stocker l'ensemble de la forêt
    # d'arbre couvrant de poids minimal
    mst = Graph([i+1 for i in range(g.nb_nodes)])

    # Effectuer l'algorithme de Kruskal
    for w, u, v, p in edges:
        if find(u) != find(v):
            union(u, v)
            mst.add_edge(u, v, p, w)

    return mst

# Fonctions auxiliaires pour l'implémentation de l'algorithme de Kruskal
parent = {}
rank = {}

def find(u):
    if parent[u] != u:
        parent[u] = find(parent[u])
    return parent[u]

def union(u, v):
    root_u = find(u)
    root_v = find(v)
    if root_u != root_v:
        if rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_u] = root_v
            if rank[root_u] == rank[root_v]:
                rank[root_v] += 1

def min_power(trajet, graph):
    max_puissance = float('-inf') # Initialisation du maximum à une valeur très petite
    
    for i in range(len(trajet)-1): # Parcours du trajet
        sommet_a, sommet_b = trajet[i], trajet[i+1]
        for edge in graph[sommet_a]: # Parcours des arêtes adjacentes à sommet_a
            if edge[0] == sommet_b: # Si l'arête relie sommet_a et sommet_b
                max_puissance = max(max_puissance, edge[1]) # Mise à jour du maximum
                
    return max_puissance