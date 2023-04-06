class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.mst = None 
        self.parent_mst = None


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
        #question 1 : implémentation de la méthode add_edge.
        #description des paramètres : les "nodes" correspondent aux sommets (la ville 1 et la ville 2, par exemple).
        #power_min donne la puissance minimale requise pour pouvoir parcourir l'arête reliant les deux sommets du graphe.
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

#QUESTION 2 : écriture de la méthode connected_components_set : elle trouve les composantes du graphe qui sont connectées entre elles.

    def connected_components(self): #étape intermédiaire : seulement le procédé pour UN sommet, puis méthode complète qui parcourt tout le graphe
        '''
        La complexité est en O(V+E) car chaque sommet et chaque arête sont parcourus au plus une fois
        '''
        deja_vu = [False]*self.nb_nodes    # liste les sommets dejà vu
        connected_components = []   # liste de liste des composantes connectées

        def parcours_graphe(q):
            '''
            Cette fonction procède récursivement, en parcourant chaque composante connectée du graphe. 
            On garde en tête que cette méthode récursive est surtout efficace pour de petits graphes (sinon, la procédure est trop lourde).
            '''
            if q==[] :
                return
            
            s = q.pop()[0] #on fait une alternative par rapport au fait d'avoir déjà vu un noeud ou pas encore.
            if deja_vu[s-1]:
                parcours_graphe(q)
            else:
                deja_vu[s-1] = True
                connected_components[-1].append(s)
                q += self.graph[s]
                parcours_graphe(q)

        for i in self.nodes: #boucle parcourant les noeuds du graphe
            if deja_vu[i-1]:
                continue #on ignore, si on l'a déjà traité
            else:
                connected_components.append([])
                parcours_graphe(self.graph[i])
        return connected_components 

    """
    La méthode connected_components() passe par un procédé unitaire : regarder les points connectés à un sommet en particulier.
    On implémente ce procédé récursivement pour évaluer ces connexions pour tous les points du graphe.
    Ainsi, l'output de notre méthode correspond à une liste de listes : pour chaque sommet on obtient la liste de tous les sommets auquels il est connecté.
    """

    def connected_components_set(self): 
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    

    #question 3.
    def get_path_with_power(self, src, dest, power):    
        # Initialisation de la table de distance à l'infini pour tous les noeuds
        distances = {node: float('inf') for node in self.graph}
        # La distance du noeud de départ à lui-même est de 0
        distances[src] = 0
    
        # Initialisation de la file de priorité avec le noeud de départ
        pq = [(0, src)]
        # Initialisation de la table des parents pour chaque noeud
        parents = {src: None}
    
        while pq:
            # Récupération du noeud avec la plus petite distance à partir du début
            (current_distance, current_node) = heapq.heappop(pq)
    
            # Si nous avons atteint la fin, nous avons trouvé le plus court chemin
            if current_node == dest:
                # Construction de la liste des noeuds parcourus
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = parents[current_node]
                path.reverse()
                return path
    
            # Pour chaque noeud voisin du noeud actuel
            for neighbor, p, weight in self.graph[current_node]:
                if p > power:
                    # Elimination des chemins demandant une trop grande puissance
                    continue
            
                # Calcul de la distance de ce noeud voisin par rapport au début
                distance = current_distance + weight
    
                # Si nous avons trouvé une distance plus courte vers ce voisin, l'ajouter à la file de priorité
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
                    # Enregistrement du parent de ce voisin
                    parents[neighbor] = current_node

        # Si nous avons parcouru tous les noeuds et n'avons pas atteint la fin, aucun chemin n'existe
        return None
        

#question 6 (renvoie chemin ET puissance minimale)
    def min_power(self, src, dest):
        """
        Should return path, min_power.
        """
        def suppression_doucblon(liste):
            liste_sans_doublon = []
            for elem in liste:
                if elem not in liste_sans_doublon:
                    liste_sans_doublon.append(elem)
            return liste_sans_doublon
        def récupération_puissance() : 
            puissance=[] 
            for i in range(self.nb_nodes) :
                for j in self.graph[i+1] :
                    puissance.append(j[1])
            puissance = suppression_doucblon(puissance)
            puissance.sort()
            return puissance
        
        puissance=récupération_puissance()

        for i in puissance : 
            chemin=self.get_path_with_power(src, dest, i)
            if chemin != None : 
                
                return chemin, i 
        return None
        """
        On a un trajet donné, on veut le chemin et la puissance nécessaire pour le parcourir.
        L'output de la fonction est un couple de valeurs : le chemin (donc liste des sommets par lesquels on passe), et la puissance requise pour effectuer cela.

          """

#fonction min_power de la séance 2 : min_power avec les arbres de kruskal
    def min_power_k(self,src,dest):
        if self.mst == None :
            self.mst=kruskal(self) 
            deja_vu = [False]*(self.nb_nodes + 1)
            parent = {1 : (1,0,0)}
            pq = [1]
            while pq:
                sommet1 = pq.pop(0)
                if deja_vu[sommet1]:
                    continue
                else:
                    deja_vu[sommet1] = True    
                    for sommet2, puissance, distance in self.graph[sommet1]:
                        if deja_vu[sommet2] :
                            continue
                        pq.append(sommet2)
                        parent[sommet2] = (sommet1, puissance,parent[sommet1][2]+1)
            self.parent_mst=parent
        chemin = [[],0]
        L = []
        if self.parent_mst[dest][2] > self.parent_mst[src][2]:
            fin = True
            n = self.parent_mst[dest][2]-self.parent_mst[src][2]
            for i in range(n):
                L.append(dest)
                chemin[1] = max(self.parent_mst[dest][1], chemin[1])
                dest = self.parent_mst[dest][0]
                L.reverse()
        else:
            fin = False
            n = self.parent_mst[src][2]-self.parent_mst[dest][2]
            for i in range(n):
                L.append(src)
                chemin[1] = max(self.parent_mst[src][1],chemin[1])
                src = self.parent_mst[src][0]
        chemin_src, chemin_dest = [],[]
        while self.parent_mst[src][0] != self.parent_mst[dest][0]:
            parent_src, puissance_src, _ = self.parent_mst[src]
            parent_dest, puissance_dest, _ = self.parent_mst[src]
            chemin_src.append(parent_src)
            chemin_dest.append(parent_dest)
            chemin[1] = max(puissance_src, puissance_dest, chemin[1])
            src,dest = parent_src,parent_dest
        chemin_dest.reverse()
        if fin:
            chemin[0] = chemin_src + [self.parent_mst[src][0]] + chemin_dest + L
        else:
            chemin[0] = L + chemin_src + [self.parent_mst[src][0]] + chemin_dest
        print(chemin)
        print(self.parent_mst)
        return chemin

#QUESTION 4 : modification de la fonction graph_from_file. lecture de la distance d'une arête qui est optionnelle.
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
    i = True
    for ligne in files:
        ligne = ligne.split(" ")
        if i:
            i = False
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
#QUESTION 12 : écriture d'une fonction kruskal()
"""
La classe UnionFind est une structure de données grâce à laquelle on représente une partition d'un ensemble fini.
Dans notre structure en arbre, chaque élément (sommet) est la racine (root) d'un arbre.
Find premet de parcourir l'arbre pour rejoindre cette racine.

find va utiliser un point d'une catégorie, choisi comme representative (par ex pour dire à quelle catégorie appartient tel sommet)
ainsi deux éléments appartiennent au même groupe si et seulement si ils ont le même representative
pour plus de clarté on peut mettre le representative comme racine de l'arbre (notre catégorie) -> d'où l'apparition de la notion de parents
on remonte alors l'arbre en allant vers la racine, en passant de parent en parent
cette remontée s'arrête bien, parce que le parent du representative = lui-même et on dit au programme de se stopper quand le parent d'un sommet = lui-même

union permet de réunir deux ensembles disjoints
avec nos graphes, on veut unir deux arbres, donc on peut poser le representative du premier arbre comme le parent du representative de l'autre arbre.
"""
class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        """
        union permet de réunir deux ensembles disjoints
        le principe est de regrouper des ensembles qui ont la même structure selon un certain critère
        """
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 == root2: #dans ce cas on a déjà les mêmes racines, donc les deux noeuds sont déjà dans le même arbre, rien à faire
            return
        
        #on fait de l'une des racines l'enfant de l'autre racine : les deux arbres sont unis
        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root1] = root2
            if self.rank[root1] == self.rank[root2]:
                self.rank[root2] += 1 
"""
Cette fonction prend en input un graphe (sommets, segments avec puissance minimale pour les parcourir) qui est enregistré selon les critères de la classe Graph.
L'output de la fonction est un autre élément de la classe Graph, qui est un arbre.
"""

def kruskal(g):
    # Création de la liste des arêtes
    edges = []
    for node in g.graph:
        for neighbor, power, dist in g.graph[node]:
            edges.append((power, node, neighbor, dist))

    # Tri de la liste des arêtes par poids croissant
    edges.sort()

    # Initialisation de la structure Union-Find
    nodes = set(g.graph.keys())
    uf = UnionFind(nodes)

    # Initialisation de la liste des arêtes de l'arbre de couverture minimale
    mst_edges = Graph([i for i in g.graph])

    # Parcours de toutes les arêtes triées par ordre croissant de poids
    for power, start, end, dist in edges:
        # Si les sommets de l'arête appartiennent à des ensembles disjoints différents
        if uf.find(start) != uf.find(end):
            # Ajout de l'arête à l'arbre de couverture minimale
            mst_edges.add_edge(start, end, power)
            # Fusion des ensembles disjoints des sommets de l'arête
            uf.union(start, end)

    return mst_edges



"""
Question 15:
    
min_power :
    appelle kruskal qui a une complexité de O(Elog(E)) avec E le nombre d'arêtes
    parcours en O(V^2) l'arbre de couverture minimal avec V le nombre de sommets
la complexité est donc en O(Elog(E)+V¨2)
""" 


#SEANCE 4 : CATALOGUES ET AFFECTATIONS DE CAMIONS
#catalogues téléchargés dans les fichiers trucks.0.in, trucks.1.in, trucks.2.in dans input

#Question 18 : programme qui retourne une collection de camions à acheter et leurs affectations sur des trajets
""" méthodes suggérées :
approche type force brute =  tester toutes les solutions possibles

analogie avec pbm du sac à dos = problème d'optimisation combinatoire.
trouver la combinaison d'éléments la plus précieuse à inclure dans un sac à dos, étant donné un ensemble d'éléments décrits par leurs poids et valeurs.
L'objectif du problème = sélectionner les objets à mettre dans le sac de façon à maximiser la valeur de ts les objets pris,  sous contrainte du poids supporté par le sac.

mais ces méthodes sont assez lourdes -> regarder des méthodes plus rapides pour traiter tous les fichiers d'entrée.
"""

class Camion: #enregistre le format des données concernant les camions, tq dans le catalogue
    def __init__(self, puissance, prix):
        self.puissance = puissance
        self.prix = prix

    
def truck_from_file(filename): #fonction qui s'applique aux fichiers de type trucks dans le dossier input du github
    liste_camion = []   #initialisation d'une liste vide de camions
    file = open(filename,"r")
    i = True
    for ligne in file:
        if i:
            i = False
            continue
        else:
            ligne = ligne.split(" ")
            camion = Camion(int(ligne[0]),int(ligne[1])) #modéliser chaque camion comme un couple de nombres (transtypage de ce qu'on lit dans le catalogue)
            liste_camion.append(camion)     #ajoute ce camion à la liste initialisée au début de la fonction
    return liste_camion     #retourne une liste de camions avec camion=(puissance,prix)


""" IDEES
voir si le budget global permet de couvrir tous les trajets
ensuite, méthode naïve : prendre tous les trajets possibles et traiter tous les cas
plus poussé : passer par le problème du sac à dos. valeur du trajet mais aussi coût (comme le poids)
si même puissance entre deux camions mais différence de prix on supprime du fichier le plus cher
"""

"""30 mars 2023 processing du catalogue de camions pour arriver à modéliser l'optimisation comme un problème du sac à dos
"""
"""METHODE ET BROUILLON
#pre-processing du catalogue trucks, pour enlever les binômes
def catalogue_optimise(filename) : #fonction qui s'applique à un catalogue trucks.in
    liste_optimale=[] #initialisation (celle que l'on veut garder à la fin)
    liste_complete=[] #initialisation de la liste brute de camions

    file = open(filename, "r") #création de la liste NON optimisée
    for ligne in file : 
        ligne=ligne.split(" ")
        camion=Camion(int(ligne[0]), int(ligne[1]))
        liste_complete.append(camion)
    return liste_complete #comme ça on a une liste de vecteurs camion=(puissance, coût)

    #création de la liste OPTIMISEE
    lignes=file.readlines() #on lit toutes les lignes car on veut trouver tous les doublons

    for ligne in lignes : 
        if : #cas où la puissance sur cette ligne n'a pas (encore) de doublon
            liste_optimale.append(ligne) #si pas de doublon, on l'ajoute
        else : #si elle a un ou des doublons, on compare les prix et on remplace les plus chères par celle-là
    #ensuite, on refait la même boucle mais en comparant les prix (un cas sans doublon, un cas où on supprime le plus faible)
    for ligne in lignes : 
        if : #cas où le prix de cette ligne n'a pas (encore) de doublon -> elle est déjà ok sur puissance, elle est ok sur prix, donc on la garde
        else : #elle a des doublons sur le pov du prix, donc on classe selon puissance pour garder la plus puissante

    return liste_optimisée #on récupère une liste sans doublons : pour un prix, on a gardé le camion le plus puissant, pour une puissance, le moins cher
"""


"""ETAPE 1) si on a deux camions qui ont le même prix mais n'ont pas la même puissance, on supprime le plus faible
si on a deux camions qui ont la même puissance mais pas le même prix, on supprime le plus cher
ainsi, on obtient une liste optimisée des camions (avec seulement les meilleurs de chaque doublon)
"""

"""ETAPE 2) autre chose à faire : arriver à décrire chaque trajet (pour l'instant modélisé comme (ville-départ, ville-destination, profit)) comme (coût, profit)
parce que par dest_min_power on peut trouver la puissance minimale nécessaire pour couvrir le trajet 
donc on a une puissance minimale, que l'on peut couvrir pour un coût C grâce aux camions de notre liste optimisée
ainsi on arrivera à trajet=(coût, profit)"""

"""ENFIN on peut résoudre notre problème comme un pbm du sac à dos
-> contrainte : le budget de l'entreprise
-> fonction objectif : le profit (somme des profits de chaque trajet), que l'on veut maximiser

deux manières de résoudre : 
-> les solutions exactes : force brute, ou programme dynamique, mais assez lourd à réaliser.
-> approximations : par exemple ordonner par valeur décroissante les trajets, et les additionner tant que cela respecte la contrainte.
"""


def get_min_power():
    raise NotImplementedError()
    
class Camion: #enregistre le format des données concernant les camions, tq dans le catalogue
    def __init__(self, puissance, prix):
        self.puissance = puissance
        self.prix = prix

    
def truck_from_file(filename): #fonction qui s'applique aux fichiers de type trucks dans le dossier input du github
    liste_camion = []   #initialisation d'une liste vide de camions
    file = open(filename,"r")
    i = True
    for ligne in file:
        if i:
            i = False
            continue
        else:
            ligne = ligne.split(" ")
            camion = Camion(int(ligne[0]),int(ligne[1])) #modéliser chaque camion comme un couple de nombres (transtypage de ce qu'on lit dans le catalogue)
            liste_camion.append(camion)     #ajoute ce camion à la liste initialisée au début de la fonction
    return liste_camion     #retourne une liste de camions avec camion=(puissance,prix)


def catalogue_optimise(liste_camion) : #fonction qui s'applique à une liste de camions
    liste_optimale=[] #initialisation (celle que l'on veut garder à la fin)
    #création de la liste OPTIMISEE
    for camion in liste_camion: #optimisation de la puissance pour un même prix
        b = True
        puissance = camion.puissance
        prix = camion.prix
        for i in range(len(liste_optimale)): 
            if liste_optimale[i].puissance <= puissance and liste_optimale[i].prix <= prix:
                liste_optimale.pop(i)
                liste_optimale.append(camion)
                b = False
                break
            elif liste_optimale[i].puissance == puissance and liste_optimale[i].prix < prix:
                b = False
                break
            else:
                continue
        if b:
            liste_optimale.append(camion)
            
    liste_camion = liste_optimale.copy()
    liste_optimale = []
    
    for camion in liste_camion:#optimisation du prix pour une même puissance
        b = True
        puissance = camion.puissance
        prix = camion.prix
        for i in range(len(liste_optimale)):
            if liste_optimale[i].prix == prix and liste_optimale[i].puissance > puissance:
                b = False
                break
            elif liste_optimale[i].prix >= prix and liste_optimale[i].puissance < puissance:
                liste_optimale.pop(i)
                liste_optimale.append(camion)
                b = False
                break
            else:
                continue
        if b:
            liste_optimale.append(camion)
    
    return liste_optimale

def optimisation(liste_camion,budget,liste_trajet):
    liste_profit = []
    for s1,s2,profit in liste_trajet:
        liste_profit.append([get_min_power(s1,s2),profit,[s1,s2]])
    liste_camion = sorted(liste_camion,lambda camion : camion.puisssance)
    #liste camion est telle que pour chaque puissance on a le prix minimal du camion qui la parcourt
    profit_prix = []
    #création d'une liste donnant le rapport maximum entre le profit et le prix pour chaque trajet
    for power,profit,trajet in liste_profit:
        for camion in liste_camion:
            if camion.power > power:
                profit_prix.append([profit/camion.prix,camion.prix,trajet,camion])
                break
            else:
                continue
    profit_prix = list(reversed(sorted(profit_prix)))
    #tri de la liste par ordre décroissant
    s = 0
    affectation = []
    #on complète affectation avec en priorité le meilleurs rapport profit/prix
    #solution non optimale mais de complexité linéaire en la longueur de profit_prix
    #complexité est O(C*T) avec C le nombre de Camion et T le nombre de Trajet
    for _,prix,trajet,camion in profit_prix:
        if s + prix < budget:
            affectation.append([camion,trajet])
            s += prix
        else:
            continue
    return affectation




