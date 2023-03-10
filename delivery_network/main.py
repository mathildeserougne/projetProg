from graph import Graph, graph_from_file
import sys
sys.setrecursionlimit = 10**10

data_path = "input/"
file_name = "network.6.in"

g = graph_from_file(data_path + file_name)
print(g)
p=g.get_path_with_power(1, 17264, 10**10)
print(p)
