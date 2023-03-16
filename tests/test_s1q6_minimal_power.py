# This will work if ran from the root folder.
import sys 

sys.path.append("delivery_network")

from graph import graph_from_file, min_power
import unittest   # The test framework

class Test_MinimalPower(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file("input/network.00.in")
        self.assertEqual(min_power(g,[1, 4])[1], 11)
        #self.assertEqual(min_power(g,[2, 4])[1], 10)

    def test_network1(self):
        g = graph_from_file("input/network.04.in")
        #self.assertEqual(min_power(g,[1, 4])[1], 4)

if __name__ == '__main__':
    unittest.main()
