
import unittest
import numpy as np
import simPathStock

class testLsmAmericanPut(unittest.TestCase):
    def test_stockStep(self):
        sim = 0.69133305 #simulate normal random variable
        self.assertAlmostEqual(simPathStock.stockStep(S=50, r=0.09, vol=0.2, timeStep=0.2, normRV=sim),53.93921223968333)

if __name__ == '__main__':
    unittest.main()