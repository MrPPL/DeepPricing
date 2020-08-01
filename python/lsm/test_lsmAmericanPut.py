import unittest
import numpy as np
import lsmAmericanPut 

class testLsmAmericanPut(unittest.TestCase):
    def test_putCall(self):
        self.assertEqual(lsmAmericanPut.putCall(36,40), 4)
        self.assertEqual(lsmAmericanPut.putCall(40,36), 0)
        self.assertEqual(lsmAmericanPut.putCall(36,36), 0)
        #self.assertEqual(lsmAmericanPut.putCall(-1,-1))
    
    def test_PV(self):
        path1 = [0.00, 0.00, 0.00]
        path2 = [0.00, 0.00, 0.00]
        path3 = [0.00, 0.00, 0.07]
        path4 =[0.17, 0.00, 0.00]
        path5 =[0.00, 0.00, 0.00]
        path6 =[0.34, 0.00, 0.00]
        path7 =[0.18, 0.00, 0.00]
        path8 =[0.22, 0.00, 0.00]
        knownMatrix = np.array( [path1, path2, path3, path4, path5, path6, path7, path8 ])
        #known result from basic example p 120 Longstaff
        self.assertAlmostEqual(lsmAmericanPut.findPV(0.06, knownMatrix, 1), 0.114434330)
        path1 =[1.00, 0.00, 0.00]
        path2 =[0.00, 1.50, 0.00]
        path3 =[0.00, 0.00, 2.25]
        knownMatrix1 = np.array( [path1, path2, path3] )
        self.assertAlmostEqual(lsmAmericanPut.findPV(0.02, knownMatrix1, 5), 1.569072405179642)
        self.assertAlmostEqual(lsmAmericanPut.findPV(0.12, knownMatrix1, 5), 1.4998936353164598)
        self.assertAlmostEqual(lsmAmericanPut.findPV(-0.12, knownMatrix1, 5), 1.6720069416736358)
        knownMatrix2 = np.array([[0,0,0,0,0,3.3,0,2]])
        self.assertAlmostEqual(lsmAmericanPut.findPV(0.2, knownMatrix2, 4), 3.7853402203209474)

    @staticmethod
    def test_CashFlowMatrix():
        #stockMatrix 
        path1 = [1,1.09,1.08,1.34]
        path2 = [1,1.16,1.26,1.54]
        path3 = [1,1.22,1.07,1.03]
        path4 = [1,0.93,0.97,0.92]
        path5 = [1,1.11,1.56,1.52]
        path6 = [1,0.76,0.77,0.90]
        path7 = [1,0.92,0.84,1.01]
        path8 = [1,0.88,1.22,1.34]
        stockMatrix = np.array([path1,path2, path3, path4, path5, path6, path7, path8])
        #Variables for american put
        spot=1
        r=0.06
        vol=0.4
        timePointsYear=3
        T=1
        n=8
        strike = 1.1
        choice = 2

        cashFlowMatrix = lsmAmericanPut.cashflow(spot, r, vol, timePointsYear, strike, T, n, choice, stockMatrix)
        #We know result form Longstaff paper p. 120
        path1 = [0,0,0]
        path2 = [0,0,0]
        path3 = [0,0,0.07]
        path4 = [0.17,0.00,0.00]
        path5 = [0,0,0]
        path6 = [0.34,0.0,0.0]
        path7 = [0.18,0.0,0.0]
        path8 = [0.22,0.0,0.0]
        knownMatrix = np.array([path1,path2, path3, path4, path5, path6, path7, path8])
        np.testing.assert_array_almost_equal(cashFlowMatrix, knownMatrix)
        

if __name__ == '__main__':
    unittest.main()