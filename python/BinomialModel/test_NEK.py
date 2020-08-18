import unittest
import numpy as np
import NEK

class testNEK(unittest.TestCase):
    def test_EurCallMax(self):
        steps=2
        T=1
        r=0.1
        strike = 10
        Tree1 = {1:np.array([[5,8,100],[-50,3,-6]])}
        self.assertEqual(NEK.EurCalMax(steps, Tree1, T, r, strike), 40.71768381161818)

    @staticmethod
    def test_Tree():
        #stockMatrix 
        A = 1
        B = 1
        u=np.exp(0.5)
        d=np.exp(-3/2)
        #after two timeperiods from inception at time 0
        out1 = [d**2*A, d**2*B]
        out2 = [d**2*A,u*d*B]
        out3 = [d**2*A,u**2*B]
        out4 = [u*d*A, d**2*B]
        out5 = [u*d*A, u*d*B]
        out6 = [u*d*A, u**2*B]
        out7 = [u**2*A, d**2*B]
        out8 = [u**2*A,d*u*B]
        out9 = [u**2*A,u**2*B]
        #known result
        Matrix = np.array([out1,out2, out3, out4, out5, out6, out7, out8, out9])
        vol=1
        corr=0
        d=2
        r=0
        stepSize=1
        spot=1
        steps=3
        corrMatrix = NEK.createCorrM(vol=vol, corr = corr, d=d)
        lattice = NEK.makeLattice(timesteps=steps,d=d)
        Tree = NEK.createTree(corrMatrix=corrMatrix, r=r, d=d, vol=vol, stepSize=stepSize, lattice=lattice, spot=spot)
        #
        np.testing.assert_array_almost_equal(Matrix, Tree[steps-1])
        

if __name__ == '__main__':
    unittest.main()