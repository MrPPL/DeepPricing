import unittest

def test_cashFlow():
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

    cashFlowMatrix = cashflow(spot, r, vol, timePointsYear, strike, T, n)
    #We know result form Longstaff paper p. 120
    path1 = [0,0,0]
    path2 = [0,0,0]
    path3 = [0,0,0.07]
    path4 = [0.17,0.00,0.00]
    path5 = [0,0,0]
    path6 = [0,34,0.0,0.0]
    path7 = [0.18,0.0,0.0]
    path8 = [0.22,0.0,0.0]
    knownMatrix = np.array([path1,path2, path3, path4, path5, path6, path7, path8])
    assert knownMatrix==cashFlowMatrix

if __name__ == "__main__":
    test_cashFlow()
    print("Everything passed")