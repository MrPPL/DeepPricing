import numpy as np
import matplotlib.pyplot as plt
import torch
# create dataset
dataTrain = np.loadtxt("./deepLearning/hirsa19/data/mediumCEuroDataTrain.csv", delimiter=',', dtype=np.float32, skiprows=1)
dataTest = np.loadtxt("./deepLearning/hirsa19/data/mediumCEuroDataTest.csv", delimiter=',', dtype=np.float32, skiprows=1)
# here the first column is the class label, the rest are the features
X_train = torch.from_numpy(dataTrain[:, 2:]) # size [n_samples, n_features]
Y_train = torch.from_numpy(dataTrain[:, [1]]) # size [n_samples, 1]
X_test = torch.from_numpy(dataTest[:, 2:]) # size [n_samples, n_features]
Y_test = torch.from_numpy(dataTest[:, [1]]) # size [n_samples, 1]

#Design model
# polynomial regression in sklearn
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression

# simple wrapper class for multi-dimensional polynomial regression
class PolyReg:
    def __init__(self, X, Y, degree):
        
        # create monomials
        self.features = PolynomialFeatures(degree = degree)  
        self.monomials = self.features.fit_transform(X)
        
        # regress with normal equation
        self.model = LinearRegression()  
        self.model.fit(self.monomials, Y)

    def predict(self, x):
        # predict with dot product
        monomials = self.features.fit_transform(x)
        return self.model.predict(monomials)

# run regressions of degree = 1 to 6
polyRegs = [PolyReg(X_train, Y_train, degree) for degree in range(1,7)]

#predictions
poly = [polyReg.predict(X_test) for polyReg in polyRegs]

#model performance
# calculate mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

mse = [mean_squared_error(Y_test, model) for model in poly]
mae = [mean_absolute_error(Y_test, model) for model in poly]
rmse = [np.sqrt(mean_squared_error(Y_test, model)) for model in poly]
coefDet = [r2_score(Y_test, model) for model in poly]
print("MSE", mse)
print("MAE", mae)
print("RMSE", rmse)
print("Coefficient of determination", coefDet)
# Plot
from matplotlib import rcParams

# display
plt.style.use('ggplot')
fig, ax = plt.subplots(2, 3)
fig.set_size_inches(6, 7)
fig.suptitle("Polynomial Regression Vs. Actual Targets")
#rcParams['figure.figsize']=6,4
for i in [0,1]:
    for j in [0,1,2]:
        polIdx = 3*i + j
        ax[i,j].set_title("degree: " + str(1 + polIdx))
        ax[i,j].scatter(poly[polIdx], Y_test, alpha=0.5, s=1)
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/latex/Figures/polynomialEuroC.png")
plt.show()

