# Classical Option Pricing Theory and Extensions to Deep Learning
The GitHub serves two purposes;
1) To keep track of my work
2) For the interested reader of my master's thesis to see the source code

Files for pricing:

The code is written in the Python directory and the following files are used:
1. European call and put option:
ClosedForm directory and the file closedEuro.py
2. Exotic European options:
ClosedForm directory and the file rainbow2Dim.py
3. CRR:
Directory BinomialModel and file BinoHull.py
4. BEG:
Directory BinomialModel and file BEGTwoDim.py
5. LSM:
Directory DeepStopping and file AmericanPut.py (Note that the LSM is implemented by using luphord GitHub repository https://github.com/luphord/longstaff_schwartz)
6. MLP I:
Directory DeepStopping and file AmericanPut.py (Note that the MLP is my own build upon the LSM method)
7. MLP II:
Directory DeepLearning/hirsa19/evaluation and file MLPsIIPricing.py
