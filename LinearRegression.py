from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
# from sklearn.compose import TransformedTargetRegressor
from sklearn.utils import shuffle
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

from numpy import append, nan, where, array, log, exp, corrcoef, transpose, zeros, ones, sqrt, prod
from scipy.stats import pearsonr

version = '0.0.3'
comment = 'Added self.learned condition'


class DataBuilder(object):

    # object to transform dict with data to np.arrays for ml models
    def __init__(self, target, order):
        self.target = target  # define target
        self.order = order  # define feature order for array construction (list of feature names)
        self.featureArr = None
        self.targetArr = None

    def addSample(self, sampleDict):
        if self.target in sampleDict:  # if target exists in the dict
            if sampleDict[self.target] is not None:  # if target value is defined
                featList = []
                for featName in self.order:  # construct list with feature values in order as defined in self.order
                    if featName in sampleDict:  # if feature exists in dict append its value to the list
                        featList.append(sampleDict[featName])
                    else:  # if feature does not exist in dict append None to the list
                        featList.append(None)
                if self.featureArr is None:  # if we are adding for the first time define the targetArr & featureArr
                    self.targetArr = array([[sampleDict[self.target]]])
                    self.featureArr = array([featList])
                else:  # if we are not adding the first time append the values to the existing arrays
                    self.targetArr = append(self.targetArr, [[sampleDict[self.target]]], axis=0)
                    self.featureArr = append(self.featureArr, [featList], axis=0)

    def addSamples(self, samples):
        if type(samples) == dict:
            for sampleName in samples:
                self.addSample(samples[sampleName])
        else:
            for sample in samples:
                self.addSample(sample)

    def delSamples(self):
        self.featureArr = None
        self.targetArr = None

    def getFeatureArr(self):
        return self.featureArr

    def getTargetArr(self):
        return self.targetArr

    def getIntervals(self, intervalDict):
        intervals = []
        for feature in self.order:
            if feature in intervalDict:
                intervals.append(intervalDict[feature])
            else:
                intervals.append((0, 0))
        return intervals


class LinearRegressionModel(object):

    def __init__(self, strategy='mean', testSize=0.3):
        self.samplesX = None
        self.samplesY = None
        self.testSize = testSize
        # simple imputer object to fill missing data: https://scikit-learn.org/stable/modules/impute.html
        self.imputer = SimpleImputer(missing_values=nan, strategy=strategy)
        # standard scaler object for mean removal and variance scaling:
        # https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
        self.scalerX = StandardScaler()  # for features
        self.scalerY = StandardScaler()  # for result (target)
        # feature selector object:
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
        self.featureSelector = VarianceThreshold()
        # transformed target regressor object with linear regression:
        # https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        # https://stackoverflow.com/questions/38058774/scikit-learn-how-to-scale-back-the-y-predicted-result
        self.linReg = LinearRegression(fit_intercept=False)
        self.ccP = None
        self.cc = None
        self.learned = False
        self.modelSampleSize = 0
        self.initialShape = (0, 0)
        self.testRMSE = None
        self.testR2 = None
        self.allEmpty = []

    def addSamples(self, samplesXarr, samplesYarr):
        fullIndices = where(samplesYarr != None)[0]  # get samples indices which do not have empty target values
        if self.samplesX:  # if we have some samples stored already append to the samplesXarr and samplesYarr arrays
            self.samplesX = append(self.samplesX, samplesXarr[fullIndices], axis=0)
            self.samplesY = append(self.samplesX, samplesYarr[fullIndices], axis=0)
        else:  # if we are adding the first time set samplesX and samplesY
            self.samplesX = samplesXarr[fullIndices]
            self.samplesY = samplesYarr[fullIndices]
        self.initialShape = self.samplesX.shape

    def learn(self):
        # learn only if we added some samples
        if self.samplesX is not None:
            # shuffle samples: https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
            X, y = shuffle(self.samplesX, self.samplesY, random_state=0)
            fullIndices = where(X[:, 0] != None)[0]
            X = X[fullIndices]
            y = y[fullIndices]

            nAll = X.shape[0]
            nTest = int(nAll * self.testSize)
            nTrain = nAll - nTest
            if nTrain > 1.5*X.shape[1]:
                for i in range(X.shape[1]):
                    # find all features with all values equal to None and replace them with constant value 0
                    if X[:, i].any() == None:
                        self.allEmpty.append(i)
                        X[:, i] = zeros(X.shape[0])
                # fill missing data with samples means
                X = self.imputer.fit_transform(X)
                # select only non constant features
                X = self.featureSelector.fit_transform(X)
                # scale data
                X = self.scalerX.fit_transform(X)
                yTest = y[-nTest:]
                y = self.transformFun(y)
                # save the final model sample size
                self.modelSampleSize = nAll
                # fit model to data
                self.linReg.fit(X[:-nTest], y[:-nTest])
                self.learned = True
                # calculate predictions on test samples
                predRaw = self.linReg.predict(X[-nTest:])
                if max(predRaw) > 11:
                    tooLarge = where(predRaw > 11)[0]
                    predRaw[tooLarge] = ones((tooLarge.shape[0],1 ))*13
                yTest_pred = self.inverseTransformFun(predRaw)

                self.cc = corrcoef(append(X, y, axis=1))
                self.ccP = []
                for i, x in enumerate(transpose(X)):
                    self.ccP.append(list(pearsonr(x, y[:, 0])))
                # calculate root mean squared error on test samples
                self.ccP = array(self.ccP)[:, 0]

                try:
                    self.testRMSE = sqrt(mean_squared_error(yTest, yTest_pred))
                except Exception as e:
                    print(yTest_pred)
                    print(self.linReg.predict(X[-nTest:]))
                    print(X[-nTest:])
                    raise ValueError(e)
                # calculate r2 on test samples
                self.testR2 = r2_score(yTest, yTest_pred)

                return self.testR2
        return None

    def predict(self, X):
        if self.learned:
            # fill missing data with samples (train) means
            X = self.imputer.transform(X)
            # select only non constant (train) features
            X = self.featureSelector.transform(X)
            # scale data
            X = self.scalerX.transform(X)
            # predict and return result
            return self.inverseTransformFun(self.linReg.predict(X))
        return None

    def getInfluenceFactors(self, norm=False):
        if self.learned:
            if norm:  # if we want to normalize
                coef = self.linReg.coef_[0]  # get solution coefficients
                sumCoef = sum(abs(coef))  # calculate their absolute sum
                coef = coef/sumCoef  # normalize them
            else:
                coef = self.linReg.coef_[0]  # get solution coefficients
            i = 0  # for tracking indices of coeficients
            inflFac = []  # influence factor list

            for selected in self.featureSelector.get_support():  # iterate trough featureSelector selection
                if selected:  # if feature was included in the model add its coefficient to the influence factor list
                    inflFac.append(coef[i])
                    i += 1  # update i
                else:  # if feature was not included set its influence factor to zero
                    inflFac.append(0)
            return inflFac
        return None

    def getNormVal(self, featIntervals):
        if self.learned:
            # transform featIntervals with X scaler
            intervalArr = zeros((2, len(featIntervals)))
            for i, interval in enumerate(featIntervals):
                intervalArr[0, i] = interval[0]
                intervalArr[1, i] = interval[1]
            intervalArr = self.featureSelector.transform(intervalArr)
            intervalArr = self.scalerX.transform(intervalArr)
            # calculate norm value
            normVal = 0
            k = 0
            # print(self.getInfluenceFactors())
            intgProd = prod(intervalArr[1] - intervalArr[0]) # TODO remove redundant operations
            intervalArr = transpose(intervalArr)
            for i, inflFac in enumerate(self.getInfluenceFactors()):
                if inflFac:  # if influence factor is not 0 (is included in the model) calculate partial integral
                    partialIntegral = inflFac
                    for j, interval in enumerate(intervalArr):
                        if j == i - k:
                            partialIntegral *= (interval[1] ** 2 - interval[0] ** 2) / 2
                        else:
                            partialIntegral *= (interval[1] - interval[0])  # TODO remove redundant operations
                    normVal += partialIntegral / intgProd
                else:
                    k += 1
            return self.inverseTransformFun([[normVal]])[0][0]
        return None

    def transformFun(self, value):
        return self.scalerY.fit_transform(log(value))

    def inverseTransformFun(self, value):
        return exp(self.scalerY.inverse_transform(value))

    def getSampleSize(self):
        return self.modelSampleSize


class CrossValidationModel(object):

    def __init__(self, strategy='mean', testSize=0.3):
        self.samplesX = None
        self.samplesY = None
        self.testSize = testSize
        # simple imputer object to fill missing data: https://scikit-learn.org/stable/modules/impute.html
        self.imputer = SimpleImputer(missing_values=nan, strategy=strategy)
        # standard scaler object for mean removal and variance scaling:
        # https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
        self.scalerX = StandardScaler()  # for features
        self.scalerY = StandardScaler()  # for result (target)
        # feature selector object:
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
        self.featureSelector = VarianceThreshold()
        # transformed target regressor object with linear regression:
        # https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        # https://stackoverflow.com/questions/38058774/scikit-learn-how-to-scale-back-the-y-predicted-result
        self.linReg = LinearRegression(fit_intercept=False)
        self.initialShape = ()
        self.ccP = []
        self.testR2 = None
        self.allEmpty = []

    def addSamples(self, samplesXarr, samplesYarr):
        fullIndices = where(samplesYarr != None)[0]  # get samples indices which do not have empty target values
        if self.samplesX:  # if we have some samples stored already append to the samplesXarr and samplesYarr arrays
            self.samplesX = append(self.samplesX, samplesXarr[fullIndices], axis=0)
            self.samplesY = append(self.samplesX, samplesYarr[fullIndices], axis=0)
        else:  # if we are adding the first time set samplesX and samplesY
            self.samplesX = samplesXarr[fullIndices]
            self.samplesY = samplesYarr[fullIndices]
        self.initialShape = self.samplesX.shape

    def crossValidate(self):
        # validate only if we added some samples
        if self.samplesX is not None:
            nAll = len(self.samplesX)
            nTest = int(nAll*self.testSize)
            nTrain = nAll - nTest
            # shuffle samples: https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
            X, y = shuffle(self.samplesX, self.samplesY, random_state=0)
            fullIndices = where(X[:, 0] != None)[0]
            X = X[fullIndices]
            y = y[fullIndices]
            # find all features with all values equal to None and replace them with constant value 0
            for i in range(X.shape[1]):
                if X[:, i].any() == None:
                    self.allEmpty.append(i)
                    X[:, i] = zeros(X.shape[0])
            # fill missing data with samples means
            X = self.imputer.fit_transform(X)
            # select only non constant features
            X = self.featureSelector.fit_transform(X)
            # scale data
            X = self.scalerX.fit_transform(X)
            y = self.transformFun(y)
            # fit model to data
            self.linReg.fit(X[:-nTest], y[:-nTest])
            # self.cc = corrcoef(append(X, y, axis=1))
            # get Pearsons correlation coefficient
            for i, x in enumerate(transpose(X)):
                self.ccP.append(list(pearsonr(x, y[:, 0])))
            self.ccP = array(self.ccP)[:, 0]
            # get scores from cross validation
            scores = cross_validate(self.linReg, X, y, cv=10,
                                    scoring=('r2', 'neg_mean_squared_error'),
                                    return_train_score=True)
            self.testR2 = scores['test_r2'].mean()
            return self.testR2
        return None

    def getInfluenceFactors(self, norm=False):
        if norm:  # if we want to normalize
            coef = self.linReg.coef_[0]  # get solution coefficients
            sumCoef = sum(abs(coef))  # calculate their absolute sum
            coef = coef/sumCoef  # normalize them
        else:
            coef = self.linReg.coef_[0]  # get solution coefficients
        i = 0  # for tracking indices of coefficients
        inflFac = []  # influence factor list
        for selected in self.featureSelector.get_support():  # iterate trough featureSelector selection
            if selected:  # if feature was included in the model add its coefficient to the influence factor list
                inflFac.append(coef[i])
                i += 1  # update i
            else:  # if feature was not included set its influence factor to zero
                inflFac.append(0)
        return inflFac

    def transformFun(self, value):
        return self.scalerY.fit_transform(log(value))

    def inverseTransformFun(self, value):
        return exp(self.scalerY.inverse_transform(value))
