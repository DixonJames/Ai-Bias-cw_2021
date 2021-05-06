import pandas as pd
import re
import os


class FileErr(Exception):
    pass


class setupData:
    @staticmethod
    def openXLS(path):
        try:
            with open(path) as file:
                lines = file.readlines()

            return lines
        except Exception as e:
            raise FileErr() from e

    @staticmethod
    def codeReplacer(lines):
        repalcement_dict = {
            "A11": "0",
            "A12": "200",
            "A13": "200",
            "A14": "0",

            "A30": " no credits taken/ all credits paid back duly",
            "A31": " all credits at this bank paid back duly",
            "A32": " existing credits paid back duly till now",
            "A33": " delay in paying off in the past",
            "A34": " critical account/ other credits existing (not at this bank)",

            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "(vacation - does not exist?)",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",

            "A61": "100",
            "A62": "300",
            "A63": "750",
            "A64": "1000",
            "A65": "0",

            "A71": "unemployed",
            "A72": "1",
            "A73": "2.5",
            "A74": "6",
            "A75": "7",

            "A91": "male : divorced/separated",
            "A92": "female : divorced/separated/married",
            "A93": "male : single",
            "A94": "male : married/widowed",
            "A95": "female : single",

            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",

            "A121": "real estate",
            "A122": "building society savings agreement/ life insurance",
            "A123": "car or other, not in attribute 6",
            "A124": "unknown / no property",

            "A141": "bank",
            "A142": "stores",
            "A143": "none",

            "A151": "rent",
            "A152": "own",
            "A153": "for free",

            "A171": "unemployed/ unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/ self-employed/",

            "A191": "none",
            "A192": "yes, registered under the customers name",

            "A201": "yes",
            "A202": "no"}

        for record in repalcement_dict:
            for line_i in range(len(lines)):
                lines[line_i] = re.sub(record, repalcement_dict[record].replace(" ", "_"), lines[line_i])

        return lines

    @staticmethod
    def convertToDF(lines):
        datalaists = []
        col_names = ["Status of existing checking account", "Duration in month", "Credit history", "Purpose",
                     "Credit amount", "Savings account/bonds", "Present employment since",
                     "Installment rate in percentage of disposable income", "Personal status and sex",
                     "Other debtors / guarantors", "Present residence since", "Property", "Age",
                     "Other installment plans ", "Housing", "Number of existing credits", "Job",
                     "Number of people being liable to provide maintenance", "Telephone", "foreign worker",
                     "credit_decision"]

        for line in lines:
            datalaists.append(line[:-1].split(" "))
        wholedf = pd.DataFrame(datalaists)
        wholedf.columns = col_names

        return wholedf

class DataManipulation:
    @staticmethod
    def scaleData(df, target_col):
        # scales data to between 0 and 1
        from sklearn.preprocessing import normalize, MinMaxScaler

        min_max_scaler = MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(pd.DataFrame(df[target_col]))

        df[target_col] = pd.DataFrame(data_scaled)

    @staticmethod
    def easyOneHot(df, target_col):
        encodedCols = pd.get_dummies(df[target_col])

        for col in list(encodedCols):
            df[target_col + "_" + str(col)] = encodedCols[col]
        df.drop(columns=[target_col], inplace=True)

    @staticmethod
    def featureHash(df, target_col, num_cols):
        from sklearn.feature_extraction import FeatureHasher

        FHasher = FeatureHasher(n_features=num_cols, input_type='string')
        hashed_cols = pd.DataFrame(FHasher.fit_transform(pd.DataFrame([[f"{val}"] for val in list(df[target_col])], columns = ['val'])['val']).toarray())
        hashed_cols.rename(columns={key: f"{target_col}_f_hash_{key}" for key in list(hashed_cols)}, inplace=True)
        # remove old categorical columbs

        num_X_valid = df.drop([target_col], axis=1)

        # stitch it all together
        num_X_valid.reset_index(inplace=True)

        en_valid = pd.concat([num_X_valid, hashed_cols], axis=1)

        return en_valid

class attribDeriving:
    def genderLabel(self, row):
        if row['Personal status and sex'] == "A91":
            return "male"
        if row['Personal status and sex'] == "A92":
            return "female"
        if row['Personal status and sex'] == "A93":
            return "male"
        if row['Personal status and sex'] == "A94":
            return "male"
        if row['Personal status and sex'] == "A95":
            return "female"

    def maritalLabel(self, row):
        if row['Personal status and sex'] == "A91":
            return "divorced", "separated"
        if row['Personal status and sex'] == "A92":
            return "divorced","separated","married"
        if row['Personal status and sex'] == "A93":
            return "single"
        if row['Personal status and sex'] == "A94":
            return "married","widowed"
        if row['Personal status and sex'] == "A95":
            return "single"




def formatData():
    dataPath = os.path.join("data", "german.data")
    WholeDF = setupData.convertToDF((setupData.openXLS(dataPath)))

    WholeDF["sex"] = WholeDF.apply(lambda row: attribDeriving().genderLabel(row), axis=1)
    WholeDF["marital_status"] = WholeDF.apply(lambda row: attribDeriving().maritalLabel(row), axis=1)
    WholeDF.drop(["Personal status and sex", "Purpose","Other debtors / guarantors","Other installment plans ", "Housing", "Telephone"], axis=1, inplace=True)

    #do one hot encoding for
    oneHotColls = ["sex", "marital_status", "Status of existing checking account", "Credit history",
                      "Savings account/bonds", "Present employment since",
                      "Property",
                      "Job",
                      "foreign worker"]

    for atrib in oneHotColls:
        DataManipulation.easyOneHot(WholeDF, atrib)

    normalisingColls = ['Installment rate in percentage of disposable income', 'Number of existing credits',
     'Number of people being liable to provide maintenance',
     'Credit amount', 'Age', 'Present residence since','Duration in month']

    for atrib in normalisingColls:
        DataManipulation.scaleData(WholeDF, atrib)


    return WholeDF

class DataSplit:
    def __init__(self, whole_df, split):
        self.data = whole_df.drop(columns=["credit_decision"])
        self.target = whole_df["credit_decision"]

        self.split = split

    def naieveSplit(self, validation = False):
        rows = self.data.count()[0]
        x_train = self.data[: round(self.split * rows)]
        y_train = self.target[: round((self.split) * rows)]

        x_test = self.data[round((self.split) * rows):]
        y_test = self.target[round((self.split) * rows):]
        if validation:
            test_rows = x_test.count()[0]
            x_validation = x_test[:round(0.5 * test_rows)]
            y_validation = y_test[:round(0.5 * test_rows)]

            x_test = x_test[round(0.5 * test_rows):]
            y_test = y_test[round(0.5 * test_rows):]

            return x_train, y_train, x_test, y_test, x_validation, y_validation

        return x_train, y_train, x_test, y_test

class Models:
    def __init__(self, df):
        self.df = df

    def SVM_train(self, best_p = False):
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC

        # creating data sets
        dataset_creation = DataSplit(whole_df=self.df, split=0.7)
        x_train, y_train, x_test, y_test, x_validation, y_validation = dataset_creation.naieveSplit(True)

        # settting up hyperperams to tune
        parameters = dict()

        parameters["kernel"] = ["linear", "poly", "rbf", "sigmoid"]
        parameters["degree"] = [1,2,3,4,5]
        parameters["gamma"] = ["scale", "auto"]
        parameters["C"] = [float(i)/1000 for i in range(100, 150)]
        #parameters["epsilon"] = [float(i)/1000 for i in range(100, 150)]

        model = SVC()

        SVR_pred = GridSearchCV(model, parameters, n_jobs=-1, return_train_score=True)


        if not (best_p):
            SVR_pred.fit(x_train, y_train)
            return SVR_pred
        else:
            SVR_pred.fit(x_train, y_train)
            return SVR_pred.best_score_, SVR_pred.best_estimator_, SVR_pred.best_params_

if __name__ == '__main__':
    df = formatData()

    modeltraining = Models(df)
    bestSVCModel = modeltraining.SVM_train()
