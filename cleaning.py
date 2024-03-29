import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import functools
import math
import re
import os


class FileErr(Exception):
    pass


def getData():
    data_path = os.path.join("data", "german.data")
    whole_df = SetupData.convertToDF((SetupData.openXLS(data_path)))

    whole_df["sex"] = whole_df.apply(lambda row: AttribDeriving().genderLabel(row), axis=1)
    whole_df["marital_status"] = whole_df.apply(lambda row: AttribDeriving().maritalLabel(row), axis=1)
    return whole_df


def predictTarget(model, feature_vector):
    return model.predict(feature_vector)


class SetupData:
    def __init__(self):
        self.repalcement_dict = {
            "male": "male",
            "female": "female",

            "A11": "<0",
            "A12": "0-200",
            "A13": "200+",
            "A14": np.NAN,

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
            "A202": "no",

            ("divorced", "separated"): ("divorced", "separated"),
            ("divorced", "separated", "married"): ("divorced", "separated", "married"),

            ("married", "widowed"): ("married", "widowed"),
            "single": "single"}

    @staticmethod
    def openXLS(path):
        try:
            with open(path) as file:
                lines = file.readlines()

            return lines
        except Exception as e:
            raise FileErr() from e

    def codeReplacer(self, lines):

        for record in self.repalcement_dict:
            for line_i in range(len(lines)):
                lines[line_i] = re.sub(record, self.repalcement_dict[record].replace(" ", "_"), lines[line_i])

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

        df[target_col] = pd.DataFrame(data_scaled).astype("float64")

    @staticmethod
    def easyOneHot(df, target_col):
        encodedCols = pd.get_dummies(df[target_col])

        for col in list(encodedCols):
            df[target_col + "_" + str(col)] = encodedCols[col].astype("int32")
        df.drop(columns=[target_col], inplace=True)

    @staticmethod
    def featureHash(df, target_col, num_cols):
        from sklearn.feature_extraction import FeatureHasher

        FHasher = FeatureHasher(n_features=num_cols, input_type='string')
        hashed_cols = pd.DataFrame(FHasher.fit_transform(
            pd.DataFrame([[f"{val}"] for val in list(df[target_col])], columns=['val'])['val']).toarray())
        hashed_cols.rename(columns={key: f"{target_col}_f_hash_{key}" for key in list(hashed_cols)}, inplace=True)
        # remove old categorical columbs

        num_X_valid = df.drop([target_col], axis=1)

        # stitch it all together
        num_X_valid.reset_index(inplace=True)

        en_valid = pd.concat([num_X_valid, hashed_cols], axis=1)

        return en_valid

    @staticmethod
    def categorise(df, target_col):
        trans_dict = dict()
        unique_values = df[target_col].unique()
        i = 1
        for value in unique_values:
            trans_dict[value] = i
            i += 1
        df[target_col] = df[target_col].replace(trans_dict)


class AttribDeriving:
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
            return "divorced", "separated", "married"
        if row['Personal status and sex'] == "A93":
            return "single"
        if row['Personal status and sex'] == "A94":
            return "married", "widowed"
        if row['Personal status and sex'] == "A95":
            return "single"


def formatData():
    whole_df = getData()

    whole_df["sex"] = whole_df.apply(lambda row: AttribDeriving().genderLabel(row), axis=1)
    whole_df["marital_status"] = whole_df.apply(lambda row: AttribDeriving().maritalLabel(row), axis=1)
    whole_df.drop(
        ["Personal status and sex", "Purpose", "Other debtors / guarantors", "Other installment plans ", "Housing",
         "Telephone"], axis=1, inplace=True)

    # do one hot encoding for
    one_hot_cols = ["marital_status", "Status of existing checking account", "Credit history",
                    "Savings account/bonds", "Present employment since",
                    "Property",
                    "Job"]

    for attrib in one_hot_cols:
        DataManipulation.easyOneHot(whole_df, attrib)

    normalising_cols = ['Installment rate in percentage of disposable income', 'Number of existing credits',
                        'Number of people being liable to provide maintenance',
                        'Credit amount', 'Age', 'Present residence since', 'Duration in month']

    for attrib in normalising_cols:
        DataManipulation.scaleData(whole_df, attrib)

    categorising_cols = ['sex', 'foreign worker']

    for attrib in categorising_cols:
        DataManipulation.categorise(whole_df, attrib)

    cont_to_dis_cols = ['Age']
    for attrib in cont_to_dis_cols:
        whole_df['Age_cat'] = (pd.cut(whole_df[attrib], bins=[0, 1 / 3, 2], labels=[1, 2], right=False)).astype(
            'float64')

    whole_df['credit_decision'] = whole_df['credit_decision'].astype('float64')

    return whole_df


class AnalyseData:
    def __init__(self, df):
        self.df = df

        self.numeric_f = ['Installment rate in percentage of disposable income', 'Number of existing credits',
                          'Number of people being liable to provide maintenance', 'Credit amount', 'Age',
                          'Present residence since', 'Duration in month']
        self.categorical_f = ["sex", "marital_status", "Status of existing checking account", "Credit history",
                              "Savings account/bonds", "Present employment since", "Property", "Job", "foreign worker"]

    def categoricalReport(self, col_name):
        value_counts = [(SetupData().repalcement_dict[c], v) for v, c in
                        list(zip(self.df[col_name].value_counts(), self.df[col_name].value_counts().index))]
        report_dict = {'categories': [SetupData().repalcement_dict[c] for c in self.df[col_name].unique()],
                       'mode': SetupData().repalcement_dict[self.df[col_name].describe()['top']]}

        category_indexes, category_values = [c for (c, v) in value_counts], [v for (c, v) in value_counts]

        j = 0
        for i in range(min(len(category_indexes), 3)):
            report_dict[f"{i + 1}-most-frequent"] = f"{category_indexes[i]}, {category_values[i]}"
            j += 1

        for i in range(j, 3):
            report_dict[f"{i + 1}-most-frequent"] = np.NAN

        return report_dict

    def numericReport(self, col_name):
        report = dict(pd.Series(self.df[col_name]).astype('float64').describe())
        report['std'] = float('%.5g' % report['std'])
        del report["25%"]
        del report["50%"]
        del report["75%"]
        report['mode'] = self.df[col_name].mode()[0]
        return report

    def runReports(self):
        categorical = {}
        numeric = {}

        for category in self.numeric_f:
            numeric[category] = self.numericReport(category)

        for category in self.categorical_f:
            categorical[category] = self.categoricalReport(category)

        return numeric, categorical

    def latexGraphRowFormat(self, values):
        strings = [str(value) for value in values]
        wholeLine = ""
        for word in strings:
            try:
                wholeLine += (str(float('%.5g' % float(word))) + " &")
            except:
                wholeLine += (word + "&")

        return f"{wholeLine}" + "  \\" + "\\"

    def printReportGraphStyle(self, report):
        report = pd.DataFrame(report)
        cols = report.columns
        print("&" + self.latexGraphRowFormat(cols))
        for row in list(report.index):
            print(f"{row} &" + self.latexGraphRowFormat(report.loc[[row]].values[0]))

    def dataframeSubSet(self):
        self.df[["sex", "Age"]].groupby("sex").mean() * 75
        self.df[["credit_decision", "Age"]].groupby("credit_decision").mean() * 75, self.df
        self.df[["sex", "credit_decision"]].groupby("credit_decision").mode()

        # age of loan
        (self.df[self.df["credit_decision"] == "1"]["Age"] * 75).hist()

    def prediction_Comparison(self):

        # how predicted values changed with age
        ageComparison = pd.DataFrame()
        self.df['Age_cat'] = self.df['Age_cat'].astype("string")

        numYoung = len(self.df.loc[self.df['Age_cat'] == '1.0'])
        numOld = len(self.df.loc[self.df['Age_cat'] == '2.0'])

        trueOldLoans = len(self.df.loc[self.df['Age_cat'] == '2.0'].loc[self.df["true-val"] == 1.0])
        trueyoungLoans = len(self.df.loc[self.df['Age_cat'] == '1.0'].loc[self.df["true-val"] == 1.0])

        SvmOldLoans = len(self.df.loc[self.df['Age_cat'] == '2.0'].loc[self.df["SVM-predict-val"] == 1.0])
        SvmYoungLoans = len(self.df.loc[self.df['Age_cat'] == '1.0'].loc[self.df["SVM-predict-val"] == 1.0])

        RfOldLoans = len(self.df.loc[self.df['Age_cat'] == '2.0'].loc[self.df["RF-predict-val"] == 1.0])
        RfYoungLoans = len(self.df.loc[self.df['Age_cat'] == '1.0'].loc[self.df["RF-predict-val"] == 1.0])

        ageComparison["number of loans given in Dataset"] = pd.Series(
            [100 * trueOldLoans / numOld, 100 * trueyoungLoans / numYoung])
        ageComparison["SVM predicted loans given"] = pd.Series(
            [100 * SvmOldLoans / numOld, 100 * SvmYoungLoans / numYoung])
        ageComparison["RF predicted loans given"] = pd.Series(
            [100 * RfOldLoans / numOld, 100 * RfYoungLoans / numYoung])
        ageComparison.T.columns = ['Old', 'Young']

        ageComparison.rename(index={0: 'Old', 1: 'Young'}, inplace=True)

        self.printReportGraphStyle(ageComparison)






        # how predicted values changed with sex

        sexComparison = pd.DataFrame()
        self.df['sex_cat'] = self.df['sex'].astype("string")

        numMales = len(self.df.loc[self.df['sex_cat'] == '1'])
        numFemales = len(self.df.loc[self.df['sex_cat'] == '2'])

        trueMaleLoans = len(self.df.loc[self.df['sex_cat'] == '1'].loc[self.df["true-val"] == 1.0])
        trueFemaleLoans = len(self.df.loc[self.df['sex_cat'] == '2'].loc[self.df["true-val"] == 1.0])

        SvmMaleLoans = len(self.df.loc[self.df['sex_cat'] == '1'].loc[self.df["SVM-predict-val"] == 1.0])
        SvmFemaleLoans = len(self.df.loc[self.df['sex_cat'] == '2'].loc[self.df["SVM-predict-val"] == 1.0])

        RfMaleLoans = len(self.df.loc[self.df['sex_cat'] == '1'].loc[self.df["RF-predict-val"] == 1.0])
        RfFemaleLoans = len(self.df.loc[self.df['sex_cat'] == '2'].loc[self.df["RF-predict-val"] == 1.0])

        sexComparison["number of loans given in Dataset"] = pd.Series(
            [100 * trueMaleLoans / numMales, 100 * trueFemaleLoans / numFemales])
        sexComparison["SVM predicted loans given"] = pd.Series(
            [100 * SvmMaleLoans / numMales, 100 * SvmFemaleLoans / numFemales])
        sexComparison["RF predicted loans given"] = pd.Series(
            [100 * RfMaleLoans / numMales, 100 * RfFemaleLoans / numFemales])
        sexComparison.T.columns = ['Male', 'Female']

        sexComparison.rename(index={0: 'Male', 1: 'Female'}, inplace=True)

        self.printReportGraphStyle(sexComparison)

        # hostogram of prediced values
        """
        sns.set(style="darkgrid")
        figure, axis = plt.subplots(2, 3, figsize=(7, 7))

        sns.histplot(data=self.df[self.df["true-val"] == "1"]["Age"] * 75, kde=True, color="red",
                     ax=axis[0, 0]).set_title("true value grant")
        sns.histplot(data=self.df[self.df["SVM-predict-val"] == "1"]["Age"] * 75, kde=True, color="green",
                     ax=axis[0, 1]).set_title("SVM prediction grant")
        sns.histplot(data=self.df[self.df["RF-predict-val"] == "1"]["Age"] * 75, kde=True, color="blue",
                     ax=axis[0, 2]).set_title("RF prediction grant")

        sns.histplot(data=self.df[self.df["true-val"] == "2"]["Age"] * 75, kde=True, color="red",
                     ax=axis[1, 0]).set_title("true value rejection")
        sns.histplot(data=self.df[self.df["SVM-predict-val"] == "2"]["Age"] * 75, kde=True, color="green",
                     ax=axis[1, 1]).set_title("SVM prediction rejection")
        sns.histplot(data=self.df[self.df["RF-predict-val"] == "2"]["Age"] * 75, kde=True, color="blue",
                     ax=axis[1, 2]).set_title("RF prediction rejection")

        # plt.show()
    
        """

class FeatureAnalysis:
    def __init__(self, df, label):
        self.whole_df = df
        self.df = df

        self.label = label

        self.correlation_map = self.feature_correlecation(show=False)

        self.removed = []

    def feature_correlecation(self, show=False):
        df = self.df

        f_correlation = self.df.corr(method='pearson')
        f_correlation = np.abs(f_correlation)

        if show:
            sns.heatmap(f_correlation, xticklabels=2, yticklabels=False)
            plt.show()

        return f_correlation


class ModelFeatureAnalysis:
    def __init__(self, whole_df, target):
        self.whole_df = whole_df.drop(columns=[target], axis=1)

    def svm_featureImportance(self, model, name):
        self.whole_df = self.whole_df.drop(columns=['credit decision'])
        feature_imp = pd.DataFrame(np.std(self.whole_df), columns=[F'{name}_std'])
        feature_coef = model.coef_.T
        feature_importance = np.abs((list(np.std(self.whole_df, 0).T) * model.coef_).T)

        feature_imp[f"{name}_feature_coef"] = feature_coef
        feature_imp[f"{name}_feature_importance"] = feature_importance

        # feature_imp.sort_values(by=[f"{name}_feature_importance"], inplace=True)

        return pd.DataFrame(feature_imp[f"{name}_feature_importance"])

    def tree_featureImportance(self, model, name):
        feature_imp = pd.DataFrame(
            pd.Series(model.feature_importances_, [i for i in list(self.whole_df.columns) if i != 'credit decision']),
            columns=[f"{name}_feature_importance"])
        # feature_imp.sort_values(by=[f"{name}_feature_importance"], inplace=True)

        return feature_imp


class DataSplit:
    def __init__(self, whole_df, split):
        self.wholedf = whole_df
        self.data = whole_df.drop(columns=["credit_decision"])
        self.target = whole_df["credit_decision"]
        self.targetLabel = "credit_decision"

        self.split = split

    def naieveSplit(self, validation=False, input_df=None, train=True):
        try:
            if input_df.all != None:
                data = input_df
        except:
            data = self.wholedf

        rows = data.count()[0]
        x_train = data[: round(self.split * rows)].drop(columns=['credit_decision'])
        y_train = self.target[: round(self.split * rows)]

        x_test = data[round(self.split * rows):].drop(columns=['credit_decision'])
        y_test = self.target[round(self.split * rows):]
        if validation:
            test_rows = x_test.count()[0]
            x_validation = x_test[:round(0.5 * test_rows)]
            y_validation = y_test[:round(0.5 * test_rows)]

            x_test = x_test[round(0.5 * test_rows):]
            y_test = y_test[round(0.5 * test_rows):]

            return x_train, y_train, x_test, y_test, x_validation, y_validation

        if train == True:
            return x_train, y_train, x_test, y_test

    def demographicParitySplit(self, feature):
        """
        same amount of sucsess for each category
        """
        categories = list(self.wholedf[feature].unique())
        target_categories = list(self.target.unique())
        minLength = len(self.wholedf)
        new_equilised_df = (self.wholedf.loc[self.wholedf[feature] == -999])
        category_len_dict = {}
        for category in categories:
            cat_minLength = len(self.wholedf)
            for target_cat in target_categories:
                if cat_minLength > len(self.wholedf.loc[self.wholedf[feature] == category].loc[
                                       self.wholedf[self.targetLabel] == target_cat]):
                    cat_minLength = len(self.wholedf.loc[self.wholedf[feature] == category].loc[
                                        self.wholedf[self.targetLabel] == target_cat])
            category_len_dict[category] = cat_minLength


        frames = []
        for category in categories:
            for target_cat in target_categories:
                wholeSample = self.wholedf.loc[self.wholedf[feature] == category].loc[
                    self.wholedf[self.targetLabel] == target_cat]
                frames.append((wholeSample).sample(frac=category_len_dict[category] / len(wholeSample), random_state=2))

        new_equilised_df = pd.concat(frames).sample(frac=1).reset_index()

        return new_equilised_df.drop(columns=['index'], axis=1)

    def equilisedOddsSplit(self, feature):
        """
        same amount of each category
        :param feature: feature in witch categories inside it must be balanced
        :return:
        """

        categories = list(self.wholedf[feature].unique())
        minLength = len(self.wholedf)
        for category in categories:
            if minLength > len(self.wholedf.loc[self.wholedf[feature] == category]):
                minLength = len(self.wholedf.loc[self.wholedf[feature] == category])

        new_equilised_df = (self.wholedf.loc[self.wholedf[feature] == -999])
        frames = []
        for category in categories:
            wholesize = len(self.wholedf.loc[self.wholedf[feature] == category])
            frames.append((self.wholedf.loc[self.wholedf[feature] == category]).sample(frac=minLength / wholesize,
                                                                                       random_state=2))

        new_equilised_df = pd.concat(frames).sample(frac=1).reset_index()

        return new_equilised_df.drop(columns=['index'], axis=1)

    def equilisedOpertunity(self, feature):
        pass


class Models:

    def __init__(self, df, trainAll=False):
        self.df = df

        # creating data sets
        dataset_creation = DataSplit(whole_df=self.df, split=0.7)

        self.x_train, self.y_train, self.x_test, self.y_test = dataset_creation.naieveSplit(input_df=self.df)
        if trainAll:
            self.x_train = self.df.drop(columns=["credit_decision"])
            self.y_train = self.df["credit_decision"]

        f_scale = StandardScaler()

        self.svmModel = self.SVM_train()
        self.rfModel = self.RF_train()

    def SVM_train(self, best_p=False):
        # setting up hyperperams to tune
        parameters = dict()

        parameters["kernel"] = ["linear"]  # ["linear", "poly", "rbf", "sigmoid"]
        parameters["degree"] = [6]  # [1, 2, 3, 4, 5,6,7,8,9]
        parameters["gamma"] = ["scale"]  # ["scale", "auto"]
        parameters["C"] = [0.146]  # [float(i) / 1000 for i in range(100, 150)]
        # parameters["epsilon"] = [float(i)/1000 for i in range(100, 150)]

        model = SVC(probability=True)

        SVR_pred = GridSearchCV(model, parameters, n_jobs=-1, return_train_score=True, cv=5)

        if not best_p:
            SVR_pred.fit(self.x_train, self.y_train)
            return SVR_pred
        else:
            SVR_pred.fit(self.x_train, self.y_train)
            return SVR_pred.best_score_, SVR_pred.best_estimator_, SVR_pred.best_params_

    def RF_train(self, best_p=False):
        # settting up hyperperams to tune
        parameters = dict()

        parameters["n_estimators"] = [74]  # [i for i in range(30, 100)]
        parameters["criterion"] = ["entropy"]  # ["gini", "entropy"]

        # parameters["min_samples_split"] = [i for i in range(1, 5)]
        # parameters["min_samples_leaf"] = [i for i in range(1, 5)]

        model = RandomForestClassifier(n_jobs=-1)

        RF_pred = GridSearchCV(model, parameters, return_train_score=True, n_jobs=-1, cv=5)

        if not (best_p):
            RF_pred.fit(self.x_train, self.y_train)
            return RF_pred
        else:
            RF_pred.fit(self.x_train, self.y_train)
            return RF_pred.best_score_, RF_pred.best_estimator_, RF_pred.best_params_

    def predictionOutput(self):
        svmPrediction = predictTarget(self.svmModel, self.x_test)
        rfPrediction = predictTarget(self.rfModel, self.x_test)

        output_df = self.x_test.copy()
        output_df["true-val"] = self.y_test
        output_df["SVM-predict-val"] = svmPrediction
        output_df["RF-predict-val"] = rfPrediction

        return output_df


class Repair:

    def median(self, series):
        values = list(series)
        values.sort()

        vals_len = len(values)
        if vals_len % 2 == 0:
            return values[int((vals_len / 2) - 1)]
        return values[int(vals_len / 2)]

    def KCDM(self, df, sensative_col, sensative_group, benefited_group, target_col, good_outcome):
        """
        Kamiran–Calders Discrimination Measure
        :return:
        """
        num_marginalised = len(df.loc[df[sensative_col] == sensative_group])
        num_benefited = len(df.loc[df[sensative_col] == benefited_group])

        num_good_outcome_in_marginalised = len(
            df.loc[df[target_col] == good_outcome].loc[df[sensative_col] == sensative_group])
        num_good_outcome_in_benefited = len(
            df.loc[df[target_col] == good_outcome].loc[df[sensative_col] == benefited_group])

        prob_good_given_marginalised = num_good_outcome_in_marginalised / num_marginalised
        prob_good_given_benefited = num_good_outcome_in_benefited / num_benefited

        KCDM_score = prob_good_given_benefited - prob_good_given_marginalised

        return KCDM_score

    def flip_decision(self, value, values):
        if value == values[0]:
            return values[1]
        return values[0]

    def CND(self, wholeDF, target_col, protected_cols, benefitedGroups, marginalisedGroups, max_KCDM_score):
        """
        classification with no discrimination
        [4] Faisal Kamiran and Toon Calders. Classifying Without Discriminating. In Classifying Without Discriminating. Computer, Control and Communication, 2010.
        :return:
        """
        # creates the DF with probabilities for calssification
        wholeDF_with_prob = wholeDF.copy()
        repaired_df = wholeDF.copy()
        model_training = Models(wholeDF)

        svmModel = model_training.svmModel.best_estimator_
        bias_preprocessing = DataSplit(formatData(), 0.7)
        x_train, y_train, x_test, y_test = bias_preprocessing.naieveSplit()
        x_all = x_train.append(x_test)

        target_probs = pd.DataFrame(svmModel.predict_proba(wholeDF.drop(columns=["credit_decision"])),
                                    columns=["prob_credit_1", "prob_credit_2"])
        wholeDF_with_prob["prob_credit_1"], wholeDF_with_prob["prob_credit_2"] = target_probs["prob_credit_1"], \
                                                                                 target_probs["prob_credit_2"]

        target_grant = 1.0
        target_rejection = 2.0
        for col_i in range(len(protected_cols)):
            col = protected_cols[col_i]
            benefited_group = benefitedGroups[col_i]
            marginalised_group = marginalisedGroups[col_i]

            cp_df = (wholeDF_with_prob.loc[wholeDF_with_prob[col] == marginalised_group].loc[
                wholeDF_with_prob[target_col] == target_rejection]).sort_values(by="prob_credit_2",
                                                                                ascending=True).reset_index()
            cd_df = (wholeDF_with_prob.loc[wholeDF_with_prob[col] == benefited_group].loc[
                wholeDF_with_prob[target_col] == target_grant]).sort_values(by="prob_credit_1",
                                                                            ascending=False).reset_index()

            num_marginalised = len(wholeDF_with_prob.loc[wholeDF_with_prob[col] == marginalised_group])
            num_benefited = len(wholeDF_with_prob.loc[wholeDF_with_prob[col] == benefited_group])

            num_marginalised_pos = len(wholeDF_with_prob.loc[wholeDF_with_prob[col] == marginalised_group].loc[
                                           wholeDF_with_prob[target_col] == target_grant])
            num_benefited_pos = len(wholeDF_with_prob.loc[wholeDF_with_prob[col] == benefited_group].loc[
                                        wholeDF_with_prob[target_col] == target_grant])

            num_swaps = round(((num_marginalised * num_benefited_pos) - (num_benefited * num_marginalised_pos)) / (
                    num_benefited + num_marginalised))

            KCDM_score = self.KCDM(repaired_df, col, marginalised_group, benefited_group, target_col, target_grant)
            swap_i = 0
            while KCDM_score - max_KCDM_score > 0:
                flip_indexes = [int(cp_df.iloc[[swap_i]]["index"].values[0]),
                                int(cd_df.iloc[[swap_i]]["index"].values[0])]

                for index in flip_indexes:
                    targets = list(repaired_df[target_col].values)
                    targets[index] = self.flip_decision(targets[index], list(set(repaired_df[target_col].values)))
                    repaired_df[target_col] = targets

                KCDM_score = self.KCDM(repaired_df, col, marginalised_group, benefited_group, target_col, target_grant)
                swap_i += 1

                # print(KCDM_score)

        return repaired_df
        # goes though each of the gategories and applies CND

    def repair(self, wholeDF, target_cols, protected_cols, lambda_const):
        wholeDF = wholeDF.sample(len(wholeDF)).reset_index().drop(columns=['index'], axis=1)

        # get unique vlaues for each col
        col_vals = dict()
        allcols = target_cols.copy()
        allcols.extend(protected_cols)
        for col in allcols:
            col_vals[col] = list(wholeDF[col].unique())

        # add stratified columns to dataframe
        stratified_cols = []
        for col in protected_cols:
            if len(stratified_cols) == 0:
                stratified_cols = list(wholeDF[col].unique())
            else:
                new_strat = []
                for col_old in stratified_cols:
                    for sub_col in list(wholeDF[col].unique()):
                        new_strat.append([col_old, sub_col])
                stratified_cols = new_strat

        stratified_series = []
        smallest_g = len(wholeDF["Age"])
        for col in stratified_cols:
            count = 0
            conditions = []
            for result, category in zip(col, protected_cols):
                conditions.append(wholeDF[category] == result)

            indexes = set(wholeDF.index)
            for condition in conditions:
                indexes = indexes.intersection(set(wholeDF.index[condition]))

            stratified_series = []
            for i in range(len(wholeDF["Age"])):
                if i in indexes:
                    count += 1
                    stratified_series.append(1)
                else:
                    stratified_series.append(0)

            if count < smallest_g:
                smallest_g = count
            wholeDF[(col[0], col[1])] = pd.Series(stratified_series)

        # find sisze of each stratified group campered to smallest
        coll_offset = {(col[0], col[1]): math.floor(
            len(wholeDF[(col[0], col[1])].loc[wholeDF[(col[0], col[1])] == 1]) / smallest_g) for col in stratified_cols}

        # now find target for each quantile
        ratio_group_per_quantile = 1 / smallest_g

        quantile_targets = []
        quantile_indexes = []
        for q_n in range(smallest_g):
            quant_group_medians = []
            quant_group_indexes = []
            for col in stratified_cols:
                pos_stratified_col_df = pd.DataFrame(
                    wholeDF[(col[0], col[1])].loc[wholeDF[(col[0], col[1])] == 1]).reset_index()
                pos_stratified_col_df.columns = ["original_index", (col[0], col[1])]

                start_index = q_n * coll_offset[(col[0], col[1])]
                end_index = min(start_index + coll_offset[(col[0], col[1])], len(pos_stratified_col_df))

                target_values = wholeDF.loc[(wholeDF[(col[0], col[1])] == 1)][start_index:end_index][target_label]

                colMedian = self.median(list(target_values.values))

                quant_group_medians.append(colMedian)
                quant_group_indexes.extend(list(pos_stratified_col_df["original_index"].values[start_index:end_index]))

            quantile_indexes.append(quant_group_indexes)
            quantile_targets.append(self.median(quant_group_medians))

        repairedDF = wholeDF.copy()
        all_scores = list(wholeDF["credit_decision"].values.copy())
        ordered_unique_targets = list(wholeDF["credit_decision"].unique())
        ordered_unique_targets.sort()

        for q_n in range(smallest_g):

            quantile_target = quantile_targets[q_n]
            quantile_indexList = quantile_indexes[q_n]
            quantile_df = (wholeDF.iloc[[i for i in quantile_indexList]]).sort_values(
                by=['credit_decision']).reset_index()

            for id in range(len(quantile_indexList)):
                original_value = quantile_df[target_label].values[id]
                origainl_index = quantile_df["index"].values[id]

                originals_index_uniqueTargets = ordered_unique_targets.index(original_value)
                targets_index_uniqueTargets = ordered_unique_targets.index(quantile_target)
                distance = targets_index_uniqueTargets - originals_index_uniqueTargets

                direction = quantile_target - original_value
                # first_group_start
                # second_groupStart

                repair_distance = round((round(len(quantile_df) / 2) - id) * lambda_const)

                index_repair_value = max(0, min(id + repair_distance, len(quantile_df) - 1))
                # combinatorial
                # repair_value = quantile_df[target_label].values[index_repair_value]

                repair_value = float(quantile_df[target_label].values[index_repair_value])

                if repair_value != original_value:
                    all_scores[origainl_index] = repair_value

        repairedDF["credit_decision"] = all_scores

        repairedDF = repairedDF.drop(columns=stratified_cols, axis=1)

        return repairedDF


def fairSexDF():
    """
    trains the conventional ML process with a DF with equalised SEX records
    :return:
    """
    target_label = 'credit_decision'
    df = formatData()

    # unbiaes data record reweighting
    bias_preprocessing = DataSplit(df, 0.7)

    eqilised_outcome_df = bias_preprocessing.demographicParitySplit("sex")
    equilised_sex_df = bias_preprocessing.equilisedOddsSplit("sex")

    return eqilised_outcome_df, equilised_sex_df


def fairAgeDF():
    """
    trains the conventional ML process with a DF with equalised AGE records
    :return:
    """
    target_label = 'credit_decision'
    df = formatData()

    # unbiaes data record reweighting
    bias_preprocessing = DataSplit(df, 0.7)

    eqilised_outcome_age_df = bias_preprocessing.demographicParitySplit("Age_cat")
    equilised_age_df = bias_preprocessing.equilisedOddsSplit("Age_cat")

    return eqilised_outcome_age_df, equilised_age_df


def modelDataGeneralise(dataframe):
    model_training = Models(dataframe)

    svmModel = model_training.svmModel.best_estimator_
    RFmodel = model_training.rfModel.best_estimator_

    # feature correlation
    featureworkspace = FeatureAnalysis(df, df[target_label])
    correlation = (featureworkspace.feature_correlecation(show=False))['credit decision']

    # feature importance per model
    feaureimportance = ModelFeatureAnalysis(df, target_label)
    rf_f_improtance = feaureimportance.tree_featureImportance(RFmodel, 'Random Forest')
    svm_f_importance = feaureimportance.svm_featureImportance(svmModel, 'support vector')

    total_feature_importance = pd.DataFrame()
    total_feature_importance['random forest importance'] = pd.Series(
        rf_f_improtance['Random Forest_feature_importance'])
    total_feature_importance['random forest rank'] = pd.Series(
        total_feature_importance['random forest importance'].rank(method='max', ascending=False))
    total_feature_importance['support vector importance'] = pd.Series(
        svm_f_importance['support vector_feature_importance'])
    total_feature_importance['support vector rank'] = pd.Series(
        total_feature_importance['support vector importance'].rank(method='max', ascending=False))
    total_feature_importance['target correlation'] = correlation.drop(columns=['credit decision'])

    # only reports on the most relevant collumbs
    least_relevelt_cols = list((((total_feature_importance['random forest rank'] + total_feature_importance[
        'support vector rank']) / 2).sort_values()).index)[20:]
    total_feature_importance = total_feature_importance.T.drop(columns=[str(i) for i in least_relevelt_cols], axis=1).T

    AnalyseData(total_feature_importance).printReportGraphStyle(total_feature_importance)


def simpleGroupEquilisation(df):
    eqilised_outcome_sex_df, equilised_sex_df = fairSexDF()
    eqilised_outcome_age_df, equilised_age_df = fairAgeDF()

    equilisedModels = []

    print("========age============")
    for dataframe in [df, eqilised_outcome_age_df, equilised_age_df]:
        # unbiaes data record reweighting
        bias_preprocessing = DataSplit(dataframe, 0.7)

        # training data
        model_training = Models(dataframe)
        predctionDF = model_training.predictionOutput()

        svmModel = model_training.svmModel.best_estimator_
        RFmodel = model_training.rfModel.best_estimator_

        equilisedModels.append((svmModel, RFmodel))

        #outputAnalysis = AnalyseData(predctionDF)
        #outputAnalysis.prediction_Comparison()

    print("========sex============")
    for dataframe in [df, eqilised_outcome_sex_df, equilised_sex_df]:
        # unbiaes data record reweighting
        bias_preprocessing = DataSplit(dataframe, 0.7)

        # training data
        model_training = Models(dataframe)
        predctionDF = model_training.predictionOutput()

        svmModel = model_training.svmModel.best_estimator_
        RFmodel = model_training.rfModel.best_estimator_

        equilisedModels.append((svmModel, RFmodel))

        outputAnalysis = AnalyseData(predctionDF)
        outputAnalysis.prediction_Comparison()



    bias_preprocessing = DataSplit(formatData(), 0.7)
    x_train, y_train, x_test, y_test = bias_preprocessing.naieveSplit()
    model_acc = {"naive": [accuracy_score(y_test, equilisedModels[0][0].predict(x_test)),
                           accuracy_score(y_test, equilisedModels[0][1].predict(x_test))],
                 "eqilised outcome sex": [accuracy_score(y_test, equilisedModels[1][0].predict(x_test)),
                                          accuracy_score(y_test, equilisedModels[1][1].predict(x_test))],
                 "equilised sex": [accuracy_score(y_test, equilisedModels[2][0].predict(x_test)),
                                   accuracy_score(y_test, equilisedModels[2][1].predict(x_test))],
                 "eqilised outcome age": [accuracy_score(y_test, equilisedModels[3][0].predict(x_test)),
                                          accuracy_score(y_test, equilisedModels[3][1].predict(x_test))],
                 "equilised age": [accuracy_score(y_test, equilisedModels[4][0].predict(x_test)),
                                   accuracy_score(y_test, equilisedModels[4][1].predict(x_test))]}


def inicailDSAnalysis():
    # column reports for creating latex graphs
    dataAnalysis = AnalyseData(getData())
    dataAnalysis.dataframeSubSet()

    reports = dataAnalysis.runReports()
    for r in reports:
        dataAnalysis.printReportGraphStyle(r)

    # analyses bias in dataset
    df = formatData()
    dataAnalysis = AnalyseData(df)
    dataAnalysis.dataframeSubSet()


def CNDExperiment():
    target_label = 'credit_decision'
    wholedf = formatData()

    x_train, y_train, x_test, y_test = DataSplit(wholedf, 0.7).naieveSplit()
    x_train["credit_decision"] = y_train

    augmenting_df = x_train
    testing_df = x_test

    cnd_test_values = [i / 1000 for i in range(1, 100)]
    cnd_test_results = {}
    for cnd_val in cnd_test_values:
        results = {}
        CND_df = Repair().CND(augmenting_df, 'credit_decision', ["Age_cat", "sex"], [2.0, 1], [1.0, 2], cnd_val)

        model_training = Models(CND_df, trainAll=True)

        svmModel = SVC(probability=True, kernel="linear", degree=6, gamma="scale", C=0.146).fit(
            CND_df.drop(columns=[target_label]), y=CND_df[target_label])
        RFmodel = RandomForestClassifier(n_jobs=-1, n_estimators=74, criterion="entropy").fit(
            CND_df.drop(columns=[target_label]), y=CND_df[target_label])

        svmPredictions = svmModel.predict(testing_df)
        rfPredictions = RFmodel.predict(testing_df)

        # now compare the results

        svmComparisonDF = pd.DataFrame()
        svmComparisonDF['y_Actual'] = y_test
        svmComparisonDF['y_Predicted'] = svmPredictions

        rfComparisonDF = pd.DataFrame()
        rfComparisonDF['y_Actual'] = y_test
        rfComparisonDF['y_Predicted'] = rfPredictions

        svmConfMat = pd.crosstab(svmComparisonDF['y_Actual'], svmComparisonDF['y_Predicted'], rownames=['Actual'],
                                 colnames=['Predicted'])
        rfConfMat = pd.crosstab(rfComparisonDF['y_Actual'], rfComparisonDF['y_Predicted'], rownames=['Actual'],
                                colnames=['Predicted'])

        svm_TP = svmConfMat[1.0].values[0]
        svm_TN = svmConfMat[2.0].values[1]
        svm_FP = svmConfMat[1.0].values[1]
        svm_FN = svmConfMat[2.0].values[0]
        svm_utility_score = 1 - ((svm_FN / (svm_FN + svm_TP)) + (svm_FP / (svm_FP + svm_TN))) / 2
        svm_acc_score = (svm_TP + svm_TN) / (svm_FN + svm_TP + svm_FP + svm_TN)

        rf_TP = rfConfMat[1.0].values[0]
        rf_TN = rfConfMat[2.0].values[1]
        rf_FP = rfConfMat[1.0].values[1]
        rf_FN = rfConfMat[2.0].values[0]
        rf_utility_score = 1 - ((rf_FN / (rf_FN + rf_TP)) + (rf_FP / (rf_FP + rf_TN))) / 2
        rf_acc_score = (rf_TP + rf_TN) / (rf_FN + rf_TP + rf_FP + rf_TN)


        results["max_KCDM_df_score"] = cnd_val
        results["svm_acc"] = svm_acc_score
        results["svm_util"] = svm_utility_score

        results["rf_acc"] = rf_acc_score
        results["rf_util"] = rf_utility_score

        # results["svm_confMat"] = svmConfMat
        # results["rf_confMat"] = rfConfMat

        cnd_test_results[f"{cnd_val}"] = results
    return cnd_test_results


def repairExperiment():
    target_label = 'credit_decision'
    wholedf = formatData()

    x_train, y_train, x_test, y_test = DataSplit(wholedf, 0.7).naieveSplit()
    x_train["credit_decision"] = y_train

    augmenting_df = x_train
    testing_df = x_test

    lambda_vlaues = [i / 10 for i in range(1, 100)]
    repair_test_results = {}

    for lam_val in lambda_vlaues:
        results = {}
        CND_df = Repair().repair(augmenting_df, ['credit_decision'], ["Age_cat", "sex"], lam_val)

        svmModel = SVC(probability=True, kernel="linear", degree=6, gamma="scale", C=0.146).fit(
            CND_df.drop(columns=[target_label]), y=CND_df[target_label])
        RFmodel = RandomForestClassifier(n_jobs=-1, n_estimators=74, criterion="entropy").fit(
            CND_df.drop(columns=[target_label]), y=CND_df[target_label])

        svmPredictions = svmModel.predict(testing_df)
        rfPredictions = RFmodel.predict(testing_df)

        # now compare the results

        svmComparisonDF = pd.DataFrame()
        svmComparisonDF['y_Actual'] = y_test
        svmComparisonDF['y_Predicted'] = svmPredictions

        rfComparisonDF = pd.DataFrame()
        rfComparisonDF['y_Actual'] = y_test
        rfComparisonDF['y_Predicted'] = rfPredictions

        svmConfMat = pd.crosstab(svmComparisonDF['y_Actual'], svmComparisonDF['y_Predicted'], rownames=['Actual'],
                                 colnames=['Predicted'])
        rfConfMat = pd.crosstab(rfComparisonDF['y_Actual'], rfComparisonDF['y_Predicted'], rownames=['Actual'],
                                colnames=['Predicted'])

        try:
            svmConfMat[2.0]
        except:
            svmConfMat[2.0] = [0, 0]
        try:
            svmConfMat[1.0]
        except:
            svmConfMat[1.0] = [0, 0]

        try:
            rfConfMat[2.0]
        except:
            rfConfMat[2.0] = [0, 0]
        try:
            rfConfMat[1.0]
        except:
            rfConfMat[1.0] = [0, 0]

        try:
            svm_TP = svmConfMat[1.0].values[0]
            svm_TN = svmConfMat[2.0].values[1]
            svm_FP = svmConfMat[1.0].values[1]
            svm_FN = svmConfMat[2.0].values[0]
        except:
            print(" d")
        svm_utility_score = 1 - ((svm_FN / (svm_FN + svm_TP)) + (svm_FP / (svm_FP + svm_TN))) / 2
        svm_acc_score = (svm_TP + svm_TN) / (svm_FN + svm_TP + svm_FP + svm_TN)

        rf_TP = rfConfMat[1.0].values[0]
        rf_TN = rfConfMat[2.0].values[1]
        rf_FP = rfConfMat[1.0].values[1]
        rf_FN = rfConfMat[2.0].values[0]
        rf_utility_score = 1 - ((rf_FN / (rf_FN + rf_TP)) + (rf_FP / (rf_FP + rf_TN))) / 2
        rf_acc_score = (rf_TP + rf_TN) / (rf_FN + rf_TP + rf_FP + rf_TN)

        test_result_df = x_test.copy()
        test_result_df["true_credit_decision"] = y_test
        test_result_df["svm_credit_decision"] = svmPredictions
        test_result_df["rf_credit_decision"] = rfPredictions

        results["lambda_value"] = lam_val
        results["svm_acc"] = svm_acc_score
        results["svm_util"] = svm_utility_score
        # results["svm_age_legal_desperate_Impact"] = ((len(test_result_df.loc[test_result_df["Age_cat"] == 1.0].loc[test_result_df["svm_credit_decision"] == 1.0])/len(test_result_df.loc[test_result_df["Age_cat"] == 1.0]))*100 / (len(test_result_df.loc[test_result_df["Age_cat"] == 2.0].loc[test_result_df["svm_credit_decision"] == 1.0])/ len(test_result_df.loc[test_result_df["Age_cat"] == 2.0]))*100)/100
        # results["svm_sex_legal_desperate_Impact"] = (len(test_result_df.loc[test_result_df["sex"] == 1].loc[test_result_df["svm_credit_decision"] == 2]) / len(test_result_df.loc[test_result_df["sex"] == 1])*100 / (len(test_result_df.loc[test_result_df["sex"] == 1].loc[test_result_df["svm_credit_decision"] == 1.0])/len(test_result_df.loc[test_result_df["sex"] == 1]))*100)/100

        results["rf_acc"] = rf_acc_score
        results["rf_util"] = rf_utility_score
        # results["rf_age_legal_desperate_Impact"] = ((len(test_result_df.loc[test_result_df["Age_cat"] == 1.0].loc[test_result_df["rf_credit_decision"] == 1.0])/len(test_result_df.loc[test_result_df["Age_cat"] == 1.0]))*100 / (len(test_result_df.loc[test_result_df["Age_cat"] == 2.0].loc[test_result_df["rf_credit_decision"] == 1.0])/ len(test_result_df.loc[test_result_df["Age_cat"] == 2.0]))*100)/100
        # results["rf_sex_legal_desperate_Impact"] = (len(test_result_df.loc[test_result_df["sex"] == 1].loc[test_result_df["rf_credit_decision"] == 2]) / len(test_result_df.loc[test_result_df["sex"] == 1])*100 / (len(test_result_df.loc[test_result_df["sex"] == 1].loc[test_result_df["rf_credit_decision"] == 1.0])/len(test_result_df.loc[test_result_df["sex"] == 1]))*100)/100

        # results["svm_confMat"] = svmConfMat
        # results["rf_confMat"] = rfConfMat

        repair_test_results[f"{lam_val}"] = results
    return repair_test_results


def resultsGraph(repair_results, CNDresults):
    """
    test_result_df = x_test.copy()
        test_result_df["true_credit_decision"] = y_test
        test_result_df["svm_credit_decision"] = svmPredictions
        test_result_df["rf_credit_decision"] = rfPredictions

        results["lambda_value"] = lam_val
        results["svm_acc"] = svm_acc_score
        results["svm_util"] = svm_utility_score

        results["rf_acc"] = rf_acc_score
        results["rf_util"] = rf_utility_score
    :param repair_results:
    :param CNDresults:
    :return:
    """
    repair_results["SVM accuracy score"] = repair_results["svm_acc"]
    repair_results["SVM utility score"] = repair_results["svm_util"]
    repair_results["RF accuracy score"] = repair_results["rf_acc"]
    repair_results["RF utility score"] = repair_results["rf_util"]

    sns.set(style="darkgrid")
    figure, axis = plt.subplots(2, 2, figsize=(7, 7))

    sns.lineplot(data=repair_results, x="lambda_value", y="SVM accuracy score", color="red", ax=axis[0, 0])
    sns.lineplot(data=repair_results, x="lambda_value", y="SVM utility score", color="green",
                 ax=axis[0, 1])
    sns.lineplot(data=repair_results, x="lambda_value", y="RF accuracy score", color="blue",
                 ax=axis[1, 0])
    sns.lineplot(data=repair_results, x="lambda_value", y="RF utility score", color="yellow",
                 ax=axis[1, 1])

    plt.show()



    CNDresults["SVM accuracy score"] = CNDresults["svm_acc"]
    CNDresults["SVM utility score"] = CNDresults["svm_util"]
    CNDresults["RF accuracy score"] = CNDresults["rf_acc"]
    CNDresults["RF utility score"] = CNDresults["rf_util"]

    sns.set(style="darkgrid")
    figure, axis = plt.subplots(2, 2, figsize=(7, 7))

    sns.lineplot(data=CNDresults, x="max_KCDM_df_score", y="SVM accuracy score", color="red", ax=axis[0, 0])
    sns.lineplot(data=CNDresults, x="max_KCDM_df_score", y="SVM utility score", color="green",
                 ax=axis[0, 1])
    sns.lineplot(data=CNDresults, x="max_KCDM_df_score", y="RF accuracy score", color="blue",
                 ax=axis[1, 0])
    sns.lineplot(data=CNDresults, x="max_KCDM_df_score", y="RF utility score", color="yellow",
                 ax=axis[1, 1])

    #plt.show()






def runExperiments():
    target_label = 'credit_decision'
    """
    repair_results = repairExperiment()
    CNDresults = CNDExperiment()


    file_pi = open('repair_results.obj', 'wb')
    pickle.dump(repair_results, file_pi)

    file_pi = open('CNDresults.obj', 'wb')
    pickle.dump(CNDresults, file_pi)
    """
    filehandler = open("CNDresults.obj", 'rb')
    CNDresults = pd.DataFrame(pickle.load(filehandler)).T

    filehandler = open("repair_results.obj", 'rb')
    repair_results = pd.DataFrame(pickle.load(filehandler)).T

    resultsGraph(repair_results, CNDresults)


if __name__ == '__main__':
    target_label = 'credit_decision'
    #runExperiments()



    """ column reports for creating latex graphs
    dataAnalysis = AnalyseData(getData())
    dataAnalysis.dataframeSubSet()
    

    reports = dataAnalysis.runReports()
    for r in reports:
        dataAnalysis.printReportGraphStyle(r)
    """

    """analyses bias in dataset
    df = formatData()
    dataAnalysis = AnalyseData(df)
    dataAnalysis.dataframeSubSet()
    """

    df = formatData()
    simpleGroupEquilisation(df)

    CND_df = Repair().CND(df, 'credit_decision', ["Age_cat", "sex"], [2.0, 1], [1.0, 2], 0.001)
    repaired_df = Repair().repair(df, ['credit_decision'], ["Age_cat", "sex"], 0.5)

    eqilised_outcome_sex_df, equilised_sex_df = fairSexDF()
    eqilised_outcome_age_df, equilised_age_df = fairAgeDF()

    #"""HOWMODELS GENERALISED TO DATASET
    #modelDataGeneralise(df)
    #"""

    #""" equilisng subgroups to try and balance DF
    simpleGroupEquilisation(df)
    #"""
