import pandas as pd


def repair(wholeDF, target_cols, protected_cols):
    repairedDF = pd.DataFrame

    #get unique vlaues for each col
    col_vals = dict()
    for col in (target_cols.append(protected_cols)):
        col_vals[col] = list(wholeDF[col].unique())

    #add stratified collumbs to dataframe
    stratified_cols = []
    for col in protected_cols:
        if len(stratified_cols) == 0:
            stratified_cols = list(wholeDF[col].unique())
        else:
            new_strat = []
            for col_old in stratified_cols:
                for col in list(wholeDF[col].unique()):
                    new_strat.extend(col_old + [c for c in col])
            stratified_cols = new_strat

    stratified_series = []
    for col in stratified_cols:
        conditions = []
        for result, category in col, protected_cols:
            conditions += wholeDF[category] == result

        indexes = wholeDF.index
        for condition in conditions:
            indexes = set(indexes).union(set(wholeDF.index[condition]))

        stratified_series = []
        for i in len(wholeDF["Age"])






