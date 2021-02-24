import pandas as pd
import re

class FileErr(Exception):
    pass

def openXLS(path):
    try:
        keywords =  pd.read_excel(path)
        return keywords
    except Exception as e:
        raise FileErr() from e


if __name__ == '__main__':
    dataPath = "../data/adult.data"
    data = openXLS(dataPath)