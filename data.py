import numpy as np
import pandas as pd

'''
load data from given csv file and return pandas dataframe
'''
def load_data(file_name):
    df = pd.read_csv(file_name)
    return df

'''
convert all Strings to numbers in each column of the dataframe and substitute all nan with 0
'''
def convert_to_numerical(df):
    df['MSZoning'] = df['MSZoning'].replace({'A': 1, 'C (all)': 2, 'FV': 3, 'I': 4, 'RH': 5, 'RL': 6, 'RP': 7, 'RM': 8})
    df['Street'] = df['Street'].replace({'Grvl': 1, 'Pave': 2})
    df['Alley'] = df['Alley'].replace({np.nan : 0, 'Grvl': 1, 'Pave': 2}) 
    df['LotShape'] = df['LotShape'].replace({'Reg' : 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}) 
    df['Alley'] = df['Alley'].replace({np.nan : 0, 'Grvl': 1, 'Pave': 2}) 
    df['LandContour'] = df['LandContour'].replace({'Lvl' : 1, 'Bnk': 2, 'HLS': 3, 'Low': 4}) 
    df['Utilities'] = df['Utilities'].replace({'AllPub' : 1, 'NoSewr': 2, 'NoSeWa': 3, 'ELO': 4}) 
    df['LotConfig'] = df['LotConfig'].replace({'Inside' : 1, 'Corner': 2, 'CulDSac': 3, 'FR2': 4, 'FR3': 5}) 
    df['LandSlope'] = df['LandSlope'].replace({'Gtl' : 1, 'Mod': 2, 'Sev': 3}) 
    df['Neighborhood'] = df['Neighborhood'].replace({'Blmngtn' : 1, 'Blueste': 2, 'BrDale': 3, 'BrkSide': 4, 'ClearCr': 5, 'CollgCr': 6, 'Crawfor': 7, 'Edwards': 8, 'Gilbert': 9, 'IDOTRR': 10, 'MeadowV': 11, 'Mitchel': 12, 'NAmes': 13, 'NoRidge': 14, 'NPkVill': 15, 'NridgHt': 16, 'NWAmes': 17, 'OldTown': 18, 'SWISU': 19, 'Sawyer': 20, 'SawyerW': 21, 'Somerst': 22, 'StoneBr': 23, 'Timber': 24, 'Veenker': 25}) 
    df['Condition1'] = df['Condition1'].replace({'Artery' : 1, 'Feedr': 2, 'Norm': 3, 'RRNn': 4, 'RRAn': 5, 'PosN': 6, 'PosA': 7, 'RRNe': 8, 'RRAe': 9}) 
    df['Condition2'] = df['Condition2'].replace({'Artery' : 1, 'Feedr': 2, 'Norm': 3, 'RRNn': 4, 'RRAn': 5, 'PosN': 6, 'PosA': 7, 'RRNe': 8, 'RRAe': 9}) 
    df['BldgType'] = df['BldgType'].replace({'1Fam' : 1, '2fmCon': 2, 'Duplex': 3, 'TwnhsE': 4, 'Twnhs': 5}) 
    df['HouseStyle'] = df['HouseStyle'].replace({'1Story' : 1, '1.5Fin': 2, '1.5Unf': 3, '2Story': 4, '2.5Fin': 5, '2.5Unf': 6, 'SFoyer': 7, 'SLvl': 8}) 
    df['RoofStyle'] = df['RoofStyle'].replace({'Flat' : 1, 'Gable': 2, 'Gambrel': 3, 'Hip': 4, 'Mansard': 5, 'Shed': 6}) 
    df['RoofMatl'] = df['RoofMatl'].replace({'ClyTile' : 1, 'CompShg': 2, 'Membran': 3, 'Metal': 4, 'Roll': 5, 'Tar&Grv': 6, 'WdShake': 7, 'WdShngl': 8}) 
    df['Exterior1st'] = df['Exterior1st'].replace({'AsbShng' : 1, 'AsphShn': 2, 'BrkComm': 3, 'BrkFace': 4, 'CBlock': 5, 'CemntBd': 6, 'HdBoard': 7, 'ImStucc': 8, 'MetalSd': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15, 'WdShing': 16, 'Other': 17}) 
    df['Exterior2nd'] = df['Exterior2nd'].replace({'AsbShng' : 1, 'AsphShn': 2, 'Brk Cmn': 3, 'BrkFace': 4, 'CBlock': 5, 'CmentBd': 6, 'HdBoard': 7, 'ImStucc': 8, 'MetalSd': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15, 'Wd Shng': 16, 'Other': 17}) 
    df['MasVnrType'] = df['MasVnrType'].replace({'None': 0, np.nan: 0,'BrkCmn' : 1, 'BrkFace': 2, 'CBlock': 3, 'Stone': 4}) 
    df['ExterQual'] = df['ExterQual'].replace({'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['ExterCond'] = df['ExterCond'].replace({'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['Foundation'] = df['Foundation'].replace({'BrkTil' : 1, 'CBlock': 2, 'PConc': 3, 'Slab': 4, 'Stone': 5, 'Wood': 6}) 
    df['BsmtQual'] = df['BsmtQual'].replace({np.nan: 0, 'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['BsmtCond'] = df['BsmtCond'].replace({np.nan: 0, 'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['BsmtExposure'] = df['BsmtExposure'].replace({np.nan: 0, 'No' : 1, 'Mn': 2, 'Av': 3, 'Gd': 4}) 
    df['BsmtFinType1'] = df['BsmtFinType1'].replace({np.nan: 0, 'Unf' : 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}) 
    df['BsmtFinType2'] = df['BsmtFinType2'].replace({np.nan: 0, 'Unf' : 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}) 
    df['Heating'] = df['Heating'].replace({'Floor': 0, 'GasA' : 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5}) 
    df['HeatingQC'] = df['HeatingQC'].replace({'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['CentralAir'] = df['CentralAir'].replace({'N' : 0, 'Y': 1}) 
    df['Electrical'] = df['Electrical'].replace({np.nan: 0, 'SBrkr' : 1, 'FuseA': 2, 'FuseF': 3, 'FuseP': 4, 'Mix': 5}) 
    df['Electrical'] = df['Electrical'].replace({np.nan: 0, 'SBrkr' : 1, 'FuseA': 2, 'FuseF': 3, 'FuseP': 4, 'Mix': 5}) 
    df['KitchenQual'] = df['KitchenQual'].replace({'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['Functional'] = df['Functional'].replace({'Sal' : 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}) 
    df['FireplaceQu'] = df['FireplaceQu'].replace({np.nan: 0, 'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['GarageType'] = df['GarageType'].replace({np.nan: 0, '2Types' : 1, 'Attchd': 2, 'Basment': 3, 'BuiltIn': 4, 'CarPort': 5, 'Detchd': 6}) 
    df['GarageFinish'] = df['GarageFinish'].replace({np.nan: 0, 'Unf' : 1, 'RFn': 2, 'Fin': 3}) 
    df['GarageQual'] = df['GarageQual'].replace({np.nan: 0, 'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})  
    df['GarageCond'] = df['GarageCond'].replace({np.nan: 0, 'Po' : 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}) 
    df['PavedDrive'] = df['PavedDrive'].replace({'N': 0, 'P' : 1, 'Y': 2}) 
    df['PoolQC'] = df['PoolQC'].replace({np.nan: 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}) 
    df['Fence'] = df['Fence'].replace({np.nan: 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}) 
    df['MiscFeature'] = df['MiscFeature'].replace({np.nan: 0, 'TenC': 1, 'Shed': 2, 'Othr': 3, 'Gar2': 4, 'Elev': 5}) 
    df['SaleType'] = df['SaleType'].replace({'Oth': 1, 'ConLD': 2, 'ConLI': 3, 'ConLw': 4, 'Con': 5, 'COD': 6, 'New': 7, 'VWD': 8, 'CWD': 9, 'WD': 10}) 
    df['SaleCondition'] = df['SaleCondition'].replace({'Partial': 1, 'Family': 2, 'Alloca': 3, 'AdjLand': 4, 'Abnorml': 5, 'Normal': 6}) 
    df = df.replace({np.nan: 0})
    df = df.drop('Id', axis=1)
    
    return df

'''
split given dataframe in to train fataframe and validation dataframe
'''
def train_test_split(df, test_split=0.0):
    df = df.sample(frac=1)
    split = int(df.shape[0] * test_split)
    df_train = df.iloc[split+1:]
    df_validate = df.iloc[:split]
    return df_train, df_validate

'''
extract the SalePrice column from given dataframe
'''
def extract_labels(df):
    df_data = df.drop('SalePrice', axis=1)
    df_label = df['SalePrice']
    return df_data, df_label

'''
prepare data for training a DNN
'''
def prepare_data(file_names, test_split):
    # load train data and convert all Strings to numbers
    df_train = convert_to_numerical(load_data(file_names[0]))
    # load test data and convert all Strings to numbers
    df_test = convert_to_numerical(load_data(file_names[1]))
    # normalize of train and test data
    df_train, X_test = normalize(df_train, df_test)
    # split train data in train and validation set
    df_train, df_validate = train_test_split(df_train, test_split=test_split)
    if df_train.shape[1] == 80:
        # extract labels of test and validation set
        X_train, y_train = extract_labels(df_train)
        X_validate, y_validate = extract_labels(df_validate) 

    # return all datas as numpy array
    return X_train.to_numpy(), X_validate.to_numpy(), y_train.to_numpy(), y_validate.to_numpy(), X_test.to_numpy()

'''
normalize all data between 0 and 1
'''
def normalize(df, df_test):
    # iterate through all column of given dataframes
    for column in df_test:
        # get the max value of given columns of dataframe
        maximum = max(max_value(df[column]), max_value(df_test[column]))
        # divide all values in both columns with maximum value 
        df[column] = df[column]/maximum
        df_test[column] = df_test[column]/maximum
    return df, df_test

'''
find the maximum value of given dataframe column
'''
def max_value(df):
    max = 0
    # iterate through all rows in column
    for row in df:
        # save biggest value found
        if max < row:
            max = row
    return max

if __name__ == '__main__':
    df_train = load_data('train.csv')
    print(df_train)

    df_test = load_data('test.csv')
    print(df_test)

    df_train = convert_to_numerical(df_train)
    print(df_train)

    df_test = convert_to_numerical(df_test)

    X_train, X_validate, y_train, y_validate, X_test, _ = prepare_data(['train.csv', 'test.csv'], test_split=0.1)
    print(f'X: {X_train} \n Shape: {X_train.shape}')
    print()
    print(f'y: {y_train} \n Shape: {y_train.shape}')
    print()
    print()
    print(f'X: {X_validate} \n Shape: {X_validate.shape}')
    print(f'y: {y_validate} \n Shape: {y_validate.shape}')
    print()
    print()
    print(f'X: {X_test} \n Shape: {X_test.shape}')
    # df_train, df_test, _ = normalize(df_train, df_test)
    # print(df_train)
    # print(df_test)

    