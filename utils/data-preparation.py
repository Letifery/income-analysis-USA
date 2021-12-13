import pandas as pd
import numpy as np

def feature_selection(df) -> 'DataFrame':
    '''
    This function removes unusable columns (based on the results of our uni-/bivariate analysis)
    and modifies several other columns (i.e, mapping strings to numbers)
    
    Parameters
    ------------
    Returns
    ------------
    prepared_Adult_dataset  :A DataFrame, with following characteristics:
                                 - income and sex are now numeric, 
                                 - rows have the equality 'native-country' == 'United-States'
                                 - without columns: ['fnlwgt', 'education', 'marital-status', 'relationship', 'race', 'native-country']
    '''
    
    df.drop(['fnlwgt', 'education', 'marital-status', 'relationship', 'race'], axis = 1, inplace = True) 
    data = np.array(df)

    data =  np.array([row  for row in data if ' ?' not in row])

    data = data[data[:,8 ] == " United-States"]
    data = np.c_[data[:, :8], data[:, 9]]
 
    df = pd.DataFrame(data=data)

    codes, uniques = pd.factorize(df[4]) 
    codes = codes.reshape(np.shape(data[:,4]))
    data[:, 4] = codes
    codes, uniques = pd.factorize(df[8]) 
    codes = codes.reshape(np.shape(data[:,8]))
    data[:, 8] = codes
    df = pd.DataFrame(data=data)
    df = df.set_axis(['age', 'workclass','education-num','occupation', 'sex','capital-gain', 'capital-loss','hours-per-week','income'], axis=1, inplace=False)
    return  df

def factorize(df):
    """
    This function factorizes all columns, where the column is of type str
    Parameters
    ------------
    df                    : is the Adult dataset in a DataFrame with the columnnames  
                          ['age', 'workclass','fnlwgt','education','education-num', 'marital-status','occupation', 
                          relationship','race', 'sex','capital-gain', 'capital-loss','hours-per-week', 'native-country','income']
		  
    Returns
    ------------
    prepared_Adult_dataset  :A DataFrame, with following characteristics
                             - columns with type string now transformed like this:
                               e. g. [['a'], ['b'], ['c'], ['a'], ['a']] -> [[1], [2], [3], [1], [1]]
    """

    column_names = df.columns
    arr = np.zeros(np.shape(np.array(df)))
    for i,col in enumerate(column_names):
        codes, uniques = pd.factorize(df[col]) 
        if isinstance(uniques[0], str):
            arr[:, i] = codes.reshape(np.shape(arr[:,i]))
        else: 
            arr[:, i] = df[col]
    return pd.DataFrame(data=arr.astype(float))

def one_hot_encodings(df):
    """
    This function one hot encodings all columns, where the column is of type str
    Parameters
    ------------
    df               : is the Adult dataset in a DataFrame with the columnnames  
                     ['age', 'workclass','fnlwgt','education','education-num', 'marital-status','occupation', 
                     relationship','race', 'sex','capital-gain', 'capital-loss','hours-per-week', 'native-country','income']

     Returns
     ------------
     prepared_Adult_dataset  :A DataFrame, with following characteristics
                             - columns with type string now transformed like this:
                               e.g [['a'], ['b'], ['c'], ['a'], ['a']] -> [ ['a','b','c'] <- column names
                                                                            [ 1,  0,  0]
                                                                            [ 0,  1,  0]
                                                                            [ 0,  0,  1]
                                                                            [ 1,  0,  0]
                                                                            [ 1,  0,  0]]                          
    """

    column_names = df.columns
    arr = np.array(df['age'])
    for i,col in enumerate(column_names[1:]):
        if isinstance(df[col][0], str):
            dummies = pd.get_dummies(df[col])
            arr  = np.c_[arr, np.array(dummies)]
        else: 
            arr = np.c_[arr, np.array(df[col])]
    return pd.DataFrame(data=arr.astype(float))
