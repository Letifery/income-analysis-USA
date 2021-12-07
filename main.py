# imports 
import pandas as pd
import numpy as np

def feature_selection(factorize = True) -> 'DataFrame':
    '''
		Implements the Feature selection and Dataset preparation for the Adult dataset. 
		https://archive.ics.uci.edu/ml/datasets/adult

		Parameters
		------------
		factorize	                :Your Input - Has to a bool, default is True
		  
		Returns
		------------
		prepared_Adult_dataset  :A DataFrame, without missing values in any row, 
                                    income and sex are now binary, 
                                    delated all rows, where 'native-country' unequal 'United-States'
                                    and delated the columns ['fnlwgt', 'education', 'marital-status', 'relationship', 'race', 'native-country'], 
                                    the columns 'education-num' and 'workclass' are swaped, 
                                    if factorize == True: 
                                        columns 'workclass' and 'occupation' factorized e. g. [['a'], ['b'], ['c'], ['a'], ['a']] -> [[1], [2], [3], [1], [1]]
                                    else: 
                                        columns 'workclass' and 'occupation' are One Hot Encode e.g [['a'], ['b'], ['c'], ['a'], ['a']] -> [['a','b','c']
                                                                                                                                            [ 1,  0,  0]
                                                                                                                                            [ 0,  1,  0]
                                                                                                                                            [ 0,  0,  1]
                                                                                                                                            [ 1,  0,  0]
                                                                                                                                            [ 1,  0,  0]]
	'''

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=',', header=None)
    df = df.set_axis(['age', 'workclass','fnlwgt', 'education','education-num', 'marital-status','occupation', 'relationship',
                   'race', 'sex','capital-gain', 'capital-loss','hours-per-week', 'native-country','income'], axis=1, inplace=False)
    
    df.drop(['fnlwgt', 'education', 'marital-status', 'relationship', 'race'], axis = 1, inplace = True) 
    data = np.array(df)

    data =  np.array([row  for row in data if ' ?' not in row])

    data = data[data[:,8 ] == " United-States"]
    data = np.delete(data, np.s_[8], axis=1)    
 
    arr = np.append(np.reshape(data[:, 0], (len(data[:, 0]), 1)), np.reshape(data[:, 2], (len(data[:, 2]), 1)), axis=1)
    df = pd.DataFrame(data = data, columns=['age', 'workclass', 'education-num','occupation',  
                                            'sex','capital-gain', 'capital-loss','hours-per-week', 'income'])
    
    if factorize: 
        column_names = df.columns
        for col in ['workclass', 'occupation']:
            codes, uniques = pd.factorize(df[col]) 
            arr = np.c_[arr, np.reshape(codes, (len(codes),1))]

    else:
        for col in ['workclass','occupation']:
            dummies = pd.get_dummies(df[col])
            arr = np.append(arr, dummies, axis=1)

        codes, uniques = pd.factorize(df['sex']) 
        arr = np.append(arr, np.reshape(codes, (len(codes),1)), axis=1)

    arr = np.c_[arr, np.array(df[['capital-gain', 'capital-loss','hours-per-week']])]
    codes, uniques = pd.factorize(df['income']) 
    arr = np.append(arr, np.reshape(codes, (len(codes),1)), axis=1)

    return  pd.DataFrame(data=arr.astype(float))
