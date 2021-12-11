import pandas as pd
import numpy as np


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=',', header=None)
df = df.set_axis(['age', 'workclass','fnlwgt', 'education','education-num', 'marital-status','occupation', 'relationship',
                   'race', 'sex','capital-gain', 'capital-loss','hours-per-week', 'native-country','income'], axis=1, inplace=False)

def feature_selection(df) -> 'DataFrame':
    '''
		This function removes unusable columns (based on the results of our uni-/bivariate analysis)
		and modifies several other columns (i.e, mapping strings to numbers)
		
		Parameters
		------------
		df			: is the Adult dataset in a DataFrame with the columnnames  
					['age', 'workclass','fnlwgt','education','education-num', 'marital-status','occupation', 
					'relationship','race', 'sex','capital-gain', 'capital-loss','hours-per-week', 'native-country','income']
		
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

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=',', header=None)
df = df.set_axis(['age', 'workclass','fnlwgt', 'education','education-num', 'marital-status','occupation', 'relationship',
                   'race', 'sex','capital-gain', 'capital-loss','hours-per-week', 'native-country','income'], axis=1, inplace=False)
