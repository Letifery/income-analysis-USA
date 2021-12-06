def feature_selection(df: 'DataFrame') -> ('np.array(data)', '[column_names]' ):

    '''
		Implements the Feature selection and Dataset preparation for the Adult dataset
		    https://archive.ics.uci.edu/ml/datasets/adult

		Parameters
		------------
		df		                  :Your Input - Has to be the Adult dataset, as a DataFrame,
                              with the correct column names 
		  
		Returns
		------------
		prepared_Adult_dataset  :A DataFrame, without missing values in any row, 
                              changed the income column type in binary and renamed the row to 'income_over_50K',
                              delated all rows, where 'native-country' unequal 'United-States'
                              and delated the columns ['fnlwgt', 'education', 'marital-status', 
                                                       'relationship', 'race', 'native-country'] 
	  '''

    df.drop(['fnlwgt', 'education', 'marital-status', 'relationship', 'race'], axis = 1, inplace = True) 
    df_arr = np.array(df)

    df_arr = df_arr[df_arr[:,8 ] == " United-States"]
    df_arr = np.delete(df_arr, np.s_[8], axis=1)    
    
    df_arr = np.array([row  for row in df_arr if ' ?' not in row])
    df_arr = np.c_[df_new, [0 if " <=50K" else 1  for value in df_arr.T[8]]]

    return pd.DataFrame(data=df_arr, index= df_arr, columns=['age', 'workclass', 'education-num','occupation',  'sex',
                                                             'capital-gain', 'capital-loss','hours-per-week','income_over_50K'])
