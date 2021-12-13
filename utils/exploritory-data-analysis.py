import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def multi_occ_plot(df, typemask:[int]):
		'''
		Plots the occurences of a column in a dataframe, whilst the type of plot depends
		on the values in the typemask array.
		
		Parameters
		------------
		df			: A dataframe with a header row
		typemask	: An array of length k = |header|, where the elements have to be in range [0,3].
					The respective number specifies what plotting method will be used for that column;
					0 - Skips column/No plot
					1 - Barplot
					2 - Lineplot
					3 - Boxplot
		
		Returns
		------------
		Nothing
		'''

	for i,x in enumerate(df.columns.values.tolist()):
		match typemask[i]:
			case 1:
				df[x].value_counts().sort_index().plot(kind='bar', xlabel=x, ylabel="occurences")
			case 2:
				df[x].value_counts().sort_index().plot(xlabel=x, ylabel="occurences")
			case 3:
				df[x].value_counts().sort_index().plot(kind='box', xlabel=x)
		plt.show()
		
		
def correlaion_matrix_plot(df: 'DataFrame'):
    '''
	Plots the the correlation_matrix of the DataFrame df 
		
	Parameters
	------------
	df			: A dataframe with a header row
    '''
    column_names = df.columns
    data_arr = np.array([])
    for column_name in column_names:
        codes, uniques = pd.factorize(df[column_name]) 
        if data_arr.size == 0:
            data_arr = codes
        else:
            data_arr = np.c_[data_arr, codes]
            
    df = pd.DataFrame(data_arr)
    df = df.set_axis(column_names, axis=1, inplace=False)

    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(),cmap=plt.cm.Reds,annot=True)
    plt.title('Heat map displaying the relationship between the features of the data',
         fontsize=13)
    plt.show()
		
