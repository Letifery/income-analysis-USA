import numpy as np
import pandas as pd
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
		