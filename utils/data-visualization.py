import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class data_visualization():
    def prepare_data(self, df, best_modle, NB):
        '''
        Visualization of decision boundaries from the classifiers.
        A dimension reduction to two features is applied, to visualize the 
        boundaries. 
        ------------
        df          : dataframe
                    Your datatable
        best_model  :(np.array(int))
                    An indexlist of the fold which has performed best overall (only need this)
        NB          : bool
                    Data from NB, must be positive, if negative
        Returns
        ------------
        features_0  :(np.array(float)) X_train of the best performing modle
        y_0         :(np.array(float)) y_train of the best performing modle
        features_1  :(np.array(float)) X_train of the second best performing modle
        y_1         :(np.array(float)) y_train of the second best performing modle
        '''
        arr_0 = np.array(df.iloc[best_modle[0]])
        arr_1 = np.array(df.iloc[best_modle[1]])
        features_0 = self.dimension_reduction_to_2D(arr_0[:, :-1])
        features_1 = self.dimension_reduction_to_2D(arr_1[:, :-1])
        y_0 = arr_0[:, -1]
        y_1 = arr_1[:, -1]
        if NB:
            features_0 = MinMaxScaler().fit_transform(self.dimension_reduction_to_2D(features_0))
            features_1 = MinMaxScaler().fit_transform(self.dimension_reduction_to_2D(features_1))
        return features_0, y_0, features_1, y_1


    def visualization(self, df, best_model, modle, NB = False):
        '''
        Visualization of decision boundaries from the classifiers.
        A dimension reduction to two features is applied, to visualize the 
        boundaries. 
        ------------
        df          : dataframe
                    Your datatable
        best_model  :(np.array(int))
                    An indexlist of the fold which has performed best overall (only need this)
        modle:      is classifier with the parameters, of the best fold 
        NB          : bool
                    Data from NB, must be positive, if negative
        '''
        f0_redu, y_0, f1_redu, y_1 =  self.prepare_data(df, best_model, NB)
        self.plot_decision_boundary2D(f0_redu, y_0, modle)
        self.plot_decision_boundary2D(f1_redu, y_1, modle)


    def dimension_reduction_to_2D(self, data):
        """
        Performs a dimension reduction to 2 components. Use StandardScaler, for 
        a nicer visualization. For the reduction is used PCA. 
        ------------
        df                  : (np.array(float)), data for dimension reduction 
        Returns
        ------------
        principalComponents : (np.array(float)) data of only to features 
        """
        x = StandardScaler().fit_transform(data)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        return principalComponents
    

    def plot_decision_boundary2D(self,features, target, modle):
        """
        Plots the feature points and draws the decision boundaries from the classifier.
        We used part of the code of the Website for this function:
        https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html
        ------------
        features    :(np.array(float))  X_train for the best fold 
        target      :(np.array(float)) target data of the feature data  
        modle       : scikit learn modle 
        """
        y = np.array(target).flatten()
        modle.fit(features, target)

        # Step size of the mesh
        h = .01   

        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for points in mesh 
        Z = modle.predict(np.c_[xx.ravel(), yy.ravel()])    

        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        # Predictions to obtain the classification results
        Z = modle.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Plotting
        plt.title(modle)
        colors = ['indigo', 'orange']
        co = [colors[x] for x in y.astype(int)]
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.scatter(features[:, 0], features[:, 1], c= co, alpha = 0.1)
        red_patch = mpatches.Patch(color='indigo', label="Income > 50K")
        blue_patch = mpatches.Patch(color='orange', label="Income <= 50K")
        plt.legend(handles=[red_patch, blue_patch])
        plt.xlabel("Feature-1")
        plt.ylabel("Feature-2")
        plt.show()

# # Example 
# df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=',', header=None)
# df = df.set_axis(['age', 'workclass','fnlwgt', 'education','education-num', 'marital-status','occupation', 'relationship',
#                    'race', 'sex','capital-gain', 'capital-loss','hours-per-week', 'native-country','income'], axis=1, inplace=False)
# df = feature_selection(df)
# dv = data_visualization()

# # DT 
# modle = DecisionTreeClassifier(**best_model_DT[1])
# dv.visualization(df, best_model_DT[0], modle)

# # NN
# modle = Perceptron(**best_model_NN[1])
# dv.visualization(df, best_model_NN[0], modle)

# # KNN
# modle = KNeighborsClassifier(**best_model_KNN[1])
# dv.visualization(df, best_model_KNN[0], modle)

# NB 
# modle = ComplementNB(**best_model_NB[1])
# dv.visualization(df, best_model_NB[0], modle , True)

# # SVC
# modle = SVM(**best_model_SVM[1])
# dv.visualization(df, best_model_SVM[0], modle)
