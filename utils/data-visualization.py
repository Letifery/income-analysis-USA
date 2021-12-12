class data_visualization():
    def __init__(self, df, target):
        self.features = df.loc[:, df.columns!=target].values
        self.target =  df[target].values.astype(int)

    def dimension_reduction_to_2D(self):
        fa = FactorAnalysis(n_components = 2).fit_transform(self.features)
        print(fa)
        return fa 

    def dim_re_to_2D_visualization(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        data = self.dimension_reduction_to_2D()
        over50K = data[self.target == 1]
        below50K = data[self.target == 0]

        ax.scatter(over50K[:,0], over50K[:,1], color = 'red',  alpha = 0.1, label = "Income > 50K")
        ax.scatter(below50K[:,0], below50K[:,1],color = 'blue',  alpha = 0.1, label = "Income <= 50K")

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.legend(loc="upper right")
        plt.show()
    
    def plot_decision_boundary2D(self,model):
        X = self.dimension_reduction_to_2D()
        y = np.array(self.target).flatten()

        model.fit(X, y)

        # Step size of the mesh
        h = .01   

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for points in mesh 
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        # Predictions to obtain the classification results
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Plotting
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.xlabel("Feature-1",fontsize=15)
        plt.ylabel("Feature-2",fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
