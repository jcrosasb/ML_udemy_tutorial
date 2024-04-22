import numpy as np
import matplotlib.pyplot as plt


class Plotter:

    def __init__(self, x, y, xlabel, ylabel, predict, evaluate):
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.predict = predict
        self.evaluate = evaluate

        if self.x.shape[1] == 1:
            plt.scatter(x, y, color='red')
            plt.plot(np.sort(x, axis=0), self.predict(np.sort(self.x, axis=0)), color='blue')
            plt.grid()
            plt.xlabel(f'{self.xlabel[0]}')
            plt.ylabel(f'{self.ylabel}')

            ax = plt.gca()
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5)
            textstr = f'$MSE$={round(self.evaluate()[0],6)}, $R^2$={round(self.evaluate()[1],6)}'
            ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=bbox_props)

            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Generate grid points for plotting the surface
            X1_grid, X2_grid = np.meshgrid(np.linspace(min(x[:, 0]), max(x[:, 0]), 100), np.linspace(min(x[:, 1]), max(x[:, 1]), 100))
            Z_grid = self.predict(np.column_stack((X1_grid.ravel(), X2_grid.ravel())))
            Z_grid = Z_grid.reshape(X1_grid.shape)

            # Plot the data points
            ax.scatter(x[:, 0], x[:, 1], y, color='red')
            #ax.scatter(x[:, 0], y, color='red', label=self.xlabel[0])
            #ax.scatter(x[:, 1], y, color='green', label=self.xlabel[1])

            # Plot the decision surface
            ax.plot_surface(X1_grid, X2_grid, Z_grid, alpha=0.5, cmap='magma')

            ax.set_xlabel(xlabel[0])
            ax.set_ylabel(xlabel[1])
            ax.set_zlabel(ylabel)
            ax.set_title(f'$MSE=${round(self.evaluate()[0], 6)}, $R^2=${round(self.evaluate()[1], 6)}')

            plt.show()