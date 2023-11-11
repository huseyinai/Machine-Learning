
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to visualize 3D scatter and regression plane
def visualize_regression(X, y, model, model_name, ax):
    # Fit model
    model.fit(X, y.ravel())
    
    # Predictions for visualization
    x1_pred = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_pred = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx_pred, yy_pred = np.meshgrid(x1_pred, x2_pred)
    model_viz = np.c_[xx_pred.ravel(), yy_pred.ravel()]
    
    predictions = model.predict(model_viz).reshape(xx_pred.shape)
    
    # Plotting the function
    ax.plot_surface(xx_pred, yy_pred, predictions, alpha=0.5, label=model_name)
    ax.scatter(X[:, 0], X[:, 1], y, color='red', s=10, alpha=1)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title(model_name)

# Generate synthetic data
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, (100, 2))
y = X[:, 0] - 2 * (X[:, 1] ** 2) + np.random.normal(-3, 3, 100)

# Adding outliers
X = np.vstack((X, np.random.uniform(low=-4, high=4, size=(20, 2))))
y = np.append(y, np.random.uniform(low=-10, high=10, size=20))

# Initialize models
linear_model = LinearRegression()
ransac_model = RANSACRegressor()
theilsen_model = TheilSenRegressor(random_state=42)
huber_model = HuberRegressor()
svr_model = SVR(kernel='rbf')
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Visualization
fig = plt.figure(figsize=(18, 12))

# Linear Regression
ax1 = fig.add_subplot(231, projection='3d')
visualize_regression(X, y, linear_model, 'Linear Regression', ax1)

# RANSAC Regression
ax2 = fig.add_subplot(232, projection='3d')
visualize_regression(X, y, ransac_model, 'RANSAC Regression', ax2)

# Theil-Sen Regression
ax3 = fig.add_subplot(233, projection='3d')
visualize_regression(X, y, theilsen_model, 'Theil-Sen Regression', ax3)

# Huber Regression
ax4 = fig.add_subplot(234, projection='3d')
visualize_regression(X, y, huber_model, 'Huber Regression', ax4)

# SVR
ax5 = fig.add_subplot(235, projection='3d')
visualize_regression(X, y, svr_model, 'SVR', ax5)

# Random Forest Regression
ax6 = fig.add_subplot(236, projection='3d')
visualize_regression(X, y, random_forest_model, 'Random Forest Regression', ax6)

plt.tight_layout()
plt.show()
