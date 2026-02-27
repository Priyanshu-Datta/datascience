import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
   
    dataset = fetch_california_housing()

    
    X = dataset.data[:, [0]]  
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Coefficient:\t{model.coef_[0]:.4f}")
    print(f"Intercept:\t{model.intercept_:.4f}")
    print(f"Mean squared error:\t{mse:.4f}")
    print(f"R^2 score:\t{r2:.4f}")

    plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="actual")
    plt.plot(X_test, y_pred, color="red", linewidth=2, label="prediction")
    plt.xlabel(dataset.feature_names[0])
    plt.ylabel("Median house value (100k$)")
    plt.title("Simple Linear Regression: House value vs " + dataset.feature_names[0])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
