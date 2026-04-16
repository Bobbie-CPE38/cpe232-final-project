from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor


def train_model(X_train, y_train, n_iter=10):
    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        # n_jobs=-1,
        # verbose=1,
        n_jobs=1,
        verbose=2,
        random_state=42,
        return_train_score=True,
        error_score='raise'
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("Best params:", search.best_params_)
    print(f"Best CV MAE: {-search.best_score_:.4f}")

    return best_model, search.best_params_, -search.best_score_