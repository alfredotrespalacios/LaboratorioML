from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def obtener_regresor(nombre: str, random_state: int = 42):
    if nombre == "Árbol de decisión":
        return DecisionTreeRegressor(random_state=random_state)
    if nombre == "Random Forest":
        return RandomForestRegressor(n_estimators=150, random_state=random_state)
    if nombre == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=random_state)
    if nombre == "KNN Regressor":
        return KNeighborsRegressor(n_neighbors=5)
    if nombre == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("XGBoost no está instalado. Revise requirements.txt e instale xgboost.") from exc
        return XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    if nombre == "Support Vector Regression":
        return SVR(kernel="rbf")
    raise ValueError(f"Algoritmo no reconocido: {nombre}")

def obtener_clasificador(nombre: str, random_state: int = 42):
    if nombre == "Regresión logística":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(random_state=random_state)
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=150, random_state=random_state)
    if nombre == "KNN Classifier":
        return KNeighborsClassifier(n_neighbors=5)
    if nombre == "Support Vector Machine":
        return SVC(kernel="rbf", probability=True, random_state=random_state)
    if nombre == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Algoritmo no reconocido: {nombre}")

def obtener_red_neuronal(tipo: str, hidden_layer_sizes, activation: str, max_iter: int, learning_rate_init: float, random_state: int = 42):
    if tipo == "Regresión":
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            random_state=random_state,
        )
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        random_state=random_state,
    )
