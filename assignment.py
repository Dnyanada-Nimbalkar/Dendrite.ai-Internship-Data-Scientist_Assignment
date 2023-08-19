import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Load JSON configuration
with open("algoparams_from_ui.json", "r") as json_file:
    config = json.load(json_file)

# Load dataset
data = pd.read_csv("iris.csv") 
# Extract target and features
target_name = config['target']['target']
features = [feature for feature in data.columns if feature != target_name]

X = data[features]
y = data[target_name]

# Preprocess categorical features
categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    label_encoder = LabelEncoder()
    X.loc[:, col] = label_encoder.fit_transform(X[col])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# Feature reduction
reduction_method = config['feature_reduction']['feature_reduction_method']
num_features_to_keep = int(config['feature_reduction']['num_of_features_to_keep'])

if reduction_method == 'Tree-based':
    model = RandomForestRegressor(n_estimators=int(config['feature_reduction']['num_of_trees']),
                                   max_depth=int(config['feature_reduction']['depth_of_trees']))
    model.fit(X_scaled, y)
    feature_importances = model.feature_importances_
    selected_indices = feature_importances.argsort()[-num_features_to_keep:][::-1]
    X_reduced = X_scaled[:, selected_indices]
else:
    raise ValueError("Unsupported feature reduction method: {}".format(reduction_method))

# Model creation and hyperparameter tuning
models = []
for model_name, model_config in config['algorithms'].items():
    if model_config['is_selected']:
        if model_name == 'RandomForestRegressor':
            model = RandomForestRegressor()
            param_grid = {
                'n_estimators': range(model_config['min_trees'], model_config['max_trees'] + 1),
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 2, 4]
            }
            grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
            models.append((model_name, grid_search))

# Model fitting and evaluation
for model_name, model in models:
    model.fit(X_reduced, y)
    y_pred = model.predict(X_reduced)
    mse = mean_squared_error(y, y_pred)
    print("Model:", model_name)
    print("Best Parameters:", model.best_params_)
    print("Mean Squared Error:", mse)
    print("=" * 40)
