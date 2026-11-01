import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Week_4_Hypothetical_Logistics_Dataset.csv")
num = ["distance_km","package_weight_kg","number_of_stops","warehouse_load","dispatch_hour"]
cat = ["transport_mode","weather_condition","traffic_level","region"]
X, y = df[num + cat], df["delivery_time_hours"]
prep = ColumnTransformer([
 ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num),
 ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)
])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)
models = {
 "Linear Regression": LinearRegression(),
 "Decision Tree": DecisionTreeRegressor(max_depth=8,min_samples_leaf=5,random_state=42),
 "Random Forest": RandomForestRegressor(n_estimators=160,max_depth=12,min_samples_leaf=3,random_state=42)
}
for name, estimator in models.items():
 model=Pipeline([("prep",prep),("model",estimator)]).fit(X_train,y_train)
 p=model.predict(X_test)
 print(name, "MAE",mean_absolute_error(y_test,p),"RMSE",mean_squared_error(y_test,p)**.5,"R2",r2_score(y_test,p))
search=GridSearchCV(
 Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=120,random_state=42))]),
 {"model__max_depth":[8,12,None],"model__min_samples_leaf":[2,4,6],"model__max_features":["sqrt",.8]},
 cv=KFold(5,shuffle=True,random_state=42),scoring="neg_mean_absolute_error")
search.fit(X_train,y_train)
p=search.best_estimator_.predict(X_test)
print("Best parameters:",search.best_params_)
print("Tuned MAE/RMSE/R2:",mean_absolute_error(y_test,p),mean_squared_error(y_test,p)**.5,r2_score(y_test,p))
