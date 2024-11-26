import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer

# loading features
with open('features.pkl', 'rb') as file:
    X_train, X_val, y_train, y_val = pickle.load(file)

# loading model
with open('xgboost_model.pkl', 'rb') as file:
    model_xgb_final = pickle.load(file)

# retrain model
model_xgb_final.fit(X_train, y_train)

# evaluate the model on validation
val_auc = roc_auc_score(y_val, model_xgb_final.predict_proba(X_val)[:, 1])
print(f"Validation ROC-AUC for final model: {val_auc}")

# save final model
with open('final_xgboost_model.pkl', 'wb') as file:
    pickle.dump(model_xgb_final, file)

print("Final model trained and saved as 'final_xgboost_model.pkl'")


