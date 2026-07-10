from sklearn.ensemble import RandomForestRegressor

class RandomForestPipeline:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        
    def train(self, X_train, y_train):
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        print("Predicting with Random Forest model...")
        return self.model.predict(X_test)
