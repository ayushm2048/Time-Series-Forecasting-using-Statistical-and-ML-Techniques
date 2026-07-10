from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SVRPipeline:
    def __init__(self, kernel='rbf', C=1, gamma=0.1, epsilon=0.01):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train(self, X_train, y_train):
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        print("Training SVR model...")
        self.model.fit(X_train_scaled, y_train_scaled)
        
    def predict(self, X_test):
        X_test_scaled = self.scaler_X.transform(X_test)
        print("Predicting with SVR model...")
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred
