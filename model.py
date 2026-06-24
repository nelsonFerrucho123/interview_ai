import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class InterviewModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        prob = self.model.predict_proba([features])[0][1]
        decision = "Avanza" if prob > 0.7 else "No avanza"
        return decision, prob

    def save(self, path="model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="model.pkl"):
        self.model = joblib.load(path)