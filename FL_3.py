import joblib
import numpy as np

class XGBoostFederatedEnsemble:
    def __init__(self, models):
        self.models = models
        print(f"Federated Ensemble initialized with {len(models)} XGBoost clients.")

    def predict(self, X):
        """
        Aggregate predictions by voting (majority class).
        """
        # Collect predictions from all models
        # Each model.predict returns an array of 0s and 1s
        preds = np.array([model.predict(X) for model in self.models])
        
        # Perform majority voting across the models (axis 0)
        # If 2 out of 3 models say '1', the result is '1'
        final_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=preds
        )
        return final_preds

    def predict_proba(self, X):
        """
        Aggregate predictions by averaging probabilities.
        """
        # Collect probability scores for the positive class (Attack)
        # Each model.predict_proba returns [[prob_0, prob_1], ...]
        probs = np.array([model.predict_proba(X) for model in self.models])
        
        # Average the probabilities across all 3 models
        avg_probs = np.mean(probs, axis=0)
        return avg_probs

def aggregate_xgb_models(model_files):
    """
    Loads XGBoost pkl files and creates a Federated Ensemble.
    """
    loaded_models = []
    print("--- Federated XGBoost Aggregation ---")
    
    for file in model_files:
        try:
            model = joblib.load(file)
            loaded_models.append(model)
            print(f"Successfully loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not loaded_models:
        return None

    # Create the global ensemble wrapper
    global_ensemble = XGBoostFederatedEnsemble(loaded_models)
    return global_ensemble

if __name__ == "__main__":
    # Ensure these names match your XGBoost .pkl files
    xgb_files = [
        'model_3_1.pkl', 
        'model_3_2.pkl', 
        'model_3_3.pkl'
    ]
    
    # Generate the Global Federated Model
    global_model = aggregate_xgb_models(xgb_files)
    
    if global_model:
        # Save the ensemble object
        joblib.dump(global_model, 'global_federated_xgb_model.pkl')
        print("\nSUCCESS: 'global_federated_xgb_model.pkl' created.")
        print("This ensemble uses collaborative intelligence to detect DDoS, PortScan, and DoS.")