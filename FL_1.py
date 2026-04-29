import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

def perform_federated_averaging(model_files):
    """
    Aggregates weights from multiple Logistic Regression pkl files.
    """
    weights_list = []
    intercepts_list = []
    
    print(f"--- Federated Learning Aggregation ---")
    
    for file in model_files:
        try:
            # Load the client model
            model = joblib.load(file)
            
            # Extract weights (coef_) and bias (intercept_)
            norm_coef = normalize(model.coef_, norm='l2')
            weights_list.append(norm_coef)
            intercepts_list.append(model.intercept_)
            
            print(f"Successfully loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not weights_list:
        print("No models were loaded. Aborting.")
        return None

    # --- STEP 1: Average the parameters ---
    # We use np.mean across the 'axis=0' to average each of the 29 features
    weights_list = [w.flatten() for w in weights_list]
    global_weights = np.mean(weights_list, axis=0).reshape(1, -1)
    global_intercept = np.mean(intercepts_list, axis=0)

    # --- STEP 2: Reconstruct the Global Model ---
    # We initialize a new model and manually inject the averaged parameters
    global_model = LogisticRegression(solver='saga')
    
    # Manually setting attributes (Requires setting classes_ as well)
    global_model.coef_ = global_weights
    global_model.intercept_ = global_intercept
    global_model.classes_ = np.array([0, 1]) # 0 for Benign, 1 for Attack
    
    # Optional: Verify shape (should be 1 x 29)
    print(f"\nGlobal Model Shape: {global_model.coef_.shape}")
    return global_model

if __name__ == "__main__":
    # Ensure these names match your saved files exactly
    files = [
        'model_1.pkl', 
        'model_2.pkl', 
        'model_3.pkl'
    ]
    
    # Run the aggregator
    federated_model = perform_federated_averaging(files)
    
    if federated_model:
        # Save the finalized "Global Brain"
        joblib.dump(federated_model, 'global_federated_model.pkl')
        print("\nSUCCESS: 'global_federated_model.pkl' created!")
        print("This model is now ready to detect DDoS, PortScan, and DoS attacks.")