import numpy
import joblib
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

def aggregate_random_forests(model_files):
    """
    Combines multiple Random Forest models into a single Global Ensemble.
    """
    print(f"--- Federated Random Forest Aggregation ---")
    
    models = []
    for file in model_files:
        try:
            m = joblib.load(file)
            normalized_weights = [normalize(w, norm='l2') for w in client_weights]
            global_weights = np.mean(normalized_weights, axis=0)
            print(f"Loaded: {file} with {len(m.estimators_)} trees.")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not models:
        return None

    # Step 1: Use the first model as a template for the Global Model
    global_rf = copy.deepcopy(models[0])
    
    # Step 2: Combine all trees (estimators) from all models
    all_trees = []
    for m in models:
        all_trees.extend(m.estimators_)
    
    # Step 3: Update the Global Model with the combined pool of trees
    global_rf.estimators_ = all_trees
    global_rf.n_estimators = len(all_trees)
    
    # Step 4: Ensure the global model knows it's detecting both classes
    global_rf.classes_ = models[0].classes_
    global_rf.n_classes_ = models[0].n_classes_

    print(f"\nGlobal Model Created with {global_rf.n_estimators} total trees.")
    return global_rf

if __name__ == "__main__":
    # Ensure these names match your Random Forest .pkl files
    rf_files = [
        'model_2_1.pkl', 
        'model_2__2.pkl', 
        'model_2__3.pkl'
    ]
    
    # Create the Global Forest
    global_forest = aggregate_random_forests(rf_files)
    
    if global_forest:
        # Save the combined model
        joblib.dump(global_forest, 'global_federated_rf_model.pkl')
        print("SUCCESS: 'global_federated_rf_model.pkl' is ready for deployment.")