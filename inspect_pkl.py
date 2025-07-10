import pickle

# Load the model from the .pkl file
filename = 'heart-disease-prediction-knn-model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Print the model details
print("Model Details:")
print(model)

# If the model is a scikit-learn estimator, you can also inspect its attributes
if hasattr(model, 'get_params'):
    print("\nModel Parameters:")
    print(model.get_params())

