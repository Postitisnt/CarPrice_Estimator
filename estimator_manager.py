from tensorflow import keras

class EstimatorManager:
    """
    This class is used to manage the pre-trained Neural Network models.
    
    Attributes
    ----------
    models : dict
        A dictionary containing the names of the models as keys and the paths
        to the saved models as values.
    
    Methods
    ----------
    add_model(model_name, path_to_saved_model_directory)
        Add a pre-trained NN model to the model manager.
    predict(model_name, data)
        Load a specific Neural Network pre-trained model and use it to make
        predictions on new data.
    """
    def __init__(self):
        self.models = {}

    def add_model(self, model_name, path_to_saved_model_directory):
        """
        This method is used to add a pre-trained NN model to the model manager.
        
        Parameters
        ----------
        model_name : str
            The name of the model to be added.
        path_to_saved_model_directory : str
            The path to the saved model directory.
        """
        self.models[model_name] = path_to_saved_model_directory

    def predict(self, model_name, data):
        """
        This method is used to load a specific Neural Network pre-trained model
        and use it to make predictions on new data.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.
        data : numpy.ndarray
            The data to be used for predictions.
        """
        path_to_model_dir = self.models.get(model_name)

        if path_to_model_dir is None:
            print(f'Model {model_name} not found.')
            return None
        else:
            estimator = keras.models.load_model(path_to_model_dir);
            predictions = estimator.predict(data);
            return predictions