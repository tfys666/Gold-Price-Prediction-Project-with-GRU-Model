# Gold Price Prediction Project with GRU Model

This project aims to forecast gold prices using a GRU (Gated Recurrent Unit) neural network through a comprehensive process involving data preprocessing, model development, hyperparameter tuning, and evaluation on a test set. Here's a detailed breakdown of the project steps:

## 1. Data Preparation & Preprocessing
- **Data Import**: The project initiates by reading gold price data from a file named "Gold.csv".
- **Normalization**: Utilizes `MinMaxScaler` to scale the gold settlement price series between 0 and 1 for effective neural network training.
- **Dataset Partitioning**: Splits the data into training (80%), validation (10%), and testing (10%) sets for model training and performance assessment.

## 2. Dataset Construction
- **Sequence Generation**: Defines a function `create_dataset` that, based on a specified `look_back` period, transforms historical price data into input-output pairs suitable for GRU model training.

## 3. Hyperparameter Optimization
- **Parameter Grid**: Sets up a grid of hyperparameters including `look_back`, number of GRU units, dropout rate, number of epochs, and batch size.
- **Grid Search**: Iterates over the parameter grid, creating and training models for each configuration, then selects the combination yielding the lowest Mean Squared Error (MSE) on the validation set.

## 4. Model Building & Training
- **Model Construction with Optimal Parameters**: Constructs the GRU model using the identified best hyperparameters, consisting of two GRU layers followed by an output layer.
- **Training Phase** Conducts training on the training set while monitoring performance on the validation set.

## 5. Testing & Evaluation
- **Prediction Generation**: Applies the trained optimal model to make predictions on the test set.
- **Inverse Transformation**: Restores both predicted values and actual test labels from their normalized state back to original price units.
- **Performance Metrics Calculation**: Computes test set metrics including MSE, RMSE, MAE, and MAPE to quantify prediction accuracy.
- **Visualization of Results**: Plots a comparison chart of actual versus predicted gold prices to visually assess the model's predictive power.

## Conclusion
The project demonstrates the application of a GRU-based deep learning model for time series forecasting, enhanced with hyperparameter optimization to refine the model's performance in predicting gold prices. The evaluation on the test set, along with visual and quantitative analyses, provides insights into the model's efficacy and potential real-world application.
