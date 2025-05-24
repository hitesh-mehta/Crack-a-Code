"""
Restaurant Anomaly Detection Predictor
=====================================

This module provides a trained model for detecting anomalies in restaurant operations.
It can predict anomalies based on zone, time, occupancy, resource usage, and cleaning status.

Usage:
    from restaurant_anomaly_predictor import RestaurantAnomalyPredictor
    
    predictor = RestaurantAnomalyPredictor()
    predictor.load_model('restaurant_anomaly_model.pkl')
    
    result = predictor.predict('Kitchen01', 14, 25, 8.5, 16.2, 'pending')
    print(f"Prediction: {result['prediction']}")
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class RestaurantAnomalyPredictor:
    """
    A class for predicting anomalies in restaurant operations.
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = ['zone', 'hour', 'occupancy', 'power_use', 'water_use', 'cleaning_status']
        self.categorical_features = ['zone', 'cleaning_status']
        self.numeric_features = ['hour', 'occupancy', 'power_use', 'water_use']
        
    def train_model(self, csv_file_path, save_model=True, model_filename='restaurant_anomaly_model.pkl'):
        """
        Train the anomaly detection model on restaurant data.
        
        Args:
            csv_file_path (str): Path to the CSV file containing training data
            save_model (bool): Whether to save the trained model
            model_filename (str): Filename for saving the model
        
        Returns:
            dict: Training results including accuracy and classification report
        """
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(csv_file_path)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['anomaly']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_features)
        ])
        
        # Create model pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print("Training model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Anomaly rate in dataset: {y.mean():.2%}")
        
        # Save model if requested
        if save_model:
            self.save_model(model_filename)
            print(f"Model saved as {model_filename}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_predictions': y_pred,
            'test_actual': y_test
        }
    
    def save_model(self, filename='restaurant_anomaly_model.pkl'):
        """
        Save the trained model to a file.
        
        Args:
            filename (str): Name of the file to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved successfully as {filename}")
    
    def load_model(self, filename='restaurant_anomaly_model.pkl'):
        """
        Load a pre-trained model from a file.
        
        Args:
            filename (str): Name of the file containing the saved model
        """
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.categorical_features = model_data['categorical_features']
            self.numeric_features = model_data['numeric_features']
            self.is_trained = True
            print(f"Model loaded successfully from {filename}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {filename} not found. Please train a model first.")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, zone, hour, occupancy, power_use, water_use, cleaning_status):
        """
        Predict if a restaurant reading is anomalous.
        
        Args:
            zone (str): Zone identifier ('Store01', 'Dining01', 'Kitchen01', 'Hallway01')
            hour (int): Hour of day (0-23)
            occupancy (int): Number of people (0-100)
            power_use (float): Power usage (1-10)
            water_use (float): Water usage (1-20)
            cleaning_status (str): Cleaning status ('pending', 'done', 'inprogress')
        
        Returns:
            dict: Prediction results including prediction, probabilities, and risk level
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        # Validate inputs
        self._validate_inputs(zone, hour, occupancy, power_use, water_use, cleaning_status)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'zone': [zone],
            'hour': [hour],
            'occupancy': [occupancy],
            'power_use': [power_use],
            'water_use': [water_use],
            'cleaning_status': [cleaning_status]
        })
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        
        # Determine risk level
        anomaly_prob = probabilities[1]
        if anomaly_prob < 0.3:
            risk_level = "Low"
        elif anomaly_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'prediction': 'Anomaly' if prediction == 1 else 'Normal',
            'anomaly_probability': float(anomaly_prob),
            'normal_probability': float(probabilities[0]),
            'risk_level': risk_level,
            'input_data': {
                'zone': zone,
                'hour': hour,
                'occupancy': occupancy,
                'power_use': power_use,
                'water_use': water_use,
                'cleaning_status': cleaning_status
            }
        }
    
    def predict_batch(self, data_list):
        """
        Make predictions for multiple readings at once.
        
        Args:
            data_list (list): List of tuples/lists containing 
                            (zone, hour, occupancy, power_use, water_use, cleaning_status)
        
        Returns:
            list: List of prediction dictionaries
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        results = []
        for data in data_list:
            if len(data) != 6:
                raise ValueError("Each data entry must contain exactly 6 values")
            
            result = self.predict(*data)
            results.append(result)
        
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            dict: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature names after preprocessing
        preprocessor = self.model.named_steps['preprocessor']
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        all_feature_names = self.numeric_features + list(cat_feature_names)
        
        # Get importances
        importances = self.model.named_steps['classifier'].feature_importances_
        
        feature_importance = dict(zip(all_feature_names, importances))
        
        # Sort by importance
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_features
    
    def _validate_inputs(self, zone, hour, occupancy, power_use, water_use, cleaning_status):
        """Validate input parameters."""
        valid_zones = ['Store01', 'Dining01', 'Kitchen01', 'Hallway01']
        valid_cleaning_status = ['pending', 'done', 'inprogress']
        
        if zone not in valid_zones:
            raise ValueError(f"Zone must be one of: {valid_zones}")
        
        if not (0 <= hour <= 23):
            raise ValueError("Hour must be between 0 and 23")
        
        if not (0 <= occupancy <= 100):
            raise ValueError("Occupancy must be between 0 and 100")
        
        if not (1 <= power_use <= 10):
            raise ValueError("Power use must be between 1 and 10")
        
        if not (1 <= water_use <= 20):
            raise ValueError("Water use must be between 1 and 20")
        
        if cleaning_status not in valid_cleaning_status:
            raise ValueError(f"Cleaning status must be one of: {valid_cleaning_status}")
    
    def analyze_zone_patterns(self, csv_file_path):
        """
        Analyze anomaly patterns by zone.
        
        Args:
            csv_file_path (str): Path to the CSV file containing data
        
        Returns:
            dict: Analysis results by zone
        """
        df = pd.read_csv(csv_file_path)
        
        zone_analysis = {}
        for zone in df['zone'].unique():
            zone_data = df[df['zone'] == zone]
            zone_analysis[zone] = {
                'total_records': len(zone_data),
                'anomaly_count': zone_data['anomaly'].sum(),
                'anomaly_rate': zone_data['anomaly'].mean(),
                'avg_occupancy': zone_data['occupancy'].mean(),
                'avg_power_use': zone_data['power_use'].mean(),
                'avg_water_use': zone_data['water_use'].mean(),
                'most_common_cleaning_status': zone_data['cleaning_status'].mode()[0]
            }
        
        return zone_analysis
