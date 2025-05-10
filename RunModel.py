import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ClinicalPredictionModel import ClinicalPredictionModel
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import sys
from datetime import datetime
sys.path.append('.')  # Add current directory to path

# ==========================================================================
# CONFIGURATION SETTINGS
# ==========================================================================
# Pipeline mode: 'train', 'continue_training', or 'predict'
MODE = 'train'  

# Data and output directories
DATA_DIR = 'data'  # Directory with CSV files from FHIR data
OUTPUT_DIR = '2025-4-21'  # Directory for saving output files

# Timeline generation settings
FORCE_NEW_TIMELINES = False  # Set to True to regenerate all patient timelines from scratch
APPEND_NEW_MONTHS = False     # Set to True to append new months of data for existing patients
TIMELINE_BATCH_SIZE = 100    # Number of patients to process in each batch

# Model settings
MODEL_VERSION = None  # Specific model version to use (None = latest)
FINE_TUNING_ENABLED = True  # Whether to use reduced learning rate for continued training

# ==========================================================================

def main():
#with open("2025-4-23 Run.txt", "w") as f:
    #sys.stdout = f
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('patient_data', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('model_metadata', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('performance_comparisons', exist_ok=True)
    

    
    # Initialize the predictor
    predictor = ClinicalPredictionModel(DATA_DIR)
    
    if MODE == 'train':
        # Run the ML pipeline for initial training
        print("\n======== INITIAL MODEL TRAINING ========")
        pipeline_results = predictor.run_pipeline(force_new_timelines=FORCE_NEW_TIMELINES)
        
        if not pipeline_results:
            print("Pipeline failed to complete successfully.")
            return
        
        models_results = pipeline_results['model_results']
        clinical_insights = pipeline_results['clinical_insights']
        timeline_dataFrame = pipeline_results['timeline_df']
        
        print("==Model Results==\n", models_results)
        print("==Clinical Insights==\n", clinical_insights)
        print("==Timeline DataFrame==\n", timeline_dataFrame)
        
        print("Initial training complete. Check the output directory for visualizations and reports.")
        
    elif MODE == 'continue_training':
        # Continue training with new data
        print("\n======== CONTINUING MODEL TRAINING ========")
        
        # Continue training existing models
        updated_models = predictor.continue_training(
            new_patient_ids=None,
            fine_tuning=FINE_TUNING_ENABLED,
            force_new=FORCE_NEW_TIMELINES,
            append_new_months=APPEND_NEW_MONTHS
        )
        
        if not updated_models:
            print("Incremental training failed.")
            return
            
        print("Continued training complete. Models have been updated.")
        
    elif MODE == 'predict':
        # Make predictions using saved models
        print("\n======== MAKING PREDICTIONS ========")
        
        # Load saved models
        loaded_models = predictor.load_saved_models()
        
        if not loaded_models:
            print("No saved models found for prediction.")
            return
        
        # Load data
        if not predictor.load_data():
            print("Failed to load data for prediction. Exiting.")
            return
            
        # Identify cerebral palsy patients
        cp_patient_ids = predictor.identify_cerebral_palsy_patients()
        
        # Determine primary care status
        primary_care_status = predictor.identify_primary_care_status()
        
        # Identify emergency visits (for later evaluation)
        emergency_visits = predictor.identify_emergency_visits()
        
        # Create patient timelines (will use existing ones if available)
        timeline_df = predictor.load_or_append_patient_timelines(
            cp_patient_ids, 
            primary_care_status, 
            emergency_visits,
            force_new=FORCE_NEW_TIMELINES,
            append_new_months=APPEND_NEW_MONTHS
        )
        
        if timeline_df.empty:
            print("No patient timeline data available for prediction.")
            return
            
        # Prepare data for prediction (just using the latest month for each patient)
        latest_records = timeline_df.groupby('PatientUID').apply(
            lambda x: x.sort_values('TimePoint', ascending=False).iloc[0]
        ).reset_index(drop=True)
        
        # Make predictions
        prediction_results = []
        
        for _, patient in latest_records.iterrows():
            patient_id = patient['PatientUID']
            has_primary_care = patient['HasPrimaryCare']
            interaction_freq = patient['InteractionCategory']
            
            care_group = 'primary_care' if has_primary_care else 'non_primary_care'
            
            # Try to find the specific model for this patient's profile
            if (care_group in loaded_models and 
                interaction_freq in loaded_models[care_group] and 
                'model' in loaded_models[care_group][interaction_freq]):
                
                model_info = loaded_models[care_group][interaction_freq]
                model = model_info['model']
                preprocessor = model_info['preprocessor']
                feature_names = model_info['feature_names']
                
                # Prepare features (match expected features)
                features = patient[feature_names].to_frame().T
                
                # Transform features
                try:
                    features_processed = preprocessor.transform(features)
                    
                    # Make prediction
                    risk_score = float(model.predict(features_processed)[0][0])
                    
                    # Add result
                    prediction_results.append({
                        'PatientUID': patient_id,
                        'Age': patient['Age'],
                        'Gender': patient['Gender'],
                        'HasPrimaryCare': has_primary_care,
                        'InteractionCategory': interaction_freq,
                        'DaysSincePrimaryCare': patient['DaysSincePrimaryCare'],
                        'EncounterCount': patient['EncounterCount'],
                        'EmergencyRiskScore': risk_score,
                        'RiskCategory': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
                        'ModelUsed': f"{care_group}_{interaction_freq}"
                    })
                    
                except Exception as e:
                    print(f"Error making prediction for patient {patient_id}: {e}")
            else:
                print(f"No suitable model found for patient {patient_id} (care_group={care_group}, freq={interaction_freq})")
        
        # Create dataframe with results
        if prediction_results:
            results_df = pd.DataFrame(prediction_results)
            
            # Save predictions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(OUTPUT_DIR, f'predictions_{timestamp}.csv')
            results_df.to_csv(results_path, index=False)
            
            # Create distribution visualization
            plt.figure(figsize=(12, 8))
            
            sns.countplot(x='RiskCategory', hue='HasPrimaryCare', data=results_df)
            plt.title('Patient Risk Distribution by Primary Care Status')
            plt.xlabel('Risk Category')
            plt.ylabel('Number of Patients')
            plt.savefig(os.path.join(OUTPUT_DIR, f'risk_distribution_{timestamp}.png'))
            plt.close()
            
            # List high-risk patients
            high_risk = results_df[results_df['RiskCategory'] == 'High'].sort_values('EmergencyRiskScore', ascending=False)
            
            if not high_risk.empty:
                print("\nHigh-risk patients identified:")
                for _, patient in high_risk.head(10).iterrows():
                    print(f"Patient {patient['PatientUID']}: {patient['EmergencyRiskScore']:.2f} risk score")
                    
                high_risk_path = os.path.join(OUTPUT_DIR, f'high_risk_patients_{timestamp}.csv')
                high_risk.to_csv(high_risk_path, index=False)
                print(f"Saved high-risk patients to {high_risk_path}")
            
            print(f"\nPredictions complete! Results saved to {results_path}")
        else:
            print("No predictions could be made. Check model compatibility with patient data.")
    else:
        print(f"Unknown mode: {MODE}. Please use 'train', 'continue_training', or 'predict'.")


if __name__ == "__main__":
    main()