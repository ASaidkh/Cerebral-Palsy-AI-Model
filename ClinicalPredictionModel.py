import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import defaultdict
import os
from datetime import datetime, timedelta
import json
import keras_tuner as kt
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy
import shap

class ClinicalPredictionModel:
    def __init__(self, data_dir):
        """
        Initialize the enhanced prediction model.
        
        Args:
            data_dir: Directory containing CSV files exported from FHIR data
        """
        self.data_dir = data_dir
        self.patients_df = None
        self.encounters_df = None
        self.conditions_df = None
        self.observations_df = None
        self.medications_df = None
        self.procedures_df = None
        self.immunizations_df = None
        self.devices_df = None
        self.imaging_studies_df = None
        self.allergies_df = None
        self.careplans_df = None
        
        # Models dictionary to store different prediction models
        self.models = {
            'emergency_visit': None,
            'diagnosis': None,
            'medication': None,
            'procedure': None
        }
        
        # Preprocessors for each model
        self.preprocessors = {}
        
        # Labels and categories
        self.diagnosis_categories = []
        self.medication_categories = []
        self.procedure_categories = []
        
    def load_data(self):
        """Load all relevant dataframes from CSV files"""
        try:
            # Load core datasets
            self.patients_df = pd.read_csv(os.path.join(self.data_dir, 'patients.csv'))
            self.encounters_df = pd.read_csv(os.path.join(self.data_dir, 'encounters.csv'))
            self.conditions_df = pd.read_csv(os.path.join(self.data_dir, 'conditions.csv'))
            self.observations_df = pd.read_csv(os.path.join(self.data_dir, 'observations.csv'))
            self.medications_df = pd.read_csv(os.path.join(self.data_dir, 'medications.csv'))
            self.procedures_df = pd.read_csv(os.path.join(self.data_dir, 'procedures.csv'))
            
            # Load additional datasets
            self.immunizations_df = pd.read_csv(os.path.join(self.data_dir, 'immunizations.csv'))
            self.devices_df = pd.read_csv(os.path.join(self.data_dir, 'devices.csv'))
            
            # Load if available
            try:
                self.imaging_studies_df = pd.read_csv(os.path.join(self.data_dir, 'imaging_studies.csv'))
                self.allergies_df = pd.read_csv(os.path.join(self.data_dir, 'allergies.csv'))
                self.careplans_df = pd.read_csv(os.path.join(self.data_dir, 'careplans.csv'))
            except FileNotFoundError:
                # Not all datasets might be available
                pass
                
            # Print raw date examples before conversion
            print("\n=== DATE FORMATS BEFORE CONVERSION ===")
            if not self.patients_df.empty and 'BIRTHDATE' in self.patients_df.columns:
                print(f"PATIENT BIRTHDATE example: {self.patients_df['BIRTHDATE'].iloc[0]}")
            
            if not self.encounters_df.empty and 'START' in self.encounters_df.columns:
                print(f"ENCOUNTER START example: {self.encounters_df['START'].iloc[0]}")
            
            # Convert date columns to datetime with consistent timezone handling
            # Always use utc=True to properly handle timezone-aware dates and convert to UTC
            print("\n=== CONVERTING DATE COLUMNS TO TIMEZONE-NAIVE ===")
            
            # Define a helper function to convert dates to timezone-naive
            def convert_to_naive(df, date_columns):
                for col in date_columns:
                    if col in df.columns:
                        # First convert to datetime with timezone awareness
                        df[col] = pd.to_datetime(df[col], utc=True)
                        # Then strip timezone info to make naive
                        df[col] = df[col].dt.tz_localize(None)
                        if not df.empty:
                            print(f"Converted {col} to timezone-naive. Example: {df[col].iloc[0]}")
            
            # Convert all date columns to timezone-naive
            convert_to_naive(self.patients_df, ['BIRTHDATE'])
            convert_to_naive(self.encounters_df, ['START', 'STOP'])
            convert_to_naive(self.conditions_df, ['START', 'STOP'])
            convert_to_naive(self.observations_df, ['DATE'])
            convert_to_naive(self.medications_df, ['START', 'STOP'])
            convert_to_naive(self.procedures_df, ['START'])
            
            if hasattr(self, 'immunizations_df') and self.immunizations_df is not None:
                convert_to_naive(self.immunizations_df, ['DATE'])
            
            if hasattr(self, 'devices_df') and self.devices_df is not None:
                convert_to_naive(self.devices_df, ['START', 'STOP'])
            
            if hasattr(self, 'imaging_studies_df') and self.imaging_studies_df is not None:
                convert_to_naive(self.imaging_studies_df, ['DATE'])
            
            if hasattr(self, 'allergies_df') and self.allergies_df is not None:
                convert_to_naive(self.allergies_df, ['START', 'STOP'])
            
            if hasattr(self, 'careplans_df') and self.careplans_df is not None:
                convert_to_naive(self.careplans_df, ['START', 'STOP'])
            
            # Initialize target categories
            self._initialize_target_categories()
            
            print(f"Loaded {len(self.patients_df)} patients")
            print(f"Loaded {len(self.encounters_df)} encounters")
            print(f"Loaded {len(self.conditions_df)} conditions")
            print(f"Loaded {len(self.observations_df)} observations")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _initialize_target_categories(self):
        """Initialize target categories for multi-label prediction"""
        # For diagnosis prediction - identify top conditions, excluding findings and medication reviews
        if not self.conditions_df.empty:
            # Filter out conditions that contain "finding" or "Medication review"
            filtered_conditions = self.conditions_df[
                self.conditions_df['DESCRIPTION'].str.contains(
                    r"\(disorder\)", 
                    case=False, 
                    na=False, 
                    regex=True
                )
            ]
            
            # Get value counts of the filtered conditions
            condition_counts = filtered_conditions['DESCRIPTION'].value_counts().reset_index()
            condition_counts.columns = ['condition', 'count']
            
            # Take top 20 conditions or all if less than 20
            top_n = min(50, len(condition_counts))
            self.diagnosis_categories = condition_counts.head(top_n)['condition'].tolist()
            
            print(f"== Selected  {len(self.diagnosis_categories)} Diagnosis Categories: == \n {self.diagnosis_categories}")
        
        # For medication prediction - identify top medications
        if not self.medications_df.empty:
            medication_counts = self.medications_df['DESCRIPTION'].value_counts().reset_index()
            medication_counts.columns = ['medication', 'count']
            # Take top 20 medications or all if less than 20
            top_n = min(50, len(medication_counts))
            self.medication_categories = medication_counts.head(top_n)['medication'].tolist()
            print(f"== Selected  {len(self.medication_categories)} Medications Categories: ==\n {self.medication_categories}")
        
        # For procedure prediction - identify top procedures
        if not self.procedures_df.empty:
            procedure_counts = self.procedures_df['DESCRIPTION'].value_counts().reset_index()
            procedure_counts.columns = ['procedure', 'count']
            # Take top 20 procedures or all if less than 20
            top_n = min(50, len(procedure_counts))
            self.procedure_categories = procedure_counts.head(top_n)['procedure'].tolist()
            print(f"== Selected  {len(self.procedure_categories)} Procedure Categories: == \n {self.procedure_categories}")
    
    def identify_target_patients(self, condition_keywords=None):
        """
        Identify patients with specific conditions
        
        Args:
            condition_keywords: List of keywords to search for in condition descriptions
            
        Returns:
            List of patient IDs matching the criteria
        """
        if condition_keywords is None:
            # Default to cerebral palsy keywords
            condition_keywords = [
                'cerebral palsy', 'CP', 'spastic diplegia', 'spastic quadriplegia',
                'dyskinetic cerebral palsy', 'ataxic cerebral palsy'
            ]
        
        # Create a regex pattern for case-insensitive matching
        pattern = '|'.join(condition_keywords)
        
        # Filter conditions dataframe
        matching_conditions = self.conditions_df[
            self.conditions_df['DESCRIPTION'].str.lower().str.contains(pattern, na=False, case=False)
        ]
        
        # Get unique patient IDs
        patient_ids = matching_conditions['PATIENT'].unique().tolist()
        
        print(f"Identified {len(patient_ids)} patients with matching conditions")
        return patient_ids
    
    def extract_clinical_features(self, patient_id, start_date, end_date):
        """
        Extract comprehensive clinical features for a patient in a given time period
        
        Args:
            patient_id: Patient ID
            start_date: Beginning of the observation period
            end_date: End of the observation period
            
        Returns:
            Dictionary of clinical features
        """
        # Get relevant data for this patient and time period
        patient_observations = self.observations_df[
            (self.observations_df['PATIENT'] == patient_id) &
            (self.observations_df['DATE'] >= start_date) &
            (self.observations_df['DATE'] <= end_date)
        ]
        
        patient_medications = self.medications_df[
            (self.medications_df['PATIENT'] == patient_id) &
            (self.medications_df['START'] >= start_date) &
            (self.medications_df['START'] <= end_date)
        ]
        
        patient_conditions = self.conditions_df[
            (self.conditions_df['PATIENT'] == patient_id) &
            (self.conditions_df['START'] <= end_date) &
            ((self.conditions_df['STOP'].isna()) | (self.conditions_df['STOP'] >= start_date))
        ]
        
        patient_procedures = self.procedures_df[
            (self.procedures_df['PATIENT'] == patient_id) &
            (self.procedures_df['START'] >= start_date) &
            (self.procedures_df['START'] <= end_date)
        ]
        
        # Optional datasets
        patient_immunizations = pd.DataFrame()
        if hasattr(self, 'immunizations_df') and self.immunizations_df is not None:
            patient_immunizations = self.immunizations_df[
                (self.immunizations_df['PATIENT'] == patient_id) &
                (self.immunizations_df['DATE'] >= start_date) &
                (self.immunizations_df['DATE'] <= end_date)
            ]
        
        patient_imaging = pd.DataFrame()
        if hasattr(self, 'imaging_studies_df') and self.imaging_studies_df is not None:
            patient_imaging = self.imaging_studies_df[
                (self.imaging_studies_df['PATIENT'] == patient_id) &
                (self.imaging_studies_df['DATE'] >= start_date) &
                (self.imaging_studies_df['DATE'] <= end_date)
            ]
        
        patient_devices = pd.DataFrame()
        if hasattr(self, 'devices_df') and self.devices_df is not None:
            patient_devices = self.devices_df[
                (self.devices_df['PATIENT'] == patient_id) &
                (self.devices_df['START'] <= end_date) &
                ((self.devices_df['STOP'].isna()) | (self.devices_df['STOP'] >= start_date))
            ]
        
        patient_allergies = pd.DataFrame()
        if hasattr(self, 'allergies_df') and self.allergies_df is not None:
            patient_allergies = self.allergies_df[
                (self.allergies_df['PATIENT'] == patient_id) &
                (self.allergies_df['START'] <= end_date) &
                ((self.allergies_df['STOP'].isna()) | (self.allergies_df['STOP'] >= start_date))
            ]
        
        # Initialize features dictionary
        features = {}
        
        # Vital signs features
        vital_signs = {
            'BMI': None,
            'SYSTOLIC': None,
            'DIASTOLIC': None,
            'HEART_RATE': None,
            'RESPIRATORY_RATE': None,
            'TEMPERATURE': None,
            'OXYGEN_SATURATION': None,
            'WEIGHT': None,
            'HEIGHT': None,
            'PAIN_SCORE': None,
            'GLUCOSE': None
        }
        
        # Extract vital signs from observations
        for _, obs in patient_observations.iterrows():
            description = str(obs['DESCRIPTION']).lower() if pd.notna(obs['DESCRIPTION']) else ''
            value = obs['VALUE']
            
            if pd.notna(value):
                try:
                    value_float = float(value)
                    
                    # Match to vital signs categories
                    if 'bmi' in description:
                        vital_signs['BMI'] = value_float
                    elif 'systolic' in description:
                        vital_signs['SYSTOLIC'] = value_float
                    elif 'diastolic' in description:
                        vital_signs['DIASTOLIC'] = value_float
                    elif 'heart rate' in description or 'pulse' in description:
                        vital_signs['HEART_RATE'] = value_float
                    elif 'respiratory rate' in description:
                        vital_signs['RESPIRATORY_RATE'] = value_float
                    elif 'temperature' in description:
                        vital_signs['TEMPERATURE'] = value_float
                    elif 'oxygen' in description or 'o2' in description:
                        vital_signs['OXYGEN_SATURATION'] = value_float
                    elif 'weight' in description and 'kg' in description:
                        vital_signs['WEIGHT'] = value_float
                    elif 'height' in description or 'length' in description:
                        vital_signs['HEIGHT'] = value_float
                    elif 'pain' in description and 'score' in description:
                        vital_signs['PAIN_SCORE'] = value_float
                    elif 'glucose' in description or 'blood sugar' in description:
                        vital_signs['GLUCOSE'] = value_float
                except:
                    # If conversion fails, skip this observation
                    pass
        
        # Add vital signs to features
        for vital, value in vital_signs.items():
            features[f'VITAL_{vital}'] = value
        
        # Calculate derived vital sign metrics if available
        if vital_signs['SYSTOLIC'] is not None and vital_signs['DIASTOLIC'] is not None:
            # Mean Arterial Pressure (MAP)
            features['VITAL_MAP'] = (vital_signs['SYSTOLIC'] + 2 * vital_signs['DIASTOLIC']) / 3
            
            # Pulse Pressure
            features['VITAL_PULSE_PRESSURE'] = vital_signs['SYSTOLIC'] - vital_signs['DIASTOLIC']
        
        # BMI calculation if missing but weight and height available
        if vital_signs['BMI'] is None and vital_signs['WEIGHT'] is not None and vital_signs['HEIGHT'] is not None:
            # Height should be in meters, weight in kg
            height_m = vital_signs['HEIGHT'] / 100 if vital_signs['HEIGHT'] > 3 else vital_signs['HEIGHT']
            features['VITAL_BMI'] = vital_signs['WEIGHT'] / (height_m ** 2)
        
        # Extract medication categories and details
        medication_categories = defaultdict(int)
        medication_details = {}
        
        for _, med in patient_medications.iterrows():
            description = str(med['DESCRIPTION']).lower() if pd.notna(med['DESCRIPTION']) else ''
            
            # Track all medications (for frequency analysis)
            medication_details[description] = medication_details.get(description, 0) + 1
            
            # Categorize medications
            if any(term in description for term in ['antibiotic', 'amoxicillin', 'penicillin', 'clindamycin']):
                medication_categories['ANTIBIOTIC'] += 1
            elif any(term in description for term in ['analgesic', 'pain', 'acetaminophen', 'ibuprofen', 'opioid']):
                medication_categories['PAIN_RELIEF'] += 1
            elif any(term in description for term in ['anticonvulsant', 'seizure', 'epilep']):
                medication_categories['ANTICONVULSANT'] += 1
            elif any(term in description for term in ['muscle relaxant', 'spastic', 'baclofen', 'dantrolene']):
                medication_categories['MUSCLE_RELAXANT'] += 1
            elif any(term in description for term in ['steroid', 'corticosteroid', 'prednisone', 'methylprednisolone']):
                medication_categories['STEROID'] += 1
            elif any(term in description for term in ['psychiatric', 'antidepressant', 'anxiolytic', 'ssri']):
                medication_categories['PSYCHIATRIC'] += 1
            elif any(term in description for term in ['gastro', 'acid', 'reflux', 'gerd', 'proton pump']):
                medication_categories['GASTROINTESTINAL'] += 1
            elif any(term in description for term in ['respiratory', 'inhaler', 'broncho', 'albuterol']):
                medication_categories['RESPIRATORY'] += 1
            elif any(term in description for term in ['supplement', 'vitamin', 'mineral']):
                medication_categories['SUPPLEMENT'] += 1
            else:
                medication_categories['OTHER'] += 1
        
        # Add medication categories to features
        for category, count in medication_categories.items():
            features[f'MED_{category}'] = count
        
        # Total number of unique medications
        features['MED_UNIQUE_COUNT'] = len(medication_details)
        
        # Total medication count across all categories
        features['MED_TOTAL_COUNT'] = sum(medication_categories.values())
        
        # Extract condition categories
        condition_categories = defaultdict(int)
        current_conditions = set()
        
        for _, cond in patient_conditions.iterrows():
            description = str(cond['DESCRIPTION']).lower() if pd.notna(cond['DESCRIPTION']) else ''
            current_conditions.add(description)
            
            # Categorize conditions
            if any(term in description for term in ['infection', 'pneumonia', 'uti', 'sepsis']):
                condition_categories['INFECTION'] += 1
            elif any(term in description for term in ['seizure', 'epilep']):
                condition_categories['SEIZURE'] += 1
            elif any(term in description for term in ['pain', 'headache', 'migraine']):
                condition_categories['PAIN'] += 1
            elif any(term in description for term in ['gastro', 'constipation', 'reflux', 'gerd']):
                condition_categories['GI'] += 1
            elif any(term in description for term in ['respiratory', 'asthma', 'breathing', 'pneumonia']):
                condition_categories['RESPIRATORY'] += 1
            elif any(term in description for term in ['cardiac', 'heart', 'hypertension']):
                condition_categories['CARDIAC'] += 1
            elif any(term in description for term in ['psych', 'depression', 'anxiety', 'mental']):
                condition_categories['PSYCHIATRIC'] += 1
            elif any(term in description for term in ['neuro', 'brain', 'spinal']):
                condition_categories['NEUROLOGICAL'] += 1
            elif any(term in description for term in ['orthopedic', 'fracture', 'hip', 'bone']):
                condition_categories['ORTHOPEDIC'] += 1
            elif any(term in description for term in ['developmental', 'cognitive', 'intellectual']):
                condition_categories['DEVELOPMENTAL'] += 1
            elif any(term in description for term in ['urinary', 'bladder', 'kidney']):
                condition_categories['URINARY'] += 1
            else:
                condition_categories['OTHER'] += 1
        
        # Add condition categories to features
        for category, count in condition_categories.items():
            features[f'COND_{category}'] = count
        
        # Total number of unique conditions
        features['COND_UNIQUE_COUNT'] = len(current_conditions)
        
        # Total condition count across all categories
        features['COND_TOTAL_COUNT'] = sum(condition_categories.values())
        
        # Extract procedure categories
        procedure_categories = defaultdict(int)
        
        for _, proc in patient_procedures.iterrows():
            description = str(proc['DESCRIPTION']).lower() if pd.notna(proc['DESCRIPTION']) else ''
            
            # Categorize procedures
            if any(term in description for term in ['imaging', 'x-ray', 'mri', 'ct', 'ultrasound']):
                procedure_categories['IMAGING'] += 1
            elif any(term in description for term in ['surgery', 'operation', 'incision']):
                procedure_categories['SURGERY'] += 1
            elif any(term in description for term in ['therapy', 'physical therapy', 'pt', 'ot', 'speech']):
                procedure_categories['THERAPY'] += 1
            elif any(term in description for term in ['injection', 'botox', 'steroid']):
                procedure_categories['INJECTION'] += 1
            elif any(term in description for term in ['evaluation', 'assessment']):
                procedure_categories['EVALUATION'] += 1
            elif any(term in description for term in ['lab', 'blood', 'test']):
                procedure_categories['LABORATORY'] += 1
            else:
                procedure_categories['OTHER'] += 1
        
        # Add procedure categories to features
        for category, count in procedure_categories.items():
            features[f'PROC_{category}'] = count
        
        # Total procedure count
        features['PROC_TOTAL_COUNT'] = sum(procedure_categories.values())
        
        # Extract imaging study features if available
        if not patient_imaging.empty:
            imaging_categories = defaultdict(int)
            
            for _, img in patient_imaging.iterrows():
                modality = str(img['MODALITY_DESCRIPTION']).lower() if pd.notna(img.get('MODALITY_DESCRIPTION')) else ''
                bodysite = str(img['BODYSITE_DESCRIPTION']).lower() if pd.notna(img.get('BODYSITE_DESCRIPTION')) else ''
                
                # Count by modality
                if 'xray' in modality or 'x-ray' in modality:
                    imaging_categories['XRAY'] += 1
                elif 'mri' in modality:
                    imaging_categories['MRI'] += 1
                elif 'ct' in modality:
                    imaging_categories['CT'] += 1
                elif 'ultrasound' in modality:
                    imaging_categories['ULTRASOUND'] += 1
                else:
                    imaging_categories['OTHER'] += 1
                
                # Count body sites
                if 'brain' in bodysite or 'head' in bodysite:
                    imaging_categories['BRAIN'] += 1
                elif 'chest' in bodysite or 'lung' in bodysite:
                    imaging_categories['CHEST'] += 1
                elif 'abdomen' in bodysite:
                    imaging_categories['ABDOMEN'] += 1
                elif 'hip' in bodysite or 'pelvis' in bodysite:
                    imaging_categories['HIP'] += 1
                elif 'spine' in bodysite:
                    imaging_categories['SPINE'] += 1
                else:
                    imaging_categories['OTHER_SITE'] += 1
                    
            # Add imaging categories to features
            for category, count in imaging_categories.items():
                features[f'IMG_{category}'] = count
                
            # Total imaging count
            features['IMG_TOTAL_COUNT'] = sum(imaging_categories.values())
        else:
            # If no imaging studies, add zeros
            features['IMG_TOTAL_COUNT'] = 0
            for category in ['XRAY', 'MRI', 'CT', 'ULTRASOUND', 'OTHER', 'BRAIN', 'CHEST', 'ABDOMEN', 'HIP', 'SPINE', 'OTHER_SITE']:
                features[f'IMG_{category}'] = 0
        
        # Extract device features if available
        if not patient_devices.empty:
            device_categories = defaultdict(int)
            
            for _, dev in patient_devices.iterrows():
                description = str(dev['DESCRIPTION']).lower() if pd.notna(dev['DESCRIPTION']) else ''
                
                # Categorize devices
                if any(term in description for term in ['wheelchair', 'mobility']):
                    device_categories['MOBILITY'] += 1
                elif any(term in description for term in ['feeding', 'tube', 'gastrostomy', 'g-tube']):
                    device_categories['FEEDING'] += 1
                elif any(term in description for term in ['ventilator', 'oxygen', 'breathing']):
                    device_categories['RESPIRATORY'] += 1
                elif any(term in description for term in ['monitor', 'telemetry']):
                    device_categories['MONITORING'] += 1
                elif any(term in description for term in ['orthotic', 'brace', 'splint']):
                    device_categories['ORTHOTIC'] += 1
                elif any(term in description for term in ['catheter', 'urinary']):
                    device_categories['URINARY'] += 1
                else:
                    device_categories['OTHER'] += 1
            
            # Add device categories to features
            for category, count in device_categories.items():
                features[f'DEV_{category}'] = count
                
            # Total device count
            features['DEV_TOTAL_COUNT'] = sum(device_categories.values())
        else:
            # If no devices, add zeros
            features['DEV_TOTAL_COUNT'] = 0
            for category in ['MOBILITY', 'FEEDING', 'RESPIRATORY', 'MONITORING', 'ORTHOTIC', 'URINARY', 'OTHER']:
                features[f'DEV_{category}'] = 0
        
        # Extract allergy features if available
        if not patient_allergies.empty:
            allergy_categories = defaultdict(int)
            
            for _, allergy in patient_allergies.iterrows():
                description = str(allergy['DESCRIPTION']).lower() if pd.notna(allergy['DESCRIPTION']) else ''
                
                # Categorize allergies
                if any(term in description for term in ['penicillin', 'antibiotic']):
                    allergy_categories['ANTIBIOTIC'] += 1
                elif any(term in description for term in ['food', 'peanut', 'nut', 'dairy', 'egg']):
                    allergy_categories['FOOD'] += 1
                elif any(term in description for term in ['environmental', 'pollen', 'dust', 'mold']):
                    allergy_categories['ENVIRONMENTAL'] += 1
                elif any(term in description for term in ['latex']):
                    allergy_categories['LATEX'] += 1
                elif any(term in description for term in ['medication', 'drug']):
                    allergy_categories['MEDICATION'] += 1
                else:
                    allergy_categories['OTHER'] += 1
            
            # Add allergy categories to features
            for category, count in allergy_categories.items():
                features[f'ALLERGY_{category}'] = count
                
            # Total allergy count
            features['ALLERGY_TOTAL_COUNT'] = sum(allergy_categories.values())
        else:
            # If no allergies, add zeros
            features['ALLERGY_TOTAL_COUNT'] = 0
            for category in ['ANTIBIOTIC', 'FOOD', 'ENVIRONMENTAL', 'LATEX', 'MEDICATION', 'OTHER']:
                features[f'ALLERGY_{category}'] = 0
        
        # Extract immunization features if available
        if not patient_immunizations.empty:
            # Count total immunizations in this period
            features['IMMUNIZATION_COUNT'] = len(patient_immunizations)
            
            # Check for specific immunizations
            immunization_types = defaultdict(int)
            for _, imm in patient_immunizations.iterrows():
                description = str(imm['DESCRIPTION']).lower() if pd.notna(imm['DESCRIPTION']) else ''
                
                # Categorize immunizations
                if 'influenza' in description or 'flu' in description:
                    immunization_types['FLU'] += 1
                elif 'pneumo' in description:
                    immunization_types['PNEUMOCOCCAL'] += 1
                elif 'tetanus' in description or 'tdap' in description:
                    immunization_types['TETANUS'] += 1
                else:
                    immunization_types['OTHER'] += 1
            
            # Add immunization types to features
            for category, count in immunization_types.items():
                features[f'IMM_{category}'] = count
        else:
            # If no immunizations, add zeros
            features['IMMUNIZATION_COUNT'] = 0
            for category in ['FLU', 'PNEUMOCOCCAL', 'TETANUS', 'OTHER']:
                features[f'IMM_{category}'] = 0

        print(f"    Found {len(patient_observations)} observations, {len(patient_medications)} medications, {len(patient_conditions)} conditions, {len(patient_procedures)} procedures")
        
        return features
    
    def create_patient_timelines(self, patient_ids, output_dir='patient_data', lookback_months=12, 
                            batch_size=100, resume=False):
        """
        Create comprehensive patient timelines with enhanced clinical features
        
        Args:
            patient_ids: List of patient IDs to process
            output_dir: Directory to save timeline data
            lookback_months: Number of months to look back for each time point
            batch_size: Number of patients to process before saving
            resume: Whether to try to resume from previous run
            
        Returns:
            Tuple of (DataFrame with patient timeline features, targets dictionary)
        """
        os.makedirs(output_dir, exist_ok=True)
        all_patient_records = []
        
        # Target outcomes to predict
        target_outcomes = {
            'emergency_visit': [],  # Emergency visit in next month
            'diagnoses': [],        # New diagnoses in next month
            'medications': [],      # New medications in next month
            'procedures': []        # New procedures in next month
        }
        
        # Check for resuming previous run
        processed_patients = []
        last_patient_id = None
        
        if resume:
            processed_patients, last_patient_id = self._load_generation_state(output_dir)
            
            # Find most recent batch file
            batch_files = [f for f in os.listdir(output_dir) if f.startswith('enhanced_patient_timelines_batch_')]
            if batch_files:
                batch_files.sort(reverse=True)
                last_batch_file = os.path.join(output_dir, batch_files[0])
                
                # Load previous batch data
                print(f"Loading previous batch from {last_batch_file}")
                previous_batch = pd.read_csv(last_batch_file, low_memory=False)
                
                # Convert date columns
                previous_batch['TIME_POINT'] = pd.to_datetime(previous_batch['TIME_POINT'], format='mixed', errors='coerce')
                
                # Extract data for previous records
                for _, row in previous_batch.iterrows():
                    # Extract target lists from string representation
                    diagnoses = eval(row['NEW_DIAGNOSES_NEXT_MONTH']) if pd.notna(row['NEW_DIAGNOSES_NEXT_MONTH']) else []
                    medications = eval(row['NEW_MEDICATIONS_NEXT_MONTH']) if pd.notna(row['NEW_MEDICATIONS_NEXT_MONTH']) else []
                    procedures = eval(row['NEW_PROCEDURES_NEXT_MONTH']) if pd.notna(row['NEW_PROCEDURES_NEXT_MONTH']) else []
                    
                    # Add to all records
                    all_patient_records.append(row.to_dict())
                    
                    # Add to target outcomes
                    target_outcomes['emergency_visit'].append(row['HAD_EMERGENCY_NEXT_MONTH'])
                    
                    # Convert multi-hot encoded targets
                    diagnosis_target = [1 if dx in diagnoses else 0 for dx in self.diagnosis_categories]
                    target_outcomes['diagnoses'].append(diagnosis_target)
                    
                    medication_target = [1 if med in medications else 0 for med in self.medication_categories]
                    target_outcomes['medications'].append(medication_target)
                    
                    procedure_target = [1 if proc in procedures else 0 for proc in self.procedure_categories]
                    target_outcomes['procedures'].append(procedure_target)
        
        # Filter remaining patients
        remaining_patients = [pid for pid in patient_ids if pid not in processed_patients]
        print(f"Processing {len(remaining_patients)} remaining patients out of {len(patient_ids)} total")
        
        # Process each patient
        for i, patient_id in enumerate(remaining_patients):
            try:
                print(f"\nProcessing patient: {patient_id} ({i+1}/{len(remaining_patients)})")
                
                # Get patient demographics
                patient_info = self.patients_df[self.patients_df['Id'] == patient_id].iloc[0]
                
                # Determine the date range for this patient's records
                patient_encounters = self.encounters_df[self.encounters_df['PATIENT'] == patient_id]
                if patient_encounters.empty:
                    print(f"  No encounters found for patient {patient_id}, skipping")
                    continue  # Skip patients with no encounters
                
                # Get date range for patient
                start_date = patient_encounters['START'].min()
                end_date = patient_encounters['START'].max()
                
                print(f"  Patient date range: {start_date} to {end_date}")
                print(f"  Start date type: {type(start_date)}, tzinfo: {getattr(start_date, 'tzinfo', None)}")
                print(f"  End date type: {type(end_date)}, tzinfo: {getattr(end_date, 'tzinfo', None)}")
                
                # Create monthly timeline records
                # Make sure to create timezone-naive Timestamp objects
                current_date = pd.Timestamp(start_date.year, start_date.month, 1)
                end_of_month = pd.Timestamp(end_date.year, end_date.month, end_date.day)
                
                print(f"  Current date: {current_date}, tzinfo: {getattr(current_date, 'tzinfo', None)}")
                print(f"  End of month: {end_of_month}, tzinfo: {getattr(end_of_month, 'tzinfo', None)}")
                
                month_count = 0
                while current_date <= end_of_month:
                    month_count += 1
                    print(f"  Processing month {month_count}: {current_date}")
                    
                    # All date calculations should now produce timezone-naive timestamps
                    next_month = current_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                    lookback_start = current_date - pd.DateOffset(months=lookback_months)
                    prediction_start = next_month + pd.DateOffset(days=1)
                    prediction_end = prediction_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                    
                    print(f"    Next month end: {next_month}")
                    print(f"    Lookback start: {lookback_start}")
                    print(f"    Prediction period: {prediction_start} to {prediction_end}")
                    
                    # Verify all timestamps are timezone-naive for debugging
                    all_naive = all(getattr(ts, 'tzinfo', None) is None 
                                for ts in [current_date, next_month, lookback_start, 
                                            prediction_start, prediction_end])
                    print(f"    All timestamps are timezone-naive: {all_naive}")
                    
                    # Check for emergency visit in the next month
                    emergency_next_month = False
                    next_month_encounters = self.encounters_df[
                        (self.encounters_df['PATIENT'] == patient_id) &
                        (self.encounters_df['START'] >= prediction_start) &
                        (self.encounters_df['START'] <= prediction_end)
                    ]
                    
                    print(f"    Found {len(next_month_encounters)} encounters in next month")
                    
                    if not next_month_encounters.empty:
                        # Look for emergency department visits
                        for _, enc in next_month_encounters.iterrows():
                            description = str(enc['DESCRIPTION']).lower() if pd.notna(enc['DESCRIPTION']) else ''
                            encounter_class = str(enc['ENCOUNTERCLASS']).lower() if pd.notna(enc['ENCOUNTERCLASS']) else ''
                            
                            if ('emergency' in description or 'ed' in description or 
                                'emergency' in encounter_class or 'urgent' in encounter_class):
                                emergency_next_month = True
                                print(f"    Emergency visit found in next month")
                                break
                    
                    # Identify new diagnoses in the next month
                    new_diagnoses = []
                    next_month_conditions = self.conditions_df[
                        (self.conditions_df['PATIENT'] == patient_id) &
                        (self.conditions_df['START'] >= prediction_start) &
                        (self.conditions_df['START'] <= prediction_end)
                    ]
                    
                    if not next_month_conditions.empty:
                        new_diagnoses = next_month_conditions['DESCRIPTION'].tolist()
                        print(f"    Found {len(new_diagnoses)} new diagnoses in next month")
                    
                    # Identify new medications in the next month
                    new_medications = []
                    next_month_medications = self.medications_df[
                        (self.medications_df['PATIENT'] == patient_id) &
                        (self.medications_df['START'] >= prediction_start) &
                        (self.medications_df['START'] <= prediction_end)
                    ]
                    
                    if not next_month_medications.empty:
                        new_medications = next_month_medications['DESCRIPTION'].tolist()
                        print(f"    Found {len(new_medications)} new medications in next month")
                    
                    # Identify new procedures in the next month
                    new_procedures = []
                    next_month_procedures = self.procedures_df[
                        (self.procedures_df['PATIENT'] == patient_id) &
                        (self.procedures_df['START'] >= prediction_start) &
                        (self.procedures_df['START'] <= prediction_end)
                    ]
                    
                    if not next_month_procedures.empty:
                        new_procedures = next_month_procedures['DESCRIPTION'].tolist()
                        print(f"    Found {len(new_procedures)} new procedures in next month")
                    
                    # Extract comprehensive clinical features
                    print("    Extracting clinical features...")
                    clinical_features = self.extract_clinical_features(
                        patient_id, lookback_start, next_month
                    )
                    
                    # Create demographic features
                    demographic_features = {}
                    
                    # Calculate age at this time point
                    if pd.notna(patient_info['BIRTHDATE']):
                        # Ensure birthdate is timezone-naive for comparison
                        birthdate = patient_info['BIRTHDATE']
                        age_days = (current_date - birthdate).days
                        demographic_features['AGE_YEARS'] = age_days / 365.25
                    else:
                        demographic_features['AGE_YEARS'] = None
                    
                    # Gender
                    demographic_features['GENDER'] = patient_info['GENDER'] if pd.notna(patient_info['GENDER']) else 'Unknown'
                    
                    # Race & Ethnicity
                    demographic_features['RACE'] = patient_info['RACE'] if pd.notna(patient_info['RACE']) else 'Unknown'
                    demographic_features['ETHNICITY'] = patient_info['ETHNICITY'] if pd.notna(patient_info['ETHNICITY']) else 'Unknown'
                    
                    # Calculate healthcare utilization features
                    utilization_features = self.calculate_utilization_metrics(patient_id, lookback_start, next_month)
                    
                    # Combine all features
                    record = {
                        'PATIENT_ID': patient_id,
                        'TIME_POINT': current_date,
                        # Add demographic features
                        **{f'DEM_{k}': v for k, v in demographic_features.items()},
                        # Add clinical features
                        **clinical_features,
                        # Add utilization features
                        **utilization_features,
                        # Target outcomes
                        'HAD_EMERGENCY_NEXT_MONTH': emergency_next_month,
                        'NEW_DIAGNOSES_NEXT_MONTH': new_diagnoses,
                        'NEW_MEDICATIONS_NEXT_MONTH': new_medications,
                        'NEW_PROCEDURES_NEXT_MONTH': new_procedures
                    }
                    
                    all_patient_records.append(record)
                    
                    # Prepare multi-label targets for classification
                    target_outcomes['emergency_visit'].append(emergency_next_month)
                    
                    # For diagnoses - create multi-hot encoding
                    diagnosis_target = [1 if dx in new_diagnoses else 0 for dx in self.diagnosis_categories]
                    target_outcomes['diagnoses'].append(diagnosis_target)
                    
                    # For medications - create multi-hot encoding
                    medication_target = [1 if med in new_medications else 0 for med in self.medication_categories]
                    target_outcomes['medications'].append(medication_target)
                    
                    # For procedures - create multi-hot encoding
                    procedure_target = [1 if proc in new_procedures else 0 for proc in self.procedure_categories]
                    target_outcomes['procedures'].append(procedure_target)
                    
                    # Move to next month
                    current_date = current_date + pd.DateOffset(months=1)
                    print(f"    Moving to next month: {current_date}")
                
                # Save state and batch data periodically
                processed_patients.append(patient_id)
                last_patient_id = patient_id
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(remaining_patients)} patients")
                
                # Save batch if we've reached batch_size
                if (i + 1) % batch_size == 0 or (i + 1) == len(remaining_patients):
                    # Save current batch
                    batch_df = pd.DataFrame(all_patient_records)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    batch_file = os.path.join(output_dir, f'enhanced_patient_timelines_batch_{i+1}_{timestamp}.csv')
                    batch_df.to_csv(batch_file, index=False)
                    
                    print(f"Saved batch with {len(batch_df)} records to {batch_file}")
                    
                    # Save generation state
                    self._save_generation_state(processed_patients, last_patient_id, output_dir)
                    
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create dataframe from all records
        timeline_df = pd.DataFrame(all_patient_records)
        
        # Add target arrays as separate objects
        targets = {
            'emergency_visit': np.array(target_outcomes['emergency_visit']) if target_outcomes['emergency_visit'] else np.array([]),
            'diagnoses': np.array(target_outcomes['diagnoses']) if target_outcomes['diagnoses'] else np.array([]),
            'medications': np.array(target_outcomes['medications']) if target_outcomes['medications'] else np.array([]),
            'procedures': np.array(target_outcomes['procedures']) if target_outcomes['procedures'] else np.array([])
        }
        
        # Save the final complete timeline
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timeline_path = os.path.join(output_dir, f'enhanced_patient_timelines_complete_{timestamp}.csv')
        timeline_df.to_csv(timeline_path, index=False)
        
        print(f"Created timeline with {len(timeline_df)} records for {len(processed_patients)} patients")
        
        return timeline_df, targets
    
    def _save_generation_state(self, processed_patients, last_patient_id, output_dir):
        """
        Save the current state of timeline generation to resume later
        
        Args:
            processed_patients: List of patient IDs that have been processed
            last_patient_id: ID of the last patient processed
            output_dir: Directory to save the state file
        """
        import os
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create state object
        state = {
            'timestamp': datetime.now().isoformat(),
            'processed_patients': processed_patients,
            'last_patient_id': last_patient_id,
            'total_processed': len(processed_patients)
        }
        
        # Save state to JSON file
        state_file = os.path.join(output_dir, 'enhanced_timeline_generation_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved generation state: {len(processed_patients)} patients processed, last patient: {last_patient_id}")
    
    def _load_generation_state(self, output_dir):
        """
        Load the previous state of timeline generation
        
        Args:
            output_dir: Directory where the state file is stored
            
        Returns:
            Tuple of (processed_patients, last_patient_id) or ([], None) if no state file exists
        """
        import os
        import json
        
        state_file = os.path.join(output_dir, 'enhanced_timeline_generation_state.json')
        
        if not os.path.exists(state_file):
            return [], None
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            processed_patients = state.get('processed_patients', [])
            last_patient_id = state.get('last_patient_id')
            
            print(f"Loaded generation state: {len(processed_patients)} patients already processed")
            print(f"Resuming after patient {last_patient_id}")
            
            return processed_patients, last_patient_id
            
        except Exception as e:
            print(f"Error loading generation state: {e}")
            return [], None
            
    def load_or_append_patient_timelines(self, patient_ids, output_dir='patient_data', 
                                       force_new=False, append_new_months=True):
        """
        Load existing timeline data or append new data if needed
        
        Args:
            patient_ids: List of patient IDs to process
            output_dir: Directory for timeline data
            force_new: Whether to force new timeline creation
            append_new_months: Whether to append new months to existing patients
            
        Returns:
            Tuple of (DataFrame with patient timeline features, targets dictionary)
        """
        import os
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # Check for existing timeline data
        existing_df = None
        if os.path.exists(output_dir) and not force_new:
            timeline_files = [f for f in os.listdir(output_dir) if f.startswith('enhanced_patient_timelines_')]
            if timeline_files:
                # Sort by timestamp to get most recent
                timeline_files.sort(reverse=True)
                latest_file = os.path.join(output_dir, timeline_files[0])
                
                print(f"Loading existing timeline data from {latest_file}")
                existing_df = pd.read_csv(latest_file, low_memory=False)
                
                # Convert date columns back to datetime
                existing_df['TIME_POINT'] = pd.to_datetime(existing_df['TIME_POINT'], format='mixed', errors='coerce')
                
                print(f"Loaded {len(existing_df)} existing timeline records for {existing_df['PATIENT_ID'].nunique()} patients")
        
        if force_new or existing_df is None or existing_df.empty:
            print("Creating new timelines from scratch")
            return self.create_patient_timelines(patient_ids, output_dir)
        
        # Identify which patients need processing
        existing_patients = set(existing_df['PATIENT_ID'].unique())
        new_patients = [pid for pid in patient_ids if pid not in existing_patients]
        update_patients = [pid for pid in patient_ids if pid in existing_patients] if append_new_months else []
        
        print(f"Found {len(new_patients)} new patients to process")
        print(f"Found {len(update_patients)} existing patients to update with new data")
        

        if not new_patients and not update_patients:
            print("No new data to process. Using existing timeline.")
            # Recreate targets from existing data
            targets = self._extract_targets_from_existing_timeline(existing_df)
            return existing_df, targets
        
        # Process new patients and/or update existing ones
        timeline_df, new_targets = self._process_additional_patients(
            new_patients, update_patients, existing_df, output_dir, append_new_months
        )
        
        return timeline_df, new_targets
    
    def _extract_targets_from_existing_timeline(self, timeline_df):
        """
        Extract target arrays from an existing timeline DataFrame
        
        Args:
            timeline_df: DataFrame containing patient timeline data
            
        Returns:
            Dictionary with target arrays
        """
        import numpy as np
        
        # Initialize target dictionaries
        target_outcomes = {
            'emergency_visit': [],
            'diagnoses': [],
            'medications': [],
            'procedures': []
        }
        
        # Process each row
        for _, row in timeline_df.iterrows():
            # Emergency visit target (binary)
            # Convert string 'True'/'False' to boolean True/False, handling 'nan' as False
            emergency_val = row['HAD_EMERGENCY_NEXT_MONTH']
            if isinstance(emergency_val, str):
                emergency_val = (emergency_val.lower() == 'true')
            else:
                emergency_val = bool(emergency_val) if pd.notna(emergency_val) else False
                
            target_outcomes['emergency_visit'].append(emergency_val)
            
            # Rest of the method remains the same...
            try:
                diagnoses = eval(row['NEW_DIAGNOSES_NEXT_MONTH']) if pd.notna(row['NEW_DIAGNOSES_NEXT_MONTH']) else []
                medications = eval(row['NEW_MEDICATIONS_NEXT_MONTH']) if pd.notna(row['NEW_MEDICATIONS_NEXT_MONTH']) else []
                procedures = eval(row['NEW_PROCEDURES_NEXT_MONTH']) if pd.notna(row['NEW_PROCEDURES_NEXT_MONTH']) else []
                
                # Create multi-hot vectors
                diagnosis_target = [1 if dx in diagnoses else 0 for dx in self.diagnosis_categories]
                target_outcomes['diagnoses'].append(diagnosis_target)
                
                medication_target = [1 if med in medications else 0 for med in self.medication_categories]
                target_outcomes['medications'].append(medication_target)
                
                procedure_target = [1 if proc in procedures else 0 for proc in self.procedure_categories]
                target_outcomes['procedures'].append(procedure_target)
            except:
                # If there's an error, add zero vectors
                target_outcomes['diagnoses'].append([0] * len(self.diagnosis_categories))
                target_outcomes['medications'].append([0] * len(self.medication_categories))
                target_outcomes['procedures'].append([0] * len(self.procedure_categories))
        
        # Convert to numpy arrays
        targets = {
            'emergency_visit': np.array(target_outcomes['emergency_visit'], dtype=bool),  # Ensure boolean type
            'diagnoses': np.array(target_outcomes['diagnoses']) if target_outcomes['diagnoses'] else np.array([]),
            'medications': np.array(target_outcomes['medications']) if target_outcomes['medications'] else np.array([]),
            'procedures': np.array(target_outcomes['procedures']) if target_outcomes['procedures'] else np.array([])
        }
        
        return targets
    
    def calculate_utilization_metrics(self, patient_id, start_date, end_date):
        """
        Calculate healthcare utilization metrics for a patient in a given time period
        
        Args:
            patient_id: Patient ID
            start_date: Beginning of the observation period
            end_date: End of the observation period
            
        Returns:
            Dictionary of utilization metrics
        """
        # Get patient's encounters within the time period
        patient_encounters = self.encounters_df[
            (self.encounters_df['PATIENT'] == patient_id) &
            (self.encounters_df['START'] >= start_date) &
            (self.encounters_df['START'] <= end_date)
        ]
        
        # Initialize metrics
        metrics = {
            'TOTAL_ENCOUNTERS': 0,
            'INPATIENT_STAYS': 0,
            'EMERGENCY_VISITS': 0,
            'OUTPATIENT_VISITS': 0,
            'DAYS_SINCE_LAST_ENCOUNTER': 999,
            'AVERAGE_ENCOUNTER_DURATION_DAYS': 0,
            'PRIMARY_CARE_VISITS': 0,
            'SPECIALIST_VISITS': 0,
            'TOTAL_COST': 0
        }
        
        if not patient_encounters.empty:
            # Total encounters
            metrics['TOTAL_ENCOUNTERS'] = len(patient_encounters)
            
            # Count by encounter type
            for _, enc in patient_encounters.iterrows():
                encounter_class = str(enc['ENCOUNTERCLASS']).lower() if pd.notna(enc['ENCOUNTERCLASS']) else ''
                description = str(enc['DESCRIPTION']).lower() if pd.notna(enc['DESCRIPTION']) else ''
                
                # Categorize encounter
                if 'inpatient' in encounter_class or 'inpatient' in description:
                    metrics['INPATIENT_STAYS'] += 1
                elif 'emergency' in encounter_class or 'emergency' in description:
                    metrics['EMERGENCY_VISITS'] += 1
                elif 'ambulatory' in encounter_class or 'outpatient' in description:
                    metrics['OUTPATIENT_VISITS'] += 1
                
                # Identify primary care vs specialist
                if 'primary care' in description or 'general practice' in description:
                    metrics['PRIMARY_CARE_VISITS'] += 1
                elif any(term in description for term in ['specialist', 'cardiology', 'neurology', 'orthopedic']):
                    metrics['SPECIALIST_VISITS'] += 1
                
                # Add encounter cost if available
                if pd.notna(enc.get('TOTAL_CLAIM_COST')):
                    metrics['TOTAL_COST'] += float(enc['TOTAL_CLAIM_COST'])
                
                # Calculate encounter duration if both start and stop dates are available
                if pd.notna(enc['START']) and pd.notna(enc['STOP']):
                    duration_days = (enc['STOP'] - enc['START']).days
                    if duration_days > 0:
                        metrics['AVERAGE_ENCOUNTER_DURATION_DAYS'] += duration_days
            
            # Calculate average duration
            if metrics['TOTAL_ENCOUNTERS'] > 0:
                metrics['AVERAGE_ENCOUNTER_DURATION_DAYS'] /= metrics['TOTAL_ENCOUNTERS']
            
            # Calculate days since last encounter
            most_recent_date = patient_encounters['START'].max()
            metrics['DAYS_SINCE_LAST_ENCOUNTER'] = (end_date - most_recent_date).days
        
        return {f'UTIL_{k}': v for k, v in metrics.items()}
    
    def prepare_features(self, timeline_df, exclude_cols=None):
        """
        Prepare features for model training, handling missing values and scaling
        
        Args:
            timeline_df: DataFrame with patient timeline features
            exclude_cols: List of columns to exclude from feature set (defaults to time and target cols)
            
        Returns:
            Dictionary with processed features and metadata
        """
        # Define default columns to exclude if not provided
        if exclude_cols is None:
            exclude_cols = [
                'PATIENT_ID', 'TIME_POINT', 'HAD_EMERGENCY_NEXT_MONTH', 
                'NEW_DIAGNOSES_NEXT_MONTH', 'NEW_MEDICATIONS_NEXT_MONTH', 
                'NEW_PROCEDURES_NEXT_MONTH'
            ]
        
        # Identify all columns to exclude
        cols_to_exclude = []
        for col in exclude_cols:
            matching_cols = [c for c in timeline_df.columns if c == col or c.startswith(col + '_')]
            cols_to_exclude.extend(matching_cols)
        
        # Print diagnostic information
        print(f"Total columns: {len(timeline_df.columns)}")
        print(f"Columns being excluded: {cols_to_exclude}")
        
        # Separate features from targets and metadata
        X = timeline_df.drop(cols_to_exclude, axis=1, errors='ignore')
        print(f"Feature columns after exclusion: {len(X.columns)}")
        
        # Convert problematic numeric columns to numeric type with NaN for non-convertible values
        numeric_prefixes = ['VITAL_', 'COND_', 'MED_', 'PROC_', 'IMG_', 'DEV_', 'ALLERGY_', 
                            'IMMUNIZATION_', 'IMM_', 'UTIL_']
        
        for col in X.columns:
            if any(col.startswith(prefix) for prefix in numeric_prefixes) or col == 'DEM_AGE_YEARS':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert {col} to numeric. Error: {str(e)}")
        
        # Identify categorical and numerical features
        categorical_features = [col for col in X.columns if col.startswith('DEM_') and col not in ['DEM_AGE_YEARS']]
        numerical_features = [col for col in X.columns if col not in categorical_features]
        
        print(f"Identified {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        
        # Create preprocessor
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )
        
        # Fit preprocessor
        try:
            X_processed = preprocessor.fit_transform(X)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Simplified preprocessing as fallback
            print("Falling back to simplified preprocessing...")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ],
                remainder='passthrough'
            )
            X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names_processed = []
        
        try:
            # Try using get_feature_names_out (newer scikit-learn versions)
            feature_names_processed = preprocessor.get_feature_names_out()
        except:
            try:
                # For older scikit-learn versions
                num_features = [f"{col}" for col in numerical_features]
                
                cat_features = []
                if categorical_features:
                    cat_transformer = preprocessor.transformers_[1][1]
                    if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                        onehot = cat_transformer.named_steps['onehot']
                        if hasattr(onehot, 'get_feature_names_out'):
                            cat_features = onehot.get_feature_names_out(categorical_features)
                        elif hasattr(onehot, 'get_feature_names'):
                            cat_features = onehot.get_feature_names(categorical_features)
                
                feature_names_processed = num_features + list(cat_features)
            except Exception as e:
                print(f"Could not extract processed feature names: {e}")
                # Generate generic feature names
                feature_names_processed = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        print(f"Original feature count: {len(X.columns)}, Processed feature count: {len(feature_names_processed)}")
        
        return {
            'X': X,  # Original features DataFrame
            'X_processed': X_processed,  # Transformed feature matrix
            'preprocessor': preprocessor,
            'feature_names': X.columns.tolist(),  # Original feature names
            'feature_names_processed': feature_names_processed,  # Post-transformation feature names
            'categorical_features': categorical_features,
            'numerical_features': numerical_features
        }
    
    def build_emergency_visit_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            # Smaller network with stronger regularization
            keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01)),  # Increased L2
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),  # Increased dropout
            # Single hidden layer is often sufficient for this type of problem
            keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Define metrics
        metrics = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.F1Score(
                name='f1_score',
                threshold=0.5,
                dtype=tf.float32
            )
        ]
        
        # Modified optimizer with reduced learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-4),
            loss='binary_crossentropy',
            metrics=metrics
        )
        
        return model
    
    def build_multi_label_model(self, input_shape, num_classes, model_type):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            # Smaller initial layer
            keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01)),  # Increased L2
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),  # Increased dropout
            # Smaller network overall (removed one layer)
            keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01)),  # Added L2 to all layers
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),  # Increased dropout
            keras.layers.Dense(num_classes, activation='sigmoid')
        ])
        
        # Define metrics
        metrics = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.F1Score(
                name='f1_score',
                threshold=0.5,
                dtype=tf.float32
            )
        ]
        
        # Modified optimizer with stronger regularization
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-4),  # Reduced learning rate
            loss='binary_crossentropy',
            metrics=metrics
        )
        
        return model

    def cross_validate_model(self, X, y, build_model_fn, n_splits=5):
        """
        Perform k-fold cross-validation for more reliable performance estimates
        
        Args:
            X: Input features
            y: Target values
            build_model_fn: Function to build the model
            n_splits: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        
        # For multi-label data, create a stratification target
        # by converting the multi-hot encoding to integers
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Use the sum of positive classes as a simple stratification target
            strat_y = np.sum(y, axis=1).astype(int)
            # Cap at 3 for better stratification
            strat_y = np.minimum(strat_y, 3)
        else:
            strat_y = y.flatten()
        
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store metrics
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auc': [],
            'f1_score': []
        }
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, strat_y)):
            print(f"Training fold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Balance training data
            X_train, y_train = self.balance_multi_label_data(X_train, y_train)
            
            # Build and train model
            model = build_model_fn()
            
            model.fit(
                X_train, y_train,
                epochs=15,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            # Evaluate model
            eval_results = model.evaluate(X_val, y_val, verbose=0)
            
            # Store metrics
            for i, metric_name in enumerate(model.metrics_names):
                if metric_name in metrics:
                    metrics[metric_name].append(eval_results[i])
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        std_metrics = {k: np.std(v) for k, v in metrics.items()}
        
        print("Cross-validation results:")
        for metric in avg_metrics:
            print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
        
        return {
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'all_metrics': metrics
        }
    
    def select_important_features(self, X, y, feature_names, n_features=50):
        """
        Select the most important features using a simple model
        
        Args:
            X: Input features
            y: Target values
            feature_names: List of feature names
            n_features: Number of features to select
            
        Returns:
            List of selected feature indices and names
        """
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # For multi-label, we'll use the sum of positive labels as target
        if len(y.shape) > 1:
            # Create a binary target based on whether any positive class exists
            target = np.any(y == 1, axis=1).astype(int)
        else:
            target = y
        
        # Train a simple random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, target)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create a dataframe of features and importances
        import pandas as pd
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Select top n features
        selected_features = feature_importance.head(n_features)
        
        # Get indices of selected features
        selected_indices = [list(feature_names).index(feat) for feat in selected_features['Feature']]
        
        print(f"Selected {len(selected_indices)} features with importance range: "
            f"{selected_features['Importance'].min():.6f} - {selected_features['Importance'].max():.6f}")
        
        return selected_indices, selected_features['Feature'].tolist()
        
    def balance_multi_label_data(self, X, y, upsample_ratio=0.7):
        """
        Balance multi-label data through targeted upsampling
        
        Args:
            X: Input features
            y: Multi-hot encoded target array
            upsample_ratio: Target ratio for positive samples
            
        Returns:
            Balanced X and y
        """
        from sklearn.utils import resample
        
        # If target is 1D, convert to 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # For each label column
        n_labels = y.shape[1]
        all_indices = set(range(len(X)))
        new_X, new_y = X.copy(), y.copy()
        
        for label_idx in range(n_labels):
            # Get positive samples for this label
            pos_indices = np.where(y[:, label_idx] == 1)[0]
            pos_count = len(pos_indices)
            total_count = len(y)
            
            # Calculate how many to add for the target ratio
            current_ratio = pos_count / total_count
            target_count = int(total_count * upsample_ratio)
            
            if current_ratio < upsample_ratio and pos_count > 0:
                # Number of samples to generate through upsampling
                n_to_add = target_count - pos_count
                
                if n_to_add > 0:
                    # Upsample positive examples
                    pos_X = X[pos_indices]
                    pos_y = y[pos_indices]
                    
                    # Resample with replacement
                    upsampled_X, upsampled_y = resample(
                        pos_X, pos_y, 
                        replace=True, 
                        n_samples=n_to_add,
                        random_state=42
                    )
                    
                    # Combine with original data
                    new_X = np.vstack([new_X, upsampled_X])
                    new_y = np.vstack([new_y, upsampled_y])
        
        return new_X, new_y
            
    def train_models(self, timeline_df, targets):
        """
        Train prediction models for all target outcomes with improvements to prevent overfitting
        
        Args:
            timeline_df: DataFrame with patient timeline features
            targets: Dictionary with target arrays for each prediction task
            
        Returns:
            Dictionary with trained models and their evaluation metrics
        """
        from sklearn.model_selection import train_test_split
        import os
        import numpy as np
        from tensorflow import keras
        import tensorflow as tf
        from sklearn.ensemble import RandomForestClassifier
        
        
        # Create directory for checkpoints
        os.makedirs('saved_models/checkpoints', exist_ok=True)
        
        # Prepare features
        print("Preparing features...")
        features = self.prepare_features(timeline_df)
        
        print("Target Columns:", targets)
        
        X = features['X_processed']
        input_shape = X.shape[1]  # Number of features after processing
        
        results = {}
        
        '''
        # ----------------- EMERGENCY VISIT MODEL -----------------
        print("\nTraining emergency visit prediction model...")
        y_emergency = targets['emergency_visit']
        if len(y_emergency) > 0:
            # Convert to boolean array if needed
            if y_emergency.dtype != bool and y_emergency.dtype != np.bool_:
                y_emergency = np.array([bool(val) if val != 'nan' else False for val in y_emergency], dtype=bool)
            
            # Feature selection for emergency model
            from sklearn.ensemble import RandomForestClassifier
            
            print("Performing feature selection...")
            feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
            feature_selector.fit(X, y_emergency)
            
            # Get feature importances and select top 30 features
            importances = feature_selector.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = 30
            selected_indices = indices[:top_n]

            print("Selected Feature Indices: ", selected_indices)
            
            # Use only selected features
            X_selected = X[:, selected_indices]
            selected_feature_names = [features['feature_names'][i] for i in selected_indices]
            
            print(f"Selected {len(selected_indices)} features for emergency model")
            print("Top 30 features:", [features['feature_names'][i] for i in indices[:30]])
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_emergency, test_size=0.2, random_state=42, 
                stratify=y_emergency  # Ensure balanced splits
            )

            if np.mean(y_train) < 0.3:  # If less than 30% are positive
               X_train, y_train = self.balance_multi_label_data(X_train, y_train, upsample_ratio=0.2)
            
            # Reshape y_train and y_test to be 2D
            # This is the key fix for the F1Score metric error
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
            
            # Print class distribution
            pos_count = np.sum(y_train)
            neg_count = len(y_train) - pos_count
            print(f"Class distribution - Positive: {pos_count}/{len(y_train)} ({100*pos_count/len(y_train):.2f}%)")
            
            # Set class weights more conservatively
            if pos_count / len(y_train) < 0.3:
                class_weight = {
                    0: 1.0, 
                    1: min(3.0, neg_count / pos_count)  # Cap at 3x to prevent extreme weights
                }
            else:
                class_weight = None
                
            print(f"Using class weights: {class_weight}")
            
            

            # Build simplified emergency visit model
            emergency_model = keras.Sequential([
                keras.layers.Input(shape=(len(selected_indices),)),
                # Smaller layer with stronger regularization
                keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),  # Increased dropout
                # Only one hidden layer
                keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Define metrics
            metrics = [
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                # Remove F1Score for now to simplify
                # We can calculate it manually after training
            ]
            
            # Compile with reduced learning rate
            emergency_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-4),
                loss='binary_crossentropy',
                metrics=metrics
            )
            
            # Create checkpoint callback
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath='saved_models/checkpoints/emergency_model_epoch_{epoch:02d}.keras',
                save_best_only=False,  # Save every epoch
                save_weights_only=False,
                verbose=1
            )
            
            print("==Emergency Input Shape: ==")
            print("X Shape: ", X_train.shape)
            print("Y shape: ", y_train.shape)

            # Train with early stopping based on loss
            emergency_history = emergency_model.fit(
                X_train, y_train,
                epochs=50,  # Reduced epochs
                batch_size=32,  # Smaller batch size
                validation_split=0.2,
                class_weight=class_weight,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',  # Monitor loss instead of recall
                        patience=10,  # Reduced patience
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=0.00001,
                        verbose=1
                    ),
                    checkpoint_callback
                ],
                verbose=1
            )
            
            # Evaluate model
            print("Evaluating emergency model...")
            emergency_eval = emergency_model.evaluate(X_test, y_test)
            emergency_metrics = {name: value for name, value in zip(emergency_model.metrics_names, emergency_eval)}
            
            # Get predictions
            y_pred_prob = emergency_model.predict(X_test).flatten()
            
            # Find optimal threshold
            from sklearn.metrics import precision_recall_curve, f1_score
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            print(f"Optimal threshold: {best_threshold:.4f}")
            
            # Get predictions at optimal threshold
            y_pred = (y_pred_prob >= best_threshold).astype(int)
            
            # Calculate F1 score manually
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred)
            emergency_metrics['f1_score'] = f1
            print(f"F1 Score: {f1:.4f}")
            
            # Calculate final metrics
            from sklearn.metrics import classification_report, confusion_matrix
            print("\nClassification report:")
            print(classification_report(y_test, y_pred))
            
            # Store results
            results['emergency_visit'] = {
                'model': emergency_model,
                'history': emergency_history.history,
                'metrics': emergency_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'best_threshold': best_threshold,
                'selected_features': selected_feature_names
            }
            
            print(f"Emergency visit model metrics: {emergency_metrics}")
        '''
        # ----------------- DIAGNOSIS MODEL -----------------
        if len(self.diagnosis_categories) > 0 and 'diagnoses' in targets and len(targets['diagnoses']) > 0:
            print("\nTraining diagnosis prediction model...")
            y_diagnosis = targets['diagnoses']
            
            # Feature selection for diagnosis model
            print("Performing feature selection for diagnosis model...")
            # Create a target for feature selection (presence of any diagnosis)
            y_any_diagnosis = np.any(y_diagnosis == 1, axis=1).astype(int)
            
            feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
            feature_selector.fit(X, y_any_diagnosis)
            
            # Get feature importances and select top features
            importances = feature_selector.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = 30  # More features for multi-label
            selected_indices = indices[:top_n]
            print("Selected Feature Indices: ", selected_indices)
            # Use only selected features
            X_selected = X[:, selected_indices]
            selected_feature_names = [features['feature_names'][i] for i in selected_indices]
            
            print(f"Selected {len(selected_indices)} features for diagnosis model")
            print("Top 40 features:", [features['feature_names'][i] for i in indices[:30]])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_diagnosis, test_size=0.2, random_state=42
            )

            X_train, y_train = self.balance_multi_label_data(X_train, y_train, upsample_ratio=0.5)

            print("==Diagnosis Input Shape Before Reshaping: ==")
            print("X Shape: ", X_train.shape)
            print("Y shape: ", y_train.shape)
            
            # Check and ensure y_train and y_test are in the right shape
            # For multi-label, they should already be 2D, but let's verify
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
            
            # Simplified balancing approach
            # Just undersample negative examples to maintain a reasonable ratio
            pos_mask = np.any(y_train == 1, axis=1)
            pos_indices = np.where(pos_mask)[0]
            neg_indices = np.where(~pos_mask)[0]
            
            # If we have too many negative examples
            if len(neg_indices) > 3 * len(pos_indices) and len(pos_indices) > 0:
                # Randomly select negative samples
                np.random.seed(42)
                selected_neg_indices = np.random.choice(
                    neg_indices, size=3 * len(pos_indices), replace=False
                )
                
                # Combine indices
                selected_sample_indices = np.concatenate([pos_indices, selected_neg_indices])
                
                # Create balanced datasets
                X_train = X_train[selected_sample_indices]
                y_train = y_train[selected_sample_indices]
            
            # Build a simplified multi-label model
            diagnosis_model = keras.Sequential([
                keras.layers.Input(shape=(len(selected_indices),)),
                # Smaller network with stronger regularization
                keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                # Only one hidden layer
                keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(self.diagnosis_categories), activation='sigmoid')
            ])
            
            # Define metrics - removing F1Score to avoid shape issues
            metrics = [
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
            
            # Compile with reduced learning rate
            diagnosis_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-4),
                loss='binary_crossentropy',
                metrics=metrics
            )
            
            # Create checkpoint callback
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath='saved_models/checkpoints/diagnosis_model_epoch_{epoch:02d}.keras',
                save_best_only=False,
                save_weights_only=False,
                verbose=1
            )
            
            # Simplified training with sample weights instead of class weights
            # This works better for multi-label classification
            sample_weights = np.ones(len(y_train))
            pos_samples = np.sum(y_train, axis=1) > 0
            sample_weights[pos_samples] = 3.0  # Weight positive samples higher
            

            print("==Diagnosis Input After Reshaping: ==")
            print("X Shape: ", X_train.shape)
            print("Y shape: ", y_train.shape)

            diagnosis_history = diagnosis_model.fit(
                X_train, y_train,
                epochs=30,  # Reduced epochs
                batch_size=32,  # Smaller batch size
                validation_split=0.2,
                sample_weight=sample_weights,  # Use sample weights instead of class weights
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',  # Monitor loss instead of recall
                        patience=3,  # Reduced patience
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        min_lr=0.00001,
                        verbose=1
                    ),
                    checkpoint_callback
                ],
                verbose=1
            )
            
            # Evaluate model
            print("Evaluating diagnosis model...")
            diagnosis_eval = diagnosis_model.evaluate(X_test, y_test)
            diagnosis_metrics = {name: value for name, value in zip(diagnosis_model.metrics_names, diagnosis_eval)}
            
            # Get predictions
            y_pred_prob = diagnosis_model.predict(X_test)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate F1 scores manually
            from sklearn.metrics import f1_score
            # Calculate f1 for each class
            f1_scores = []
            for i in range(y_test.shape[1]):
                f1 = f1_score(y_test[:, i], y_pred[:, i])
                f1_scores.append(f1)
            # Average F1 score
            avg_f1 = np.mean(f1_scores)
            diagnosis_metrics['f1_score'] = avg_f1
            print(f"Average F1 Score: {avg_f1:.4f}")
            
            # Store results
            results['diagnosis'] = {
                'model': diagnosis_model,
                'history': diagnosis_history.history,
                'metrics': diagnosis_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'categories': self.diagnosis_categories,
                'selected_features': selected_feature_names
            }
            
            print(f"Diagnosis model metrics: {diagnosis_metrics}")
        else:
            print("\nSkipping diagnosis model: no diagnosis categories or target data available")

        # ----------------- MEDICATION MODEL -----------------
        if len(self.medication_categories) > 0 and 'medications' in targets and len(targets['medications']) > 0:
            print("\nTraining medication prediction model...")
            y_medication = targets['medications']
            
            # Feature selection for medication model
            print("Performing feature selection for medication model...")
            # Create a target for feature selection (presence of any medication)
            y_any_medication = np.any(y_medication == 1, axis=1).astype(int)
            
            feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
            feature_selector.fit(X, y_any_medication)
            
            # Get feature importances and select top features
            importances = feature_selector.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = 30  # More features for multi-label
            selected_indices = indices[:top_n]
            print("Selected Feature Indices: ", selected_indices)
            # Use only selected features
            X_selected = X[:, selected_indices]
            selected_feature_names = [features['feature_names'][i] for i in selected_indices]
            
            print(f"Selected {len(selected_indices)} features for medication model")
            print("Top 40 features:", [features['feature_names'][i] for i in indices[:30]])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_medication, test_size=0.2, random_state=42
            )

            X_train, y_train = self.balance_multi_label_data(X_train, y_train, upsample_ratio=0.5)
            
            # Check and ensure y_train and y_test are in the right shape
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
            
            # Simplified balancing approach
            pos_mask = np.any(y_train == 1, axis=1)
            pos_indices = np.where(pos_mask)[0]
            neg_indices = np.where(~pos_mask)[0]
            
            # If we have too many negative examples
            if len(neg_indices) > 3 * len(pos_indices) and len(pos_indices) > 0:
                # Randomly select negative samples
                np.random.seed(42)
                selected_neg_indices = np.random.choice(
                    neg_indices, size=3 * len(pos_indices), replace=False
                )
                
                # Combine indices
                selected_sample_indices = np.concatenate([pos_indices, selected_neg_indices])
                
                # Create balanced datasets
                X_train = X_train[selected_sample_indices]
                y_train = y_train[selected_sample_indices]
            
            # Build a simplified multi-label model
            medication_model = keras.Sequential([
                keras.layers.Input(shape=(len(selected_indices),)),
                # Smaller network with stronger regularization
                keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                # Only one hidden layer
                keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(self.medication_categories), activation='sigmoid')
            ])
            
            # Define metrics - removing F1Score
            metrics = [
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
            
            # Compile with reduced learning rate
            medication_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-4),
                loss='binary_crossentropy',
                metrics=metrics
            )
            
            # Create checkpoint callback
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath='saved_models/checkpoints/medication_model_epoch_{epoch:02d}.keras',
                save_best_only=False,
                save_weights_only=False,
                verbose=1
            )
            
            # Sample weights instead of class weights
            sample_weights = np.ones(len(y_train))
            pos_samples = np.sum(y_train, axis=1) > 0
            sample_weights[pos_samples] = 3.0  # Weight positive samples higher
            
            # Simplified training
            medication_history = medication_model.fit(
                X_train, y_train,
                epochs=30,  # Reduced epochs
                batch_size=32,  # Smaller batch size
                validation_split=0.2,
                sample_weight=sample_weights,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',  # Monitor loss instead of recall
                        patience=3,  # Reduced patience
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        min_lr=0.00001,
                        verbose=1
                    ),
                    checkpoint_callback
                ],
                verbose=1
            )
            
            # Evaluate model
            print("Evaluating medication model...")
            medication_eval = medication_model.evaluate(X_test, y_test)
            medication_metrics = {name: value for name, value in zip(medication_model.metrics_names, medication_eval)}
            
            # Get predictions
            y_pred_prob = medication_model.predict(X_test)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate F1 scores manually
            from sklearn.metrics import f1_score
            # Calculate f1 for each class
            f1_scores = []
            for i in range(y_test.shape[1]):
                f1 = f1_score(y_test[:, i], y_pred[:, i])
                f1_scores.append(f1)
            # Average F1 score
            avg_f1 = np.mean(f1_scores)
            medication_metrics['f1_score'] = avg_f1
            print(f"Average F1 Score: {avg_f1:.4f}")
            
            # Store results
            results['medication'] = {
                'model': medication_model,
                'history': medication_history.history,
                'metrics': medication_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'categories': self.medication_categories,
                'selected_features': selected_feature_names
            }
            
            print(f"Medication model metrics: {medication_metrics}")
        else:
            print("\nSkipping medication model: no medication categories or target data available")

        # ----------------- PROCEDURE MODEL -----------------
        if len(self.procedure_categories) > 0 and 'procedures' in targets and len(targets['procedures']) > 0:
            print("\nTraining procedure prediction model...")
            y_procedure = targets['procedures']
            
            # Feature selection for procedure model
            print("Performing feature selection for procedure model...")
            # Create a target for feature selection (presence of any procedure)
            y_any_procedure = np.any(y_procedure == 1, axis=1).astype(int)
            
            feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
            feature_selector.fit(X, y_any_procedure)
            
            # Get feature importances and select top features
            importances = feature_selector.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = 30  # More features for multi-label
            selected_indices = indices[:top_n]

            print("Selected Feature Indices: ", selected_indices)
            
            # Use only selected features
            X_selected = X[:, selected_indices]
            selected_feature_names = [features['feature_names'][i] for i in selected_indices]
            
            print(f"Selected {len(selected_indices)} features for procedure model")
            print("Top 30 features:", [features['feature_names'][i] for i in indices[:30]])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_procedure, test_size=0.2, random_state=42
            )

            X_train, y_train = self.balance_multi_label_data(X_train, y_train, upsample_ratio=0.6)
            
            # Check and ensure y_train and y_test are in the right shape
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
            
            # Simplified balancing approach
            pos_mask = np.any(y_train == 1, axis=1)
            pos_indices = np.where(pos_mask)[0]
            neg_indices = np.where(~pos_mask)[0]
            
            # If we have too many negative examples
            if len(neg_indices) > 3 * len(pos_indices) and len(pos_indices) > 0:
                # Randomly select negative samples
                np.random.seed(42)
                selected_neg_indices = np.random.choice(
                    neg_indices, size=3 * len(pos_indices), replace=False
                )
                
                # Combine indices
                selected_sample_indices = np.concatenate([pos_indices, selected_neg_indices])
                
                # Create balanced datasets
                X_train = X_train[selected_sample_indices]
                y_train = y_train[selected_sample_indices]
            
            # Build a simplified multi-label model
            procedure_model = keras.Sequential([
                keras.layers.Input(shape=(len(selected_indices),)),
                # Smaller network with stronger regularization
                keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                # Only one hidden layer
                keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=keras.regularizers.l2(0.01)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(self.procedure_categories), activation='sigmoid')
            ])
            
            # Define metrics - removing F1Score
            metrics = [
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
            
            # Compile with reduced learning rate
            procedure_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-4),
                loss='binary_crossentropy',
                metrics=metrics
            )
            
            # Create checkpoint callback
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath='saved_models/checkpoints/procedure_model_epoch_{epoch:02d}.keras',
                save_best_only=False,
                save_weights_only=False,
                verbose=1
            )
            
            # Sample weights instead of class weights
            sample_weights = np.ones(len(y_train))
            pos_samples = np.sum(y_train, axis=1) > 0
            sample_weights[pos_samples] = 3.0  # Weight positive samples higher
            
            # Simplified training
            procedure_history = procedure_model.fit(
                X_train, y_train,
                epochs=30,  # Reduced epochs
                batch_size=32,  # Smaller batch size
                validation_split=0.2,
                sample_weight=sample_weights,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',  # Monitor loss instead of recall
                        patience=3,  # Reduced patience
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        min_lr=0.00001,
                        verbose=1
                    ),
                    checkpoint_callback
                ],
                verbose=1
            )
            
            # Evaluate model
            print("Evaluating procedure model...")
            procedure_eval = procedure_model.evaluate(X_test, y_test)
            procedure_metrics = {name: value for name, value in zip(procedure_model.metrics_names, procedure_eval)}
            
            # Get predictions
            y_pred_prob = procedure_model.predict(X_test)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate F1 scores manually
            from sklearn.metrics import f1_score
            # Calculate f1 for each class
            f1_scores = []
            for i in range(y_test.shape[1]):
                f1 = f1_score(y_test[:, i], y_pred[:, i])
                f1_scores.append(f1)
            # Average F1 score
            avg_f1 = np.mean(f1_scores)
            procedure_metrics['f1_score'] = avg_f1
            print(f"Average F1 Score: {avg_f1:.4f}")
            
            # Store results
            results['procedure'] = {
                'model': procedure_model,
                'history': procedure_history.history,
                'metrics': procedure_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'categories': self.procedure_categories,
                'selected_features': selected_feature_names
            }
            
            print(f"Procedure model metrics: {procedure_metrics}")
        else:
            print("\nSkipping procedure model: no procedure categories or target data available")
        
        # Save feature preprocessor
        self.preprocessors['main'] = features['preprocessor']
        
        # Update models dictionary
        for key, value in results.items():
            if 'model' in value:
                self.models[key] = value['model']
        
        # Save training history and create plots
        self.save_training_history(results)
        
        return results
    
    def save_models(self, results, output_dir='saved_models'):
        """
        Save trained models and preprocessors
        
        Args:
            results: Dictionary with model results
            output_dir: Directory to save models
        """
        import joblib
        import json
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each model
        for model_type, result in results.items():
            if 'model' not in result:
                continue
            
            # Save TensorFlow model
            model_path = os.path.join(output_dir, f"{model_type}_model_{timestamp}.keras")
            result['model'].save(model_path)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'metrics': {k: float(v) for k, v in result.get('metrics', {}).items()},
                'categories': result.get('categories', []),
                'selected_features': result.get('selected_features', [])
            }
            
            metadata_path = os.path.join(output_dir, 'metadata', f"{model_type}_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved {model_type} model to {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(output_dir, f"preprocessor_{timestamp}.joblib")
        joblib.dump(self.preprocessors['main'], preprocessor_path)
        print(f"Saved preprocessor to {preprocessor_path}")
    
    def generate_clinical_insights(self, results):
        """
        Generate clinical insights based on model results
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Dictionary of clinical insights
        """
        insights = {
            'emergency_risk_factors': [],
            'common_diagnoses': [],
            'recommended_medications': [],
            'suggested_procedures': []
        }
        
        # Emergency risk factor insights
        if 'emergency_visit' in results:
            # Extract feature importance for emergency model
            emergency_model = results['emergency_visit']['model']
            
            # Get weights from the first layer
            if len(emergency_model.layers) > 1:
                weights = emergency_model.layers[1].get_weights()[0]
                
                # Calculate feature importance - Modified to handle different weight shapes
                # Check the shape of weights and adapt accordingly
                if len(weights.shape) == 1:
                    # 1D weights
                    importance = np.abs(weights)
                elif len(weights.shape) == 2:
                    # 2D weights - take mean across appropriate axis
                    importance = np.abs(weights).mean(axis=1)
                else:
                    # For higher dimensions, flatten and take absolute values
                    importance = np.abs(weights.flatten())
                
                # Get feature names
                feature_names = []
                if hasattr(self.preprocessors['main'], 'transformers_'):
                    # Get numerical feature names directly
                    num_transformer = self.preprocessors['main'].transformers_[0][1]
                    numerical_features = self.preprocessors['main'].transformers_[0][2]
                    feature_names.extend(numerical_features)
                    
                    # Get one-hot encoded categorical features
                    cat_transformer = self.preprocessors['main'].transformers_[1][1]
                    categorical_features = self.preprocessors['main'].transformers_[1][2]
                    if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                        onehot = cat_transformer.named_steps['onehot']
                        if hasattr(onehot, 'get_feature_names_out'):
                            cat_feature_names = onehot.get_feature_names_out(categorical_features)
                            feature_names.extend(cat_feature_names)
                
                # Match feature names with importance scores
                if feature_names and len(feature_names) == len(importance):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    })
                    
                    # Sort by importance
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Extract top risk factors
                    for _, row in importance_df.head(10).iterrows():
                        feature = row['Feature']
                        importance_value = row['Importance']
                        
                        # Clean up feature name for display
                        feature_display = feature.replace('VITAL_', 'Vital Sign: ')
                        feature_display = feature_display.replace('MED_', 'Medication: ')
                        feature_display = feature_display.replace('COND_', 'Condition: ')
                        feature_display = feature_display.replace('PROC_', 'Procedure: ')
                        feature_display = feature_display.replace('UTIL_', 'Utilization: ')
                        feature_display = feature_display.replace('DEM_', 'Demographic: ')
                        feature_display = feature_display.replace('_', ' ').title()
                        
                        insights['emergency_risk_factors'].append({
                            'factor': feature_display,
                            'importance': float(importance_value)
                        })
        
        # Generate insights for diagnosis prediction
        if 'diagnosis' in results and 'categories' in results['diagnosis']:
            diagnosis_model = results['diagnosis']['model']
            categories = results['diagnosis']['categories']
            
            # Get the average prediction probability for each diagnosis
            if 'y_pred' in results['diagnosis']:
                y_pred = results['diagnosis']['y_pred']
                avg_probs = np.mean(y_pred, axis=0)
                
                # Create a dataframe of diagnosis categories and their probabilities
                diagnosis_df = pd.DataFrame({
                    'Diagnosis': categories,
                    'Probability': avg_probs
                })
                
                # Sort by probability
                diagnosis_df = diagnosis_df.sort_values('Probability', ascending=False)
                
                # Extract top diagnoses
                for _, row in diagnosis_df.head(10).iterrows():
                    insights['common_diagnoses'].append({
                        'diagnosis': row['Diagnosis'],
                        'probability': float(row['Probability'])
                    })
        
        # Generate insights for medication recommendations
        if 'medication' in results and 'categories' in results['medication']:
            medication_model = results['medication']['model']
            categories = results['medication']['categories']
            
            # Get the average prediction probability for each medication
            if 'y_pred' in results['medication']:
                y_pred = results['medication']['y_pred']
                avg_probs = np.mean(y_pred, axis=0)
                
                # Create a dataframe of medication categories and their probabilities
                medication_df = pd.DataFrame({
                    'Medication': categories,
                    'Probability': avg_probs
                })
                
                # Sort by probability
                medication_df = medication_df.sort_values('Probability', ascending=False)
                
                # Extract top medications
                for _, row in medication_df.head(10).iterrows():
                    insights['recommended_medications'].append({
                        'medication': row['Medication'],
                        'probability': float(row['Probability'])
                    })
        
        # Generate insights for procedure suggestions
        if 'procedure' in results and 'categories' in results['procedure']:
            procedure_model = results['procedure']['model']
            categories = results['procedure']['categories']
            
            # Get the average prediction probability for each procedure
            if 'y_pred' in results['procedure']:
                y_pred = results['procedure']['y_pred']
                avg_probs = np.mean(y_pred, axis=0)
                
                # Create a dataframe of procedure categories and their probabilities
                procedure_df = pd.DataFrame({
                    'Procedure': categories,
                    'Probability': avg_probs
                })
                
                # Sort by probability
                procedure_df = procedure_df.sort_values('Probability', ascending=False)
                
                # Extract top procedures
                for _, row in procedure_df.head(10).iterrows():
                    insights['suggested_procedures'].append({
                        'procedure': row['Procedure'],
                        'probability': float(row['Probability'])
                    })
        
        return insights
    
    def _process_additional_patients(self, new_patients, update_patients, existing_df, output_dir, append_new_months=True):
        """
        Process new patients and update existing patients, appending each record to CSV as it's generated
        
        Args:
            new_patients: List of new patient IDs to process
            update_patients: List of existing patient IDs to update
            existing_df: DataFrame with existing timeline data
            output_dir: Directory for timeline data
            append_new_months: Whether to append new months to existing patients
            
        Returns:
            Tuple of (updated DataFrame, updated targets dictionary)
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define the output CSV file
        output_csv = os.path.join(output_dir, f'enhanced_patient_timelines_{timestamp}.csv')
        
        # Write the header to the CSV file if creating a new file
        if not os.path.exists(output_csv):
            # If we have existing data, use its columns
            if not existing_df.empty:
                header = existing_df.columns
            else:
                # Define a default set of columns - this might need to be expanded based on all possible features
                header = ['PATIENT_ID', 'TIME_POINT']
                # Add demographic columns
                header.extend(['DEM_AGE_YEARS', 'DEM_GENDER', 'DEM_RACE', 'DEM_ETHNICITY'])
                # Add placeholder for clinical features, utilization features (these will be dynamically expanded)
                header.extend(['HAD_EMERGENCY_NEXT_MONTH', 'NEW_DIAGNOSES_NEXT_MONTH', 
                            'NEW_MEDICATIONS_NEXT_MONTH', 'NEW_PROCEDURES_NEXT_MONTH'])
            
            # Write header to CSV
            pd.DataFrame(columns=header).to_csv(output_csv, index=False)
        
        # If existing data provided, write it to the CSV first
        if not existing_df.empty:
            existing_df.to_csv(output_csv, index=False, mode='w')  # overwrite with existing data
        
        # Target outcomes for new records
        target_outcomes = {
            'emergency_visit': [],
            'diagnoses': [],
            'medications': [],
            'procedures': []
        }
        
        # Counter for processed records
        processed_records = 0
        
        new_patients = False #TEMPORARY TO PREVENT NEW DATA, REMOVE FOR FINAL
        # Process new patients first
        if new_patients:
            print(f"\nProcessing {len(new_patients)} new patients:")
            patients_processed = 0
            
            for i, patient_id in enumerate(new_patients):
                try:
                    # Get patient demographics
                    patient_info = self.patients_df[self.patients_df['Id'] == patient_id].iloc[0]
                    
                    # Get patient's data
                    patient_encounters = self.encounters_df[self.encounters_df['PATIENT'] == patient_id]
                    
                    if patient_encounters.empty:
                        print(f"  No encounters found for patient {patient_id}, skipping")
                        continue  # Skip patients with no encounters
                    
                    start_date = patient_encounters['START'].min()
                    end_date = patient_encounters['START'].max()
                    
                    print(f"  Processing patient {patient_id} ({i+1}/{len(new_patients)}): {start_date} to {end_date}")
                    
                    # Create monthly timeline records
                    current_date = start_date.replace(day=1)
                    end_of_month = pd.Timestamp(end_date.year, end_date.month, end_date.days_in_month)
                    
                    month_records = 0  # Count of records for this patient
                    
                    while current_date <= end_of_month:
                        # Define the month period
                        next_month = current_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                        
                        # Define lookback period for features
                        lookback_months = 12  # Default lookback period
                        lookback_start = current_date - pd.DateOffset(months=lookback_months)
                        
                        # Define prediction period (the next month)
                        prediction_start = next_month + pd.DateOffset(days=1)
                        prediction_end = prediction_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                        
                        # Check for emergency visit in the next month
                        emergency_next_month = False
                        next_month_encounters = self.encounters_df[
                            (self.encounters_df['PATIENT'] == patient_id) &
                            (self.encounters_df['START'] >= prediction_start) &
                            (self.encounters_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_encounters.empty:
                            # Look for emergency department visits
                            for _, enc in next_month_encounters.iterrows():
                                description = str(enc['DESCRIPTION']).lower() if pd.notna(enc['DESCRIPTION']) else ''
                                encounter_class = str(enc['ENCOUNTERCLASS']).lower() if pd.notna(enc['ENCOUNTERCLASS']) else ''
                                
                                if ('emergency' in description or 'ed' in description or 
                                    'emergency' in encounter_class or 'urgent' in encounter_class):
                                    emergency_next_month = True
                                    break
                        
                        # Identify new diagnoses in the next month
                        new_diagnoses = []
                        next_month_conditions = self.conditions_df[
                            (self.conditions_df['PATIENT'] == patient_id) &
                            (self.conditions_df['START'] >= prediction_start) &
                            (self.conditions_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_conditions.empty:
                            new_diagnoses = next_month_conditions['DESCRIPTION'].tolist()
                        
                        # Identify new medications in the next month
                        new_medications = []
                        next_month_medications = self.medications_df[
                            (self.medications_df['PATIENT'] == patient_id) &
                            (self.medications_df['START'] >= prediction_start) &
                            (self.medications_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_medications.empty:
                            new_medications = next_month_medications['DESCRIPTION'].tolist()
                        
                        # Identify new procedures in the next month
                        new_procedures = []
                        next_month_procedures = self.procedures_df[
                            (self.procedures_df['PATIENT'] == patient_id) &
                            (self.procedures_df['START'] >= prediction_start) &
                            (self.procedures_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_procedures.empty:
                            new_procedures = next_month_procedures['DESCRIPTION'].tolist()
                        
                        # Extract comprehensive clinical features
                        clinical_features = self.extract_clinical_features(
                            patient_id, lookback_start, next_month
                        )
                        
                        # Create demographic features
                        demographic_features = {}
                        
                        # Calculate age at this time point
                        if pd.notna(patient_info['BIRTHDATE']):
                            age_days = (current_date - patient_info['BIRTHDATE']).days
                            demographic_features['AGE_YEARS'] = age_days / 365.25
                        else:
                            demographic_features['AGE_YEARS'] = None
                        
                        # Gender, Race, Ethnicity
                        demographic_features['GENDER'] = patient_info['GENDER'] if pd.notna(patient_info['GENDER']) else 'Unknown'
                        demographic_features['RACE'] = patient_info['RACE'] if pd.notna(patient_info['RACE']) else 'Unknown'
                        demographic_features['ETHNICITY'] = patient_info['ETHNICITY'] if pd.notna(patient_info['ETHNICITY']) else 'Unknown'
                        
                        # Calculate healthcare utilization features
                        utilization_features = self.calculate_utilization_metrics(patient_id, lookback_start, next_month)
                        
                        # Create record
                        record = {
                            'PATIENT_ID': patient_id,
                            'TIME_POINT': current_date,
                            # Add demographic features
                            **{f'DEM_{k}': v for k, v in demographic_features.items()},
                            # Add clinical features
                            **clinical_features,
                            # Add utilization features
                            **utilization_features,
                            # Target outcomes
                            'HAD_EMERGENCY_NEXT_MONTH': emergency_next_month,
                            'NEW_DIAGNOSES_NEXT_MONTH': str(new_diagnoses),  # Convert list to string for CSV storage
                            'NEW_MEDICATIONS_NEXT_MONTH': str(new_medications),
                            'NEW_PROCEDURES_NEXT_MONTH': str(new_procedures)
                        }
                        
                        # Create a DataFrame for this single record
                        record_df = pd.DataFrame([record])
                        
                        # Append the record to the CSV file
                        record_df.to_csv(output_csv, mode='a', header=False, index=False)
                        
                        # Increment counter
                        processed_records += 1
                        month_records += 1
                        
                        # Add to target outcomes
                        target_outcomes['emergency_visit'].append(emergency_next_month)
                        
                        # Multi-hot encoding for diagnoses, medications, procedures
                        diagnosis_target = [1 if dx in new_diagnoses else 0 for dx in self.diagnosis_categories]
                        target_outcomes['diagnoses'].append(diagnosis_target)
                        
                        medication_target = [1 if med in new_medications else 0 for med in self.medication_categories]
                        target_outcomes['medications'].append(medication_target)
                        
                        procedure_target = [1 if proc in new_procedures else 0 for proc in self.procedure_categories]
                        target_outcomes['procedures'].append(procedure_target)
                        
                        # Move to next month
                        current_date = current_date + pd.DateOffset(months=1)
                    
                    patients_processed += 1
                    print(f"    Generated {month_records} timeline records for patient {patient_id}")
                    
                    # Progress reporting
                    if patients_processed % 5 == 0:
                        print(f"  Processed {patients_processed}/{len(new_patients)} patients, {processed_records} total records")
                        
                except Exception as e:
                    print(f"  Error processing patient {patient_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        update_patients = False #REMOVE LATER, JUST TO PREVENT NEEDLESS COMPUTATION

        # Process update patients (existing patients that need new months added)
        if update_patients and append_new_months:
            print(f"\nUpdating {len(update_patients)} existing patients:")
            update_processed = 0
            
            for i, patient_id in enumerate(update_patients):
                try:
                    # Similar implementation as for new patients, but only add months past the latest in existing_df
                    # Get latest existing month for this patient
                    patient_existing = existing_df[existing_df['PATIENT_ID'] == patient_id]
                    
                    if patient_existing.empty:
                        print(f"  Patient {patient_id} not found in existing data, skipping")
                        continue
                    
                    latest_existing_date = pd.to_datetime(patient_existing['TIME_POINT'], format='mixed', errors='coerce').max()
                    
                    # Get patient's data
                    patient_encounters = self.encounters_df[self.encounters_df['PATIENT'] == patient_id]
                    
                    if patient_encounters.empty:
                        print(f"  No encounters found for patient {patient_id}, skipping")
                        continue
                    
                    # Get patient demographics
                    patient_info = self.patients_df[self.patients_df['Id'] == patient_id].iloc[0]
                    
                    # Find the latest encounter date
                    latest_encounter_date = patient_encounters['START'].max()
                    
                    # Only process if there are newer encounters than what's in the existing data
                    if latest_encounter_date <= latest_existing_date:
                        print(f"  No new data for patient {patient_id}, skipping")
                        continue
                    
                    # Start from the month after the latest existing month
                    start_date = latest_existing_date + pd.DateOffset(months=1)
                    start_date = start_date.replace(day=1)  # First day of next month
                    
                    end_date = latest_encounter_date
                    
                    print(f"  Updating patient {patient_id} ({i+1}/{len(update_patients)}): {start_date} to {end_date}")
                    
                    # Create monthly timeline records for new months
                    current_date = start_date
                    end_of_month = pd.Timestamp(end_date.year, end_date.month, end_date.days_in_month)
                    
                    month_records = 0  # Count of records for this patient update
                    
                    # Process each month (similar to new patients loop)
                    while current_date <= end_of_month:
                        # Same implementation as for new patients
                        # (Code repeated for each month, generating record and appending to CSV)
                        # Define the month period
                        next_month = current_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                        
                        # Define lookback period for features
                        lookback_months = 12
                        lookback_start = current_date - pd.DateOffset(months=lookback_months)
                        
                        # Define prediction period
                        prediction_start = next_month + pd.DateOffset(days=1)
                        prediction_end = prediction_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                        
                        # Check for emergency visit in the next month
                        emergency_next_month = False
                        next_month_encounters = self.encounters_df[
                            (self.encounters_df['PATIENT'] == patient_id) &
                            (self.encounters_df['START'] >= prediction_start) &
                            (self.encounters_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_encounters.empty:
                            # Look for emergency department visits
                            for _, enc in next_month_encounters.iterrows():
                                description = str(enc['DESCRIPTION']).lower() if pd.notna(enc['DESCRIPTION']) else ''
                                encounter_class = str(enc['ENCOUNTERCLASS']).lower() if pd.notna(enc['ENCOUNTERCLASS']) else ''
                                
                                if ('emergency' in description or 'ed' in description or 
                                    'emergency' in encounter_class or 'urgent' in encounter_class):
                                    emergency_next_month = True
                                    break
                        
                        # Identify new diagnoses, medications, procedures (same as for new patients)
                        new_diagnoses = []
                        next_month_conditions = self.conditions_df[
                            (self.conditions_df['PATIENT'] == patient_id) &
                            (self.conditions_df['START'] >= prediction_start) &
                            (self.conditions_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_conditions.empty:
                            new_diagnoses = next_month_conditions['DESCRIPTION'].tolist()
                        
                        new_medications = []
                        next_month_medications = self.medications_df[
                            (self.medications_df['PATIENT'] == patient_id) &
                            (self.medications_df['START'] >= prediction_start) &
                            (self.medications_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_medications.empty:
                            new_medications = next_month_medications['DESCRIPTION'].tolist()
                        
                        new_procedures = []
                        next_month_procedures = self.procedures_df[
                            (self.procedures_df['PATIENT'] == patient_id) &
                            (self.procedures_df['START'] >= prediction_start) &
                            (self.procedures_df['START'] <= prediction_end)
                        ]
                        
                        if not next_month_procedures.empty:
                            new_procedures = next_month_procedures['DESCRIPTION'].tolist()
                        
                        # Extract features and create record
                        clinical_features = self.extract_clinical_features(
                            patient_id, lookback_start, next_month
                        )
                        
                        demographic_features = {}
                        
                        if pd.notna(patient_info['BIRTHDATE']):
                            age_days = (current_date - patient_info['BIRTHDATE']).days
                            demographic_features['AGE_YEARS'] = age_days / 365.25
                        else:
                            demographic_features['AGE_YEARS'] = None
                        
                        demographic_features['GENDER'] = patient_info['GENDER'] if pd.notna(patient_info['GENDER']) else 'Unknown'
                        demographic_features['RACE'] = patient_info['RACE'] if pd.notna(patient_info['RACE']) else 'Unknown'
                        demographic_features['ETHNICITY'] = patient_info['ETHNICITY'] if pd.notna(patient_info['ETHNICITY']) else 'Unknown'
                        
                        utilization_features = self.calculate_utilization_metrics(patient_id, lookback_start, next_month)
                        
                        record = {
                            'PATIENT_ID': patient_id,
                            'TIME_POINT': current_date,
                            **{f'DEM_{k}': v for k, v in demographic_features.items()},
                            **clinical_features,
                            **utilization_features,
                            'HAD_EMERGENCY_NEXT_MONTH': emergency_next_month,
                            'NEW_DIAGNOSES_NEXT_MONTH': str(new_diagnoses),
                            'NEW_MEDICATIONS_NEXT_MONTH': str(new_medications),
                            'NEW_PROCEDURES_NEXT_MONTH': str(new_procedures)
                        }
                        
                        # Create a DataFrame for this single record
                        record_df = pd.DataFrame([record])
                        
                        # Append the record to the CSV file
                        record_df.to_csv(output_csv, mode='a', header=False, index=False)
                        
                        # Increment counter
                        processed_records += 1
                        month_records += 1
                        
                        # Add to target outcomes
                        target_outcomes['emergency_visit'].append(emergency_next_month)
                        
                        # Multi-hot encoding for diagnoses, medications, procedures
                        diagnosis_target = [1 if dx in new_diagnoses else 0 for dx in self.diagnosis_categories]
                        target_outcomes['diagnoses'].append(diagnosis_target)
                        
                        medication_target = [1 if med in new_medications else 0 for med in self.medication_categories]
                        target_outcomes['medications'].append(medication_target)
                        
                        procedure_target = [1 if proc in new_procedures else 0 for proc in self.procedure_categories]
                        target_outcomes['procedures'].append(procedure_target)
                        
                        # Move to next month
                        current_date = current_date + pd.DateOffset(months=1)
                    
                    update_processed += 1
                    print(f"    Generated {month_records} new timeline records for patient {patient_id}")
                    
                    # Progress reporting
                    if update_processed % 5 == 0:
                        print(f"  Updated {update_processed}/{len(update_patients)} patients, {processed_records} total new records")
                    
                except Exception as e:
                    print(f"  Error updating existing patient {patient_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Now load the complete CSV to return as a DataFrame
        print(f"\nLoading complete timeline from {output_csv}")
        combined_df = pd.read_csv(output_csv, low_memory=False)

        
        combined_df['TIME_POINT'] = pd.to_datetime(combined_df['TIME_POINT'], format='mixed', errors='coerce')
        
        # Convert targets to numpy arrays
        targets = self._extract_targets_from_existing_timeline(combined_df)
        
        print(f"Processed {processed_records} new records. Total timeline now has {len(combined_df)} records.")
        
        return combined_df, targets
        
    
    def _merge_targets(self, existing_df, new_df, existing_targets, new_targets):
        """
        Merge target arrays from existing and new data
        
        Args:
            existing_df: DataFrame with existing timeline data
            new_df: DataFrame with new timeline data
            existing_targets: Dictionary with existing target arrays
            new_targets: Dictionary with new target arrays
            
        Returns:
            Dictionary with merged target arrays
        """
        import numpy as np
        
        # If either is empty, return the other
        if len(existing_df) == 0:
            return new_targets
        if len(new_df) == 0:
            return existing_targets
        
        # Merge targets
        merged_targets = {}
        
        for key in existing_targets:
            if key in new_targets and len(new_targets[key]) > 0:
                merged_targets[key] = np.concatenate([existing_targets[key], new_targets[key]])
            else:
                merged_targets[key] = existing_targets[key]
        
        return merged_targets
    
    def run_pipeline(self, condition_keywords=None, force_new_timelines=False, append_new_months=True):
        """
        Run the complete ML pipeline for enhanced health predictions
        
        Args:
            condition_keywords: List of keywords to identify target patients (default: cerebral palsy)
            force_new_timelines: Whether to force creation of new timelines even if existing data is found
            append_new_months: Whether to append new months to existing patients
            
        Returns:
            Dictionary with pipeline results or False if pipeline failed
        """
        print("Starting enhanced health prediction pipeline")
        
        # 1. Load data
        print("\n=== Loading Data ===")
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
        
        # 2. Identify target patients
        print("\n=== Identifying Target Patients ===")
        target_patient_ids = self.identify_target_patients(condition_keywords)
        
        if len(target_patient_ids) == 0:
            print("No target patients found. Exiting.")
            return False
        
        # 3. Create or update enhanced patient timelines
        print("\n=== Creating/Updating Enhanced Patient Timelines ===")
        timeline_df, targets = self.load_or_append_patient_timelines(
            target_patient_ids, 
            force_new=force_new_timelines,
            append_new_months=append_new_months
        )
        print("Target Columns (from run pipeline):\n", targets)
        if len(timeline_df) == 0:
            print("No timeline data generated. Exiting.")
            return False
        
        # 4. Train prediction models
        print("\n=== Training Prediction Models ===")
        model_results = self.train_models(timeline_df, targets)
        
        # 5. Save models
        print("\n=== Saving Models ===")
        self.save_models(model_results)
        
        # 6. Generate clinical insights
        print("\n=== Generating Clinical Insights ===")
        clinical_insights = self.generate_clinical_insights(model_results)
        
        # Save insights to JSON
        insights_path = os.path.join('patient_data', f'clinical_insights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(insights_path, 'w') as f:
            json.dump(clinical_insights, f, indent=2)
        
        print(f"Saved clinical insights to {insights_path}")
        
        # Display key insights summary
        print("\n=== Clinical Insights Summary ===")
        
        print("\nTop Emergency Risk Factors:")
        for i, factor in enumerate(clinical_insights['emergency_risk_factors'][:5]):
            print(f"{i+1}. {factor['factor']} (Importance: {factor['importance']:.4f})")
        
        print("\nMost Common Predicted Diagnoses:")
        for i, diagnosis in enumerate(clinical_insights['common_diagnoses'][:5]):
            print(f"{i+1}. {diagnosis['diagnosis']} (Probability: {diagnosis['probability']:.4f})")
        
        print("\nTop Recommended Medications:")
        for i, med in enumerate(clinical_insights['recommended_medications'][:5]):
            print(f"{i+1}. {med['medication']} (Probability: {med['probability']:.4f})")
        
        print("\nTop Suggested Procedures:")
        for i, proc in enumerate(clinical_insights['suggested_procedures'][:5]):
            print(f"{i+1}. {proc['procedure']} (Probability: {proc['probability']:.4f})")
        
        print("\n=== Pipeline Complete ===")
        
        return {
            'model_results': model_results,
            'clinical_insights': clinical_insights,
            'timeline_df': timeline_df
        }
        
    def continue_training(self, new_patient_ids=None, fine_tuning=True, force_new=False, append_new_months=True):
        """
        Continue training existing models with new data
        
        Args:
            new_patient_ids: List of patient IDs to process (or None to use all target patients)
            fine_tuning: If True, use a lower learning rate for fine-tuning
            force_new: Whether to force creation of new timelines even if existing data is found
            append_new_months: Whether to generate new months for existing patients
            
        Returns:
            Dictionary with updated model results
        """
        import os
        
        # Check if models are already trained
        if any(model is None for model in self.models.values()):
            print("No existing models found. Run the full pipeline first.")
            return self.run_pipeline(force_new_timelines=force_new, append_new_months=append_new_months)
        
        # Load data
        print("\n=== Loading Data ===")
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
        
        # Identify target patients if not provided
        if new_patient_ids is None:
            print("\n=== Identifying Target Patients ===")
            new_patient_ids = self.identify_target_patients()
        
        if len(new_patient_ids) == 0:
            print("No target patients found. Exiting.")
            return False
        
        # Load or append patient timelines
        print("\n=== Loading/Updating Patient Timelines ===")
        timeline_df, targets = self.load_or_append_patient_timelines(
            new_patient_ids,
            force_new=force_new,
            append_new_months=append_new_months
        )
        
        if len(timeline_df) == 0:
            print("No timeline data available. Exiting.")
            return False
        
        # Prepare features
        print("\n=== Preparing Features ===")
        features = self.prepare_features(timeline_df)
        
        X = features['X_processed']
        
        updated_results = {}
        
        # Continue training each model
        for model_type, target_array in targets.items():
            if model_type not in self.models or self.models[model_type] is None:
                print(f"No existing {model_type} model found. Skipping.")
                continue
                
            if len(target_array) == 0:
                print(f"No target data for {model_type}. Skipping.")
                continue
                
            print(f"\n=== Continuing Training for {model_type} Model ===")
            model = self.models[model_type]
            
            # Use reduced learning rate for fine-tuning if requested
            if fine_tuning:
                from tensorflow import keras
                import tensorflow as tf
                
                # Get current learning rate
                if hasattr(model.optimizer, 'learning_rate'):
                    current_lr = float(model.optimizer.learning_rate.numpy())
                    # Reduce by factor of 5
                    new_lr = current_lr / 5.0
                    print(f"Reducing learning rate from {current_lr} to {new_lr} for fine-tuning")
                    
                    # Set new learning rate
                    model.optimizer.learning_rate = new_lr
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, target_array, test_size=0.2, random_state=42
            )
            
            # Continue training
            if model_type == 'emergency_visit':
                # Add class weighting for imbalanced binary classification
                pos_count = sum(y_train)
                total_count = len(y_train)
                class_weight = {0: 1., 1: (total_count - pos_count) / pos_count} if pos_count > 0 else None
                
                history = model.fit(
                    X_train, y_train,
                    epochs=25,  # Fewer epochs for fine-tuning
                    batch_size=32,
                    validation_split=0.2,
                    class_weight=class_weight,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_recall',
                            patience=5,
                            restore_best_weights=True
                        )
                    ],
                    verbose=1
                )
            else:
                # For multi-label models
                history = model.fit(
                    X_train, y_train,
                    epochs=25,  # Fewer epochs for fine-tuning
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        )
                    ],
                    verbose=1
                )
            
            # Evaluate updated model
            evaluation = model.evaluate(X_test, y_test)
            metrics = {name: value for name, value in zip(model.metrics_names, evaluation)}
            
            # Store updated model and metrics
            self.models[model_type] = model
            
            # Include categories for multi-label models
            categories = []
            if model_type == 'diagnosis':
                categories = self.diagnosis_categories
            elif model_type == 'medication':
                categories = self.medication_categories
            elif model_type == 'procedure':
                categories = self.procedure_categories
            
            updated_results[model_type] = {
                'model': model,
                'history': history.history,
                'metrics': metrics,
                'y_test': y_test,
                'y_pred': model.predict(X_test),
                'categories': categories
            }
            
            print(f"Updated {model_type} model. New metrics: {metrics}")
        
        # Save updated models
        print("\n=== Saving Updated Models ===")
        self.save_models(updated_results)
        
        # Generate updated clinical insights
        print("\n=== Generating Updated Clinical Insights ===")
        clinical_insights = self.generate_clinical_insights(updated_results)
        
        # Save insights to JSON
        insights_path = os.path.join('patient_data', f'clinical_insights_updated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(insights_path, 'w') as f:
            json.dump(clinical_insights, f, indent=2)
        
        print(f"Saved updated clinical insights to {insights_path}")
        
        return {
            'model_results': updated_results,
            'clinical_insights': clinical_insights,
            'timeline_df': timeline_df
        }
    
    def predict_for_patient(self, patient_id, lookback_months=12):
        """
        Make predictions for a specific patient
        
        Args:
            patient_id: Patient ID to make predictions for
            lookback_months: Number of months to look back for features
            
        Returns:
            Dictionary with predictions
        """
        # Check if models are trained
        if any(model is None for model in self.models.values()):
            print("Models are not trained. Run the pipeline first.")
            return None
        
        # Get patient info
        try:
            patient_info = self.patients_df[self.patients_df['Id'] == patient_id].iloc[0]
        except (IndexError, KeyError):
            print(f"Patient {patient_id} not found")
            return None
        
        # Get the latest date for this patient
        patient_encounters = self.encounters_df[self.encounters_df['PATIENT'] == patient_id]
        if patient_encounters.empty:
            print(f"No encounters found for patient {patient_id}")
            return None
        
        latest_date = patient_encounters['START'].max()
        
        # Define the lookback period
        lookback_start = latest_date - pd.DateOffset(months=lookback_months)
        
        # Extract features for this time period
        clinical_features = self.extract_clinical_features(patient_id, lookback_start, latest_date)
        
        # Create demographic features
        demographic_features = {}
        
        # Calculate age
        if pd.notna(patient_info['BIRTHDATE']):
            age_days = (latest_date - patient_info['BIRTHDATE']).days
            demographic_features['AGE_YEARS'] = age_days / 365.25
        else:
            demographic_features['AGE_YEARS'] = None
        
        # Gender, Race, Ethnicity
        demographic_features['GENDER'] = patient_info['GENDER'] if pd.notna(patient_info['GENDER']) else 'Unknown'
        demographic_features['RACE'] = patient_info['RACE'] if pd.notna(patient_info['RACE']) else 'Unknown'
        demographic_features['ETHNICITY'] = patient_info['ETHNICITY'] if pd.notna(patient_info['ETHNICITY']) else 'Unknown'
        
        # Calculate utilization metrics
        utilization_features = self.calculate_utilization_metrics(patient_id, lookback_start, latest_date)
        
        # Combine all features
        features = {
            # Add demographic features
            **{f'DEM_{k}': v for k, v in demographic_features.items()},
            # Add clinical features
            **clinical_features,
            # Add utilization features
            **utilization_features
        }
        
        # Convert to DataFrame for preprocessing
        features_df = pd.DataFrame([features])
        
        # Apply preprocessor
        if 'main' not in self.preprocessors:
            print("Preprocessor not found. Run the pipeline first.")
            return None
        
        try:
            X_processed = self.preprocessors['main'].transform(features_df)
            
            # Make predictions
            predictions = {}
            
            # Emergency visit prediction
            if self.models['emergency_visit'] is not None:
                emergency_risk = float(self.models['emergency_visit'].predict(X_processed)[0][0])
                predictions['emergency_risk'] = emergency_risk
                predictions['emergency_risk_level'] = 'High' if emergency_risk > 0.7 else 'Medium' if emergency_risk > 0.3 else 'Low'
            
            # Diagnosis prediction
            if self.models['diagnosis'] is not None and len(self.diagnosis_categories) > 0:
                diagnosis_probs = self.models['diagnosis'].predict(X_processed)[0]
                
                # Get top 5 diagnoses
                top_diagnoses = []
                for i in np.argsort(-diagnosis_probs)[:5]:
                    if diagnosis_probs[i] > 0.1:  # Only include if probability > 10%
                        top_diagnoses.append({
                            'diagnosis': self.diagnosis_categories[i],
                            'probability': float(diagnosis_probs[i])
                        })
                
                predictions['top_diagnoses'] = top_diagnoses
            
            # Medication prediction
            if self.models['medication'] is not None and len(self.medication_categories) > 0:
                medication_probs = self.models['medication'].predict(X_processed)[0]
                
                # Get top 5 medications
                top_medications = []
                for i in np.argsort(-medication_probs)[:5]:
                    if medication_probs[i] > 0.1:  # Only include if probability > 10%
                        top_medications.append({
                            'medication': self.medication_categories[i],
                            'probability': float(medication_probs[i])
                        })
                
                predictions['recommended_medications'] = top_medications
            
            # Procedure prediction
            if self.models['procedure'] is not None and len(self.procedure_categories) > 0:
                procedure_probs = self.models['procedure'].predict(X_processed)[0]
                
                # Get top 5 procedures
                top_procedures = []
                for i in np.argsort(-procedure_probs)[:5]:
                    if procedure_probs[i] > 0.1:  # Only include if probability > 10%
                        top_procedures.append({
                            'procedure': self.procedure_categories[i],
                            'probability': float(procedure_probs[i])
                        })
                
                predictions['suggested_procedures'] = top_procedures
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def predict_batch(self, patient_ids, lookback_months=12):
        """
        Make predictions for a batch of patients
        
        Args:
            patient_ids: List of patient IDs to make predictions for
            lookback_months: Number of months to look back for features
            
        Returns:
            Dictionary with predictions for each patient
        """
        predictions = {}
        
        for patient_id in patient_ids:
            patient_predictions = self.predict_for_patient(patient_id, lookback_months)
            if patient_predictions:
                predictions[patient_id] = patient_predictions
        
        return predictions
    
    def save_training_history(self, results, output_dir='metrics'):
        """
        Save training history metrics to CSV files and create visualization plots
        
        Args:
            results: Dictionary with model results
            output_dir: Directory to save metrics and plots
        """
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Loop through each model type
        for model_type, result in results.items():
            if 'history' not in result:
                print(f"No training history found for {model_type} model. Skipping.")
                continue
            
            # Get training history
            history = result['history']
            
            # Create DataFrame from history
            history_df = pd.DataFrame(history)
            
            # Add epoch column
            history_df['epoch'] = range(1, len(history_df) + 1)
            
            # Save history to CSV
            history_path = os.path.join(output_dir, f'{model_type}_training_history_{timestamp}.csv')
            history_df.to_csv(history_path, index=False)
            print(f"Saved {model_type} training history to {history_path}")
            
            # Create plots directory
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create plots for different metrics
            
            # 1. Loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(history_df['epoch'], history_df['loss'], label='Training Loss')
            if 'val_loss' in history_df.columns:
                plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
            plt.title(f'{model_type.replace("_", " ").title()} Model: Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            loss_plot_path = os.path.join(plots_dir, f'{model_type}_loss_{timestamp}.png')
            plt.savefig(loss_plot_path)
            plt.close()
            
            # 2. Accuracy plot
            if 'accuracy' in history_df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(history_df['epoch'], history_df['accuracy'], label='Training Accuracy')
                if 'val_accuracy' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
                plt.title(f'{model_type.replace("_", " ").title()} Model: Accuracy Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                acc_plot_path = os.path.join(plots_dir, f'{model_type}_accuracy_{timestamp}.png')
                plt.savefig(acc_plot_path)
                plt.close()
            
            # 3. AUC plot
            if 'auc' in history_df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(history_df['epoch'], history_df['auc'], label='Training AUC')
                if 'val_auc' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['val_auc'], label='Validation AUC')
                plt.title(f'{model_type.replace("_", " ").title()} Model: AUC Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('AUC')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                auc_plot_path = os.path.join(plots_dir, f'{model_type}_auc_{timestamp}.png')
                plt.savefig(auc_plot_path)
                plt.close()
            
            # 4. Precision plot
            if 'precision' in history_df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(history_df['epoch'], history_df['precision'], label='Training Precision')
                if 'val_precision' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['val_precision'], label='Validation Precision')
                plt.title(f'{model_type.replace("_", " ").title()} Model: Precision Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Precision')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                precision_plot_path = os.path.join(plots_dir, f'{model_type}_precision_{timestamp}.png')
                plt.savefig(precision_plot_path)
                plt.close()
            
            # 5. Recall plot
            if 'recall' in history_df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(history_df['epoch'], history_df['recall'], label='Training Recall')
                if 'val_recall' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['val_recall'], label='Validation Recall')
                plt.title(f'{model_type.replace("_", " ").title()} Model: Recall Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Recall')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                recall_plot_path = os.path.join(plots_dir, f'{model_type}_recall_{timestamp}.png')
                plt.savefig(recall_plot_path)
                plt.close()
            
            # 6. Combined metrics plot
            if any(metric in history_df.columns for metric in ['accuracy', 'auc', 'precision', 'recall']):
                plt.figure(figsize=(12, 8))
                metrics_to_plot = []
                
                if 'accuracy' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['accuracy'], label='Accuracy')
                    metrics_to_plot.append('accuracy')
                
                if 'auc' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['auc'], label='AUC')
                    metrics_to_plot.append('auc')
                
                if 'precision' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['precision'], label='Precision')
                    metrics_to_plot.append('precision')
                
                if 'recall' in history_df.columns:
                    plt.plot(history_df['epoch'], history_df['recall'], label='Recall')
                    metrics_to_plot.append('recall')
                
                plt.title(f'{model_type.replace("_", " ").title()} Model: Performance Metrics')
                plt.xlabel('Epoch')
                plt.ylabel('Metric Value')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                combined_plot_path = os.path.join(plots_dir, f'{model_type}_combined_metrics_{timestamp}.png')
                plt.savefig(combined_plot_path)
                plt.close()
        
            # Create final evaluation metrics comparison
            self._create_model_comparison_plots(results, plots_dir, timestamp)
            
            print(f"Created and saved metric plots in {plots_dir}")


    def _create_model_comparison_plots(self, results, plots_dir, timestamp):
        """
        Create plots comparing final evaluation metrics across models
        
        Args:
            results: Dictionary with model results
            plots_dir: Directory to save plots
            timestamp: Timestamp string for file naming
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import numpy as np
        
        # Extract final evaluation metrics for each model
        metrics_data = []
        
        for model_type, result in results.items():
            if 'metrics' in result:
                model_metrics = result['metrics']
                model_row = {'model_type': model_type.replace('_', ' ').title()}
                
                for metric_name, metric_value in model_metrics.items():
                    model_row[metric_name] = metric_value
                
                metrics_data.append(model_row)
        
        if not metrics_data:
            print("No evaluation metrics found for comparison plots.")
            return
        
        # Create DataFrame with metrics
        metrics_df = pd.DataFrame(metrics_data)
        
        # List of metrics to compare (if present in the data)
        metrics_to_compare = ['accuracy', 'loss', 'auc', 'precision', 'recall']
        available_metrics = [m for m in metrics_to_compare if m in metrics_df.columns]
        
        if not available_metrics:
            print("No common metrics found for comparison plots.")
            return
        
        # Create comparison bar chart for each metric
        for metric in available_metrics:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='model_type', y=metric, data=metrics_df)
            
            # Add value labels on top of bars
            for i, bar in enumerate(ax.patches):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.4f}',
                    ha='center',
                    fontsize=10
                )
            
            plt.title(f'Comparison of {metric.title()} Across Models')
            plt.xlabel('Model Type')
            plt.ylabel(metric.title())
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save the plot
            comparison_path = os.path.join(plots_dir, f'model_comparison_{metric}_{timestamp}.png')
            plt.savefig(comparison_path)
            plt.close()
        
        # Create radar chart for model comparison if we have multiple metrics
        if len(available_metrics) >= 3 and len(metrics_df) >= 2:
            self._create_radar_chart_comparison(metrics_df, available_metrics, plots_dir, timestamp)


    def _create_radar_chart_comparison(self, metrics_df, available_metrics, plots_dir, timestamp):
        """
        Create a radar chart comparing models across multiple metrics
        
        Args:
            metrics_df: DataFrame with model metrics
            available_metrics: List of metrics to include in the chart
            plots_dir: Directory to save plots
            timestamp: Timestamp string for file naming
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # Number of metrics to plot
        N = len(available_metrics)
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Angle for each metric (evenly spaced)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Close the loop
        angles += angles[:1]
        
        # For each model, add a line to the radar chart
        for i, (index, row) in enumerate(metrics_df.iterrows()):
            # Get values for this model
            values = [row[metric] for metric in available_metrics]
            
            # Close the loop
            values += values[:1]
            
            # Plot the line
            ax.plot(angles, values, linewidth=2, label=row['model_type'])
            
            # Fill the area
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], [metric.title() for metric in available_metrics])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Radar Chart: Model Performance Comparison', size=15, y=1.1)
        
        # Save the radar chart
        radar_path = os.path.join(plots_dir, f'model_comparison_radar_{timestamp}.png')
        plt.savefig(radar_path, bbox_inches='tight')
        plt.close()

    def train_optimized_models(self, timeline_df, targets):
        """
        Train prediction models with hyperparameter optimization
        
        Args:
            timeline_df: DataFrame with patient timeline features
            targets: Dictionary with target arrays for each prediction task
            
        Returns:
            Dictionary with optimized models and their evaluation metrics
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        
        # Create output directories
        os.makedirs('hyperparameter_tuning', exist_ok=True)
        os.makedirs('feature_selection', exist_ok=True)
        
        # Prepare features with explicit exclusion of non-feature columns
        exclude_cols = [
            'PATIENT_ID', 'TIME_POINT', 
            'HAD_EMERGENCY_NEXT_MONTH', 'NEW_DIAGNOSES_NEXT_MONTH',
            'NEW_MEDICATIONS_NEXT_MONTH', 'NEW_PROCEDURES_NEXT_MONTH'
        ]
        
        print("Preparing features for optimized models...")
        features = self.prepare_features(timeline_df, exclude_cols)
        
        X = features['X_processed']
        print(f"Processed feature matrix shape: {X.shape}")
        
        # Ensure we have the correct feature names that match the processed feature matrix
        processed_feature_names = features['feature_names_processed']
        
        # Check if feature names match processed feature matrix dimensions
        if len(processed_feature_names) != X.shape[1]:
            print(f"WARNING: Processed feature names length ({len(processed_feature_names)}) doesn't match feature matrix ({X.shape[1]})")
            # Generate generic feature names that match the matrix dimensions
            processed_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        results = {}
        
        # ----------------- EMERGENCY VISIT MODEL -----------------
        if 'emergency_visit' in targets and len(targets['emergency_visit']) > 0:
            print("\nOptimizing emergency visit prediction model...")
            y_emergency = targets['emergency_visit']
            
            # Convert to boolean array if needed
            if y_emergency.dtype != bool and y_emergency.dtype != np.bool_:
                y_emergency = np.array([bool(val) if val != 'nan' else False for val in y_emergency], dtype=bool)
            
            # Perform enhanced feature selection
            print("Performing enhanced feature selection for emergency model...")
            feature_selection_results = self.enhanced_feature_selection(
                X, y_emergency, processed_feature_names, 
                n_features_range=(5, 50, 5),  # Select top 30 features
                method='ensemble'  # Use ensemble of methods for better selection
            )
            
            # Use only selected features
            X_selected = X[:, feature_selection_results['selected_indices']]
            selected_feature_names = feature_selection_results['selected_features']
            
            print(f"Selected {len(selected_feature_names)} features for emergency model")
            
            # Split data for model tuning
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_emergency, test_size=0.2, random_state=42,
                stratify=y_emergency  # Ensure balanced splits
            )
            
            # Further split train into train/validation for hyperparameter tuning
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
                stratify=y_train
            )
            
            # Run hyperparameter tuning
            print("Running hyperparameter tuning for emergency model...")
            tuning_results = self.tune_model_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type='binary',
                tuning_epochs=30,
                max_trials=20
            )
            
            # Get best model
            emergency_model = tuning_results['model']
            
            # Evaluate model on test set
            print("Evaluating final emergency model on test set...")
            emergency_eval = emergency_model.evaluate(X_test, y_test)
            emergency_metrics = {name: value for name, value in zip(emergency_model.metrics_names, emergency_eval)}
            
            # Get predictions
            y_pred_prob = emergency_model.predict(X_test).flatten()
            
            # Find optimal threshold
            from sklearn.metrics import precision_recall_curve, f1_score
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            # Get predictions at optimal threshold
            y_pred = (y_pred_prob >= best_threshold).astype(int)
            
            # Calculate F1 score manually
            f1 = f1_score(y_test, y_pred)
            emergency_metrics['f1_score'] = f1
            
            # Store results
            results['emergency_visit'] = {
                'model': emergency_model,
                'metrics': emergency_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'best_threshold': best_threshold,
                'selected_features': selected_feature_names,
                'feature_selection': feature_selection_results,
                'hyperparameter_tuning': tuning_results
            }
            
            print(f"Emergency visit model optimized with metrics: {emergency_metrics}")
        
        # Continue with other models (diagnosis, medication, procedure)...
        # [Code for other models would follow the same pattern]
        
        # Save preprocessing information
        self.preprocessors['main'] = features['preprocessor']
        
        # Update models dictionary
        for key, value in results.items():
            if 'model' in value:
                self.models[key] = value['model']
        
        # Save the training results
        self.save_optimized_models_results(results)
        
        return results

    def save_optimized_models_results(self, results, output_dir='optimized_models'):
        """
        Save optimized models, their hyperparameters, and performance metrics
        
        Args:
            results: Dictionary with optimized model results
            output_dir: Directory to save models and metadata
        """
        import os
        import json
        import joblib
        from datetime import datetime
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the preprocessing pipeline
        if 'main' in self.preprocessors:
            preprocessor_path = os.path.join(output_dir, f'preprocessor_{timestamp}.joblib')
            joblib.dump(self.preprocessors['main'], preprocessor_path)
            print(f"Saved preprocessor to {preprocessor_path}")
        
        # Save each model and its metadata
        for model_type, result in results.items():
            if 'model' not in result:
                continue
            
            # Save TensorFlow model
            model_path = os.path.join(output_dir, 'models', f"{model_type}_model_{timestamp}.keras")
            result['model'].save(model_path)
            
            # Create metadata object (without large numpy arrays)
            metadata = {
                'timestamp': timestamp,
                'metrics': {k: float(v) for k, v in result.get('metrics', {}).items()},
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'selected_features': result.get('selected_features', [])
            }
            
            # Add hyperparameters if available
            if 'hyperparameter_tuning' in result and 'best_hyperparameters' in result['hyperparameter_tuning']:
                metadata['hyperparameters'] = result['hyperparameter_tuning']['best_hyperparameters']
            
            # Add model-specific information
            if model_type == 'emergency_visit' and 'best_threshold' in result:
                metadata['best_threshold'] = float(result['best_threshold'])
            
            if model_type in ['diagnosis', 'medication', 'procedure'] and 'categories' in result:
                metadata['categories'] = result['categories']
            
            # Save metadata to JSON
            metadata_path = os.path.join(output_dir, 'metadata', f"{model_type}_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Saved {model_type} model to {model_path} with metadata")
        
        # Generate comparison report of model performance
        self.create_model_comparison_report(results, timestamp, output_dir)
        
        return True

    def create_model_comparison_report(self, results, timestamp, output_dir='optimized_models'):
        """
        Create a comprehensive comparison report of model performance with visualizations
        
        Args:
            results: Dictionary with model results
            timestamp: Timestamp string for file naming
            output_dir: Base directory to save report
        """
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create reports directory
        reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Extract metrics from all models
        models_metrics = []
        
        for model_type, result in results.items():
            if 'metrics' not in result:
                continue
            
            # Get metrics
            metrics = result['metrics']
            
            # Add model type
            metrics_row = {
                'model_type': model_type,
                **{k: float(v) for k, v in metrics.items()}
            }
            
            # Add hyperparameter info if available
            if 'hyperparameter_tuning' in result and 'best_hyperparameters' in result['hyperparameter_tuning']:
                hp = result['hyperparameter_tuning']['best_hyperparameters']
                
                # Add key hyperparameters
                if 'learning_rate' in hp:
                    metrics_row['learning_rate'] = hp['learning_rate']
                
                if 'units_1' in hp:
                    metrics_row['hidden_units'] = hp['units_1']
                
                if 'dropout_1' in hp:
                    metrics_row['dropout_rate'] = hp['dropout_1']
            
            # Add feature selection info
            if 'feature_selection' in result:
                metrics_row['n_features'] = len(result.get('selected_features', []))
                metrics_row['feature_selection_method'] = result['feature_selection'].get('method', 'unknown')
            
            models_metrics.append(metrics_row)
        
        if not models_metrics:
            print("No metrics available for comparison report")
            return
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(models_metrics)
        
        # Save metrics to CSV
        metrics_file = os.path.join(reports_dir, f"model_metrics_comparison_{timestamp}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        # Create comparison visualizations
        
        # 1. Bar chart of key metrics for each model
        # Identify metrics common to all models
        common_metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in common_metrics if m in metrics_df.columns]
        
        if available_metrics:
            plt.figure(figsize=(14, 8))
            metrics_df_melted = pd.melt(
                metrics_df, 
                id_vars=['model_type'], 
                value_vars=available_metrics,
                var_name='Metric', 
                value_name='Value'
            )
            
            ax = sns.barplot(x='model_type', y='Value', hue='Metric', data=metrics_df_melted)
            plt.title('Model Performance Comparison', fontsize=16)
            plt.xlabel('Model Type', fontsize=14)
            plt.ylabel('Metric Value', fontsize=14)
            plt.legend(title='Metric', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(reports_dir, f"model_comparison_chart_{timestamp}.png"))
            plt.close()
        
        # 2. Create heatmap of metrics
        if len(metrics_df) > 1 and len(available_metrics) > 1:
            # Set model_type as index
            metrics_heatmap_df = metrics_df.set_index('model_type')[available_metrics]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics_heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title('Model Metrics Heatmap', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, f"metrics_heatmap_{timestamp}.png"))
            plt.close()
        
        # 3. Create ROC curves comparison for binary models (if applicable)
        binary_models = ['emergency_visit']
        has_binary_models = any(model in results for model in binary_models)
        
        if has_binary_models:
            from sklearn.metrics import roc_curve, auc
            
            plt.figure(figsize=(10, 8))
            
            for model_type in binary_models:
                if model_type in results and 'y_test' in results[model_type] and 'y_pred_prob' in results[model_type]:
                    result = results[model_type]
                    y_test = result['y_test']
                    y_pred_prob = result['y_pred_prob']
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, lw=2, label=f'{model_type} (AUC = {roc_auc:.3f})')
            
            # Add diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('ROC Curves Comparison', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(reports_dir, f"roc_curves_comparison_{timestamp}.png"))
            plt.close()
        
        # 4. Create feature importance plots for each model
        for model_type, result in results.items():
            if 'feature_selection' not in result or 'selected_features' not in result:
                continue
            
            feature_selection = result['feature_selection']
            
            if 'importance_scores' not in feature_selection:
                continue
            
            importance_scores = feature_selection['importance_scores']
            
            # Convert to DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(importance_scores.keys()),
                'Importance': list(importance_scores.values())
            }).sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            n_features = min(20, len(importance_df))
            plt.figure(figsize=(12, 10))
            
            sns.barplot(y='Feature', x='Importance', data=importance_df.head(n_features))
            plt.title(f'Top {n_features} Important Features for {model_type.replace("_", " ").title()} Model', fontsize=16)
            plt.xlabel('Importance Score', fontsize=14)
            plt.tight_layout()
            
            plt.savefig(os.path.join(reports_dir, f"{model_type}_feature_importance_{timestamp}.png"))
            plt.close()
        
        # 5. Create a hyperparameters comparison table
        hp_rows = []
        
        for model_type, result in results.items():
            if 'hyperparameter_tuning' not in result or 'best_hyperparameters' not in result['hyperparameter_tuning']:
                continue
            
            hp = result['hyperparameter_tuning']['best_hyperparameters']
            
            hp_row = {'model_type': model_type}
            
            # Add key hyperparameters
            for param in ['learning_rate', 'units_1', 'units_2', 'dropout_1', 'dropout_2', 
                        'activation_1', 'optimizer', 'l2_1', 'batch_norm_1']:
                if param in hp:
                    hp_row[param] = hp[param]
            
            hp_rows.append(hp_row)
        
        if hp_rows:
            hp_df = pd.DataFrame(hp_rows)
            hp_file = os.path.join(reports_dir, f"hyperparameters_comparison_{timestamp}.csv")
            hp_df.to_csv(hp_file, index=False)
        
        print(f"Created model comparison report in {reports_dir}")

    def run_optimized_pipeline(self, condition_keywords=None, force_new_timelines=False, append_new_months=True):
        """
        Run the optimized ML pipeline for health predictions with hyperparameter tuning
        
        Args:
            condition_keywords: List of keywords to identify target patients
            force_new_timelines: Whether to force creation of new timelines even if existing data is found
            append_new_months: Whether to append new months to existing patients
            
        Returns:
            Dictionary with pipeline results or False if pipeline failed
        """
        print("Starting optimized health prediction pipeline with hyperparameter tuning")
        
        # 1. Load data
        print("\n=== Loading Data ===")
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
        
        # 2. Identify target patients
        print("\n=== Identifying Target Patients ===")
        target_patient_ids = self.identify_target_patients(condition_keywords)
        
        if len(target_patient_ids) == 0:
            print("No target patients found. Exiting.")
            return False
        
        # 3. Create or update enhanced patient timelines
        print("\n=== Creating/Updating Enhanced Patient Timelines ===")
        timeline_df, targets = self.load_or_append_patient_timelines(
            target_patient_ids, 
            force_new=force_new_timelines,
            append_new_months=append_new_months
        )
        
        if len(timeline_df) == 0:
            print("No timeline data generated. Exiting.")
            return False
        
        # 4. Train optimized prediction models with hyperparameter tuning
        print("\n=== Training Optimized Prediction Models ===")
        model_results = self.train_optimized_models(timeline_df, targets)
        
        # 5. Generate clinical insights
        print("\n=== Generating Clinical Insights ===")
        clinical_insights = self.generate_clinical_insights(model_results)
        
        # Save insights to JSON
        import os
        import json
        from datetime import datetime
        
        insights_path = os.path.join('optimized_models', f'clinical_insights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(insights_path, 'w') as f:
            json.dump(clinical_insights, f, indent=2)
        
        print(f"Saved clinical insights to {insights_path}")
        
        # Display key insights summary
        print("\n=== Clinical Insights Summary ===")
        
        print("\nTop Emergency Risk Factors:")
        for i, factor in enumerate(clinical_insights['emergency_risk_factors'][:5]):
            print(f"{i+1}. {factor['factor']} (Importance: {factor['importance']:.4f})")
        
        print("\nMost Common Predicted Diagnoses:")
        for i, diagnosis in enumerate(clinical_insights['common_diagnoses'][:5]):
            print(f"{i+1}. {diagnosis['diagnosis']} (Probability: {diagnosis['probability']:.4f})")
        
        print("\nTop Recommended Medications:")
        for i, med in enumerate(clinical_insights['recommended_medications'][:5]):
            print(f"{i+1}. {med['medication']} (Probability: {med['probability']:.4f})")
        
        print("\nTop Suggested Procedures:")
        for i, proc in enumerate(clinical_insights['suggested_procedures'][:5]):
            print(f"{i+1}. {proc['procedure']} (Probability: {proc['probability']:.4f})")
        
        print("\n=== Optimized Pipeline Complete ===")
        
        return {
            'model_results': model_results,
            'clinical_insights': clinical_insights,
            'timeline_df': timeline_df
        }
    
    def train_optimized_models(self, timeline_df, targets):
        """
        Train prediction models with hyperparameter optimization
        
        Args:
            timeline_df: DataFrame with patient timeline features
            targets: Dictionary with target arrays for each prediction task
            
        Returns:
            Dictionary with optimized models and their evaluation metrics
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        
        # Create output directories
        os.makedirs('hyperparameter_tuning', exist_ok=True)
        os.makedirs('feature_selection', exist_ok=True)
        
        # Prepare features
        print("Preparing features for optimized models...")
        features = self.prepare_features(timeline_df)
        
        X = features['X_processed']
        results = {}
        
        # ----------------- EMERGENCY VISIT MODEL -----------------
        if 'emergency_visit' in targets and len(targets['emergency_visit']) > 0:
            print("\nOptimizing emergency visit prediction model...")
            y_emergency = targets['emergency_visit']
            
            # Convert to boolean array if needed
            if y_emergency.dtype != bool and y_emergency.dtype != np.bool_:
                y_emergency = np.array([bool(val) if val != 'nan' else False for val in y_emergency], dtype=bool)
            
            # Perform enhanced feature selection
            print("Performing enhanced feature selection for emergency model...")
            feature_selection_results = self.enhanced_feature_selection(
                X, y_emergency, features['feature_names'], 
                n_features_range=(5, 50, 5),  # Select top 30 features
                method='ensemble'  # Use ensemble of methods for better selection
            )
            
            # Use only selected features
            X_selected = X[:, feature_selection_results['selected_indices']]
            selected_feature_names = feature_selection_results['selected_features']
            
            print(f"Selected {len(selected_feature_names)} features for emergency model")
            
            # Split data for model tuning
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_emergency, test_size=0.2, random_state=42,
                stratify=y_emergency  # Ensure balanced splits
            )
            
            # Further split train into train/validation for hyperparameter tuning
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
                stratify=y_train
            )
            
            # Run hyperparameter tuning
            print("Running hyperparameter tuning for emergency model...")
            tuning_results = self.tune_model_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type='binary',
                tuning_epochs=30,
                max_trials=20
            )
            
            # Get best model
            emergency_model = tuning_results['model']
            
            # Evaluate model on test set
            print("Evaluating final emergency model on test set...")
            emergency_eval = emergency_model.evaluate(X_test, y_test)
            emergency_metrics = {name: value for name, value in zip(emergency_model.metrics_names, emergency_eval)}
            
            # Get predictions
            y_pred_prob = emergency_model.predict(X_test).flatten()
            
            # Find optimal threshold
            from sklearn.metrics import precision_recall_curve, f1_score
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            # Get predictions at optimal threshold
            y_pred = (y_pred_prob >= best_threshold).astype(int)
            
            # Calculate F1 score manually
            f1 = f1_score(y_test, y_pred)
            emergency_metrics['f1_score'] = f1
            
            # Store results
            results['emergency_visit'] = {
                'model': emergency_model,
                'metrics': emergency_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'best_threshold': best_threshold,
                'selected_features': selected_feature_names,
                'feature_selection': feature_selection_results,
                'hyperparameter_tuning': tuning_results
            }
            
            print(f"Emergency visit model optimized with metrics: {emergency_metrics}")
        
        # ----------------- DIAGNOSIS MODEL -----------------
        if 'diagnoses' in targets and len(targets['diagnoses']) > 0 and len(self.diagnosis_categories) > 0:
            print("\nOptimizing diagnosis prediction model...")
            y_diagnosis = targets['diagnoses']
            
            # Perform enhanced feature selection
            print("Performing enhanced feature selection for diagnosis model...")
            feature_selection_results = self.enhanced_feature_selection(
                X, y_diagnosis, features['feature_names'], 
                n_features_range=(5, 50, 5),  # Select more features for multi-label
                method='ensemble'
            )
            
            # Use selected features
            X_selected = X[:, feature_selection_results['selected_indices']]
            selected_feature_names = feature_selection_results['selected_features']
            
            print(f"Selected {len(selected_feature_names)} features for diagnosis model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_diagnosis, test_size=0.2, random_state=42
            )
            
            # Further split for hyperparameter tuning
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Run hyperparameter tuning
            print("Running hyperparameter tuning for diagnosis model...")
            tuning_results = self.tune_model_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type='multi_label',
                tuning_epochs=30,
                max_trials=20
            )
            
            # Get best model
            diagnosis_model = tuning_results['model']
            
            # Evaluate model on test set
            print("Evaluating final diagnosis model on test set...")
            diagnosis_eval = diagnosis_model.evaluate(X_test, y_test)
            diagnosis_metrics = {name: value for name, value in zip(diagnosis_model.metrics_names, diagnosis_eval)}
            
            # Get predictions
            y_pred_prob = diagnosis_model.predict(X_test)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate F1 scores manually
            from sklearn.metrics import f1_score
            f1_scores = []
            for i in range(y_test.shape[1]):
                f1 = f1_score(y_test[:, i], y_pred[:, i])
                f1_scores.append(f1)
            
            # Average F1 score
            avg_f1 = np.mean(f1_scores)
            diagnosis_metrics['f1_score'] = avg_f1
            
            # Store results
            results['diagnosis'] = {
                'model': diagnosis_model,
                'metrics': diagnosis_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'categories': self.diagnosis_categories,
                'selected_features': selected_feature_names,
                'feature_selection': feature_selection_results,
                'hyperparameter_tuning': tuning_results
            }
            
            print(f"Diagnosis model optimized with metrics: {diagnosis_metrics}")
        
        # ----------------- MEDICATION MODEL -----------------
        if 'medications' in targets and len(targets['medications']) > 0 and len(self.medication_categories) > 0:
            print("\nOptimizing medication prediction model...")
            y_medication = targets['medications']
            
            # Perform enhanced feature selection
            print("Performing enhanced feature selection for medication model...")
            feature_selection_results = self.enhanced_feature_selection(
                X, y_medication, features['feature_names'], 
                n_features_range=(5, 50, 5),
                method='ensemble'
            )
            
            # Use selected features
            X_selected = X[:, feature_selection_results['selected_indices']]
            selected_feature_names = feature_selection_results['selected_features']
            
            print(f"Selected {len(selected_feature_names)} features for medication model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_medication, test_size=0.2, random_state=42
            )
            
            # Further split for hyperparameter tuning
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Run hyperparameter tuning
            print("Running hyperparameter tuning for medication model...")
            tuning_results = self.tune_model_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type='multi_label',
                tuning_epochs=30,
                max_trials=20
            )
            
            # Get best model
            medication_model = tuning_results['model']
            
            # Evaluate model on test set
            print("Evaluating final medication model on test set...")
            medication_eval = medication_model.evaluate(X_test, y_test)
            medication_metrics = {name: value for name, value in zip(medication_model.metrics_names, medication_eval)}
            
            # Get predictions
            y_pred_prob = medication_model.predict(X_test)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate F1 scores manually
            from sklearn.metrics import f1_score
            f1_scores = []
            for i in range(y_test.shape[1]):
                f1 = f1_score(y_test[:, i], y_pred[:, i])
                f1_scores.append(f1)
            
            # Average F1 score
            avg_f1 = np.mean(f1_scores)
            medication_metrics['f1_score'] = avg_f1
            
            # Store results
            results['medication'] = {
                'model': medication_model,
                'metrics': medication_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'categories': self.medication_categories,
                'selected_features': selected_feature_names,
                'feature_selection': feature_selection_results,
                'hyperparameter_tuning': tuning_results
            }
            
            print(f"Medication model optimized with metrics: {medication_metrics}")
        
        # ----------------- PROCEDURE MODEL -----------------
        if 'procedures' in targets and len(targets['procedures']) > 0 and len(self.procedure_categories) > 0:
            print("\nOptimizing procedure prediction model...")
            y_procedure = targets['procedures']
            
            # Perform enhanced feature selection
            print("Performing enhanced feature selection for procedure model...")
            feature_selection_results = self.enhanced_feature_selection(
                X, y_procedure, features['feature_names'], 
                n_features_range=(5, 50, 5),
                method='ensemble'
            )
            
            # Use selected features
            X_selected = X[:, feature_selection_results['selected_indices']]
            selected_feature_names = feature_selection_results['selected_features']
            
            print(f"Selected {len(selected_feature_names)} features for procedure model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_procedure, test_size=0.2, random_state=42
            )
            
            # Further split for hyperparameter tuning
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Run hyperparameter tuning
            print("Running hyperparameter tuning for procedure model...")
            tuning_results = self.tune_model_hyperparameters(
                X_train, y_train, X_val, y_val,
                model_type='multi_label',
                tuning_epochs=30,
                max_trials=20
            )
            
            # Get best model
            procedure_model = tuning_results['model']
            
            # Evaluate model on test set
            print("Evaluating final procedure model on test set...")
            procedure_eval = procedure_model.evaluate(X_test, y_test)
            procedure_metrics = {name: value for name, value in zip(procedure_model.metrics_names, procedure_eval)}
            
            # Get predictions
            y_pred_prob = procedure_model.predict(X_test)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            # Calculate F1 scores manually
            from sklearn.metrics import f1_score
            f1_scores = []
            for i in range(y_test.shape[1]):
                f1 = f1_score(y_test[:, i], y_pred[:, i])
                f1_scores.append(f1)
            
            # Average F1 score
            avg_f1 = np.mean(f1_scores)
            procedure_metrics['f1_score'] = avg_f1
            
            # Store results
            results['procedure'] = {
                'model': procedure_model,
                'metrics': procedure_metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'categories': self.procedure_categories,
                'selected_features': selected_feature_names,
                'feature_selection': feature_selection_results,
                'hyperparameter_tuning': tuning_results
            }
            
            print(f"Procedure model optimized with metrics: {procedure_metrics}")
        
        # Save preprocessing information
        self.preprocessors['main'] = features['preprocessor']
        
        # Update models dictionary
        for key, value in results.items():
            if 'model' in value:
                self.models[key] = value['model']
        
        # Save the training results
        self.save_optimized_models_results(results)
        
        return results

    def save_optimized_models_results(self, results, output_dir='optimized_models'):
        """
        Save optimized models, their hyperparameters, and performance metrics
        
        Args:
            results: Dictionary with optimized model results
            output_dir: Directory to save models and metadata
        """
        import os
        import json
        import joblib
        from datetime import datetime
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the preprocessing pipeline
        if 'main' in self.preprocessors:
            preprocessor_path = os.path.join(output_dir, f'preprocessor_{timestamp}.joblib')
            joblib.dump(self.preprocessors['main'], preprocessor_path)
            print(f"Saved preprocessor to {preprocessor_path}")
        
        # Save each model and its metadata
        for model_type, result in results.items():
            if 'model' not in result:
                continue
            
            # Save TensorFlow model
            model_path = os.path.join(output_dir, 'models', f"{model_type}_model_{timestamp}.keras")
            result['model'].save(model_path)
            
            # Create metadata object (without large numpy arrays)
            metadata = {
                'timestamp': timestamp,
                'metrics': {k: float(v) for k, v in result.get('metrics', {}).items()},
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'selected_features': result.get('selected_features', [])
            }
            
            # Add hyperparameters if available
            if 'hyperparameter_tuning' in result and 'best_hyperparameters' in result['hyperparameter_tuning']:
                metadata['hyperparameters'] = result['hyperparameter_tuning']['best_hyperparameters']
            
            # Add model-specific information
            if model_type == 'emergency_visit' and 'best_threshold' in result:
                metadata['best_threshold'] = float(result['best_threshold'])
            
            if model_type in ['diagnosis', 'medication', 'procedure'] and 'categories' in result:
                metadata['categories'] = result['categories']
            
            # Save metadata to JSON
            metadata_path = os.path.join(output_dir, 'metadata', f"{model_type}_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Saved {model_type} model to {model_path} with metadata")
        
        # Generate comparison report of model performance
        self.create_model_comparison_report(results, timestamp, output_dir)
        
        return True

    def create_model_comparison_report(self, results, timestamp, output_dir='optimized_models'):
        """
        Create a comprehensive comparison report of model performance with visualizations
        
        Args:
            results: Dictionary with model results
            timestamp: Timestamp string for file naming
            output_dir: Base directory to save report
        """
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create reports directory
        reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Extract metrics from all models
        models_metrics = []
        
        for model_type, result in results.items():
            if 'metrics' not in result:
                continue
            
            # Get metrics
            metrics = result['metrics']
            
            # Add model type
            metrics_row = {
                'model_type': model_type,
                **{k: float(v) for k, v in metrics.items()}
            }
            
            # Add hyperparameter info if available
            if 'hyperparameter_tuning' in result and 'best_hyperparameters' in result['hyperparameter_tuning']:
                hp = result['hyperparameter_tuning']['best_hyperparameters']
                
                # Add key hyperparameters
                if 'learning_rate' in hp:
                    metrics_row['learning_rate'] = hp['learning_rate']
                
                if 'units_1' in hp:
                    metrics_row['hidden_units'] = hp['units_1']
                
                if 'dropout_1' in hp:
                    metrics_row['dropout_rate'] = hp['dropout_1']
            
            # Add feature selection info
            if 'feature_selection' in result:
                metrics_row['n_features'] = len(result.get('selected_features', []))
                metrics_row['feature_selection_method'] = result['feature_selection'].get('method', 'unknown')
            
            models_metrics.append(metrics_row)
        
        if not models_metrics:
            print("No metrics available for comparison report")
            return
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(models_metrics)
        
        # Save metrics to CSV
        metrics_file = os.path.join(reports_dir, f"model_metrics_comparison_{timestamp}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        # Create comparison visualizations
        
        # 1. Bar chart of key metrics for each model
        # Identify metrics common to all models
        common_metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in common_metrics if m in metrics_df.columns]
        
        if available_metrics:
            plt.figure(figsize=(14, 8))
            metrics_df_melted = pd.melt(
                metrics_df, 
                id_vars=['model_type'], 
                value_vars=available_metrics,
                var_name='Metric', 
                value_name='Value'
            )
            
            ax = sns.barplot(x='model_type', y='Value', hue='Metric', data=metrics_df_melted)
            plt.title('Model Performance Comparison', fontsize=16)
            plt.xlabel('Model Type', fontsize=14)
            plt.ylabel('Metric Value', fontsize=14)
            plt.legend(title='Metric', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(reports_dir, f"model_comparison_chart_{timestamp}.png"))
            plt.close()
        
        # 2. Create heatmap of metrics
        if len(metrics_df) > 1 and len(available_metrics) > 1:
            # Set model_type as index
            metrics_heatmap_df = metrics_df.set_index('model_type')[available_metrics]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics_heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title('Model Metrics Heatmap', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, f"metrics_heatmap_{timestamp}.png"))
            plt.close()
        
        # 3. Create ROC curves comparison for binary models (if applicable)
        binary_models = ['emergency_visit']
        has_binary_models = any(model in results for model in binary_models)
        
        if has_binary_models:
            from sklearn.metrics import roc_curve, auc
            
            plt.figure(figsize=(10, 8))
            
            for model_type in binary_models:
                if model_type in results and 'y_test' in results[model_type] and 'y_pred_prob' in results[model_type]:
                    result = results[model_type]
                    y_test = result['y_test']
                    y_pred_prob = result['y_pred_prob']
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, lw=2, label=f'{model_type} (AUC = {roc_auc:.3f})')
            
            # Add diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('ROC Curves Comparison', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(reports_dir, f"roc_curves_comparison_{timestamp}.png"))
            plt.close()
        
        # 4. Create feature importance plots for each model
        for model_type, result in results.items():
            if 'feature_selection' not in result or 'selected_features' not in result:
                continue
            
            feature_selection = result['feature_selection']
            
            if 'importance_scores' not in feature_selection:
                continue
            
            importance_scores = feature_selection['importance_scores']
            
            # Convert to DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(importance_scores.keys()),
                'Importance': list(importance_scores.values())
            }).sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            n_features = min(20, len(importance_df))
            plt.figure(figsize=(12, 10))
            
            sns.barplot(y='Feature', x='Importance', data=importance_df.head(n_features))
            plt.title(f'Top {n_features} Important Features for {model_type.replace("_", " ").title()} Model', fontsize=16)
            plt.xlabel('Importance Score', fontsize=14)
            plt.tight_layout()
            
            plt.savefig(os.path.join(reports_dir, f"{model_type}_feature_importance_{timestamp}.png"))
            plt.close()
        
        # 5. Create a hyperparameters comparison table
        hp_rows = []
        
        for model_type, result in results.items():
            if 'hyperparameter_tuning' not in result or 'best_hyperparameters' not in result['hyperparameter_tuning']:
                continue
            
            hp = result['hyperparameter_tuning']['best_hyperparameters']
            
            hp_row = {'model_type': model_type}
            
            # Add key hyperparameters
            for param in ['learning_rate', 'units_1', 'units_2', 'dropout_1', 'dropout_2', 
                        'activation_1', 'optimizer', 'l2_1', 'batch_norm_1']:
                if param in hp:
                    hp_row[param] = hp[param]
            
            hp_rows.append(hp_row)
        
        if hp_rows:
            hp_df = pd.DataFrame(hp_rows)
            hp_file = os.path.join(reports_dir, f"hyperparameters_comparison_{timestamp}.csv")
            hp_df.to_csv(hp_file, index=False)
        
        print(f"Created model comparison report in {reports_dir}")

    def run_optimized_pipeline(self, condition_keywords=None, force_new_timelines=False, append_new_months=True):
        """
        Run the optimized ML pipeline for health predictions with hyperparameter tuning
        
        Args:
            condition_keywords: List of keywords to identify target patients
            force_new_timelines: Whether to force creation of new timelines even if existing data is found
            append_new_months: Whether to append new months to existing patients
            
        Returns:
            Dictionary with pipeline results or False if pipeline failed
        """
        print("Starting optimized health prediction pipeline with hyperparameter tuning")
        
        # 1. Load data
        print("\n=== Loading Data ===")
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
        
        # 2. Identify target patients
        print("\n=== Identifying Target Patients ===")
        target_patient_ids = self.identify_target_patients(condition_keywords)
        
        if len(target_patient_ids) == 0:
            print("No target patients found. Exiting.")
            return False
        
        # 3. Create or update enhanced patient timelines
        print("\n=== Creating/Updating Enhanced Patient Timelines ===")
        timeline_df, targets = self.load_or_append_patient_timelines(
            target_patient_ids, 
            force_new=force_new_timelines,
            append_new_months=append_new_months
        )
        
        if len(timeline_df) == 0:
            print("No timeline data generated. Exiting.")
            return False
        
        # 4. Train optimized prediction models with hyperparameter tuning
        print("\n=== Training Optimized Prediction Models ===")
        model_results = self.train_optimized_models(timeline_df, targets)
        
        # 5. Generate clinical insights
        print("\n=== Generating Clinical Insights ===")
        clinical_insights = self.generate_clinical_insights(model_results)
        
        # Save insights to JSON
        import os
        import json
        from datetime import datetime
        
        insights_path = os.path.join('optimized_models', f'clinical_insights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(insights_path, 'w') as f:
            json.dump(clinical_insights, f, indent=2)
        
        print(f"Saved clinical insights to {insights_path}")
        
        # Display key insights summary
        print("\n=== Clinical Insights Summary ===")
        
        print("\nTop Emergency Risk Factors:")
        for i, factor in enumerate(clinical_insights['emergency_risk_factors'][:5]):
            print(f"{i+1}. {factor['factor']} (Importance: {factor['importance']:.4f})")
        
        print("\nMost Common Predicted Diagnoses:")
        for i, diagnosis in enumerate(clinical_insights['common_diagnoses'][:5]):
            print(f"{i+1}. {diagnosis['diagnosis']} (Probability: {diagnosis['probability']:.4f})")
        
        print("\nTop Recommended Medications:")
        for i, med in enumerate(clinical_insights['recommended_medications'][:5]):
            print(f"{i+1}. {med['medication']} (Probability: {med['probability']:.4f})")
        
        print("\nTop Suggested Procedures:")
        for i, proc in enumerate(clinical_insights['suggested_procedures'][:5]):
            print(f"{i+1}. {proc['procedure']} (Probability: {proc['probability']:.4f})")
        
        print("\n=== Optimized Pipeline Complete ===")
        
        return {
            'model_results': model_results,
            'clinical_insights': clinical_insights,
            'timeline_df': timeline_df
        }
        
    def tune_model_hyperparameters(self, X_train, y_train, X_val, y_val, model_type='binary', 
                                tuning_epochs=30, search_epochs=5, max_trials=20):
        """
        Use Keras Tuner to optimize hyperparameters for the prediction model with F1 score as the primary metric
        
        Args:
            X_train: Training feature matrix
            y_train: Training target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            model_type: Type of model ('binary' for emergency visit prediction or 'multi_label' for others)
            tuning_epochs: Number of epochs to train each trial model
            search_epochs: Number of epochs for final training with best hyperparameters
            max_trials: Maximum number of different hyperparameter combinations to try
            
        Returns:
            Optimized model and hyperparameter search results
        """
        import os
        import numpy as np
        from tensorflow import keras
        import tensorflow as tf
        import keras_tuner as kt
        from datetime import datetime
        
        # Create directory for checkpoints
        os.makedirs('optimized_saved_models/checkpoints', exist_ok=True)
        
        # Get timestamp for unique checkpoint naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        input_shape = X_train.shape[1]  # Number of features
        
        # For multi-label models, get the number of output classes
        num_classes = 1 if model_type == 'binary' else y_train.shape[1]
        
        # Define model-building function for Keras Tuner
        def build_model(hp):
            model = keras.Sequential()
            
            # Input layer
            model.add(keras.layers.Input(shape=(input_shape,)))
            
            # First hidden layer - tune number of units, activation function and regularization
            model.add(keras.layers.Dense(
                units=hp.Int('units_1', min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation_1', values=['relu', 'selu', 'elu', 'tanh']),
                kernel_regularizer=keras.regularizers.l2(
                    hp.Float('l2_1', min_value=1e-4, max_value=1e-2, sampling='log')
                )
            ))
            
            # Add BatchNormalization with optional momentum
            use_batch_norm = hp.Boolean('batch_norm_1')
            if use_batch_norm:
                model.add(keras.layers.BatchNormalization(
                    momentum=hp.Float('bn_momentum_1', min_value=0.7, max_value=0.99, default=0.9)
                ))
            
            # Add Dropout with tunable rate
            model.add(keras.layers.Dropout(
                rate=hp.Float('dropout_1', min_value=0.1, max_value=0.6, step=0.1)
            ))
            
            # Optional second hidden layer
            if hp.Boolean('second_layer'):
                model.add(keras.layers.Dense(
                    units=hp.Int('units_2', min_value=16, max_value=256, step=16),
                    activation=hp.Choice('activation_2', values=['relu', 'selu', 'elu', 'tanh']),
                    kernel_regularizer=keras.regularizers.l2(
                        hp.Float('l2_2', min_value=1e-4, max_value=1e-2, sampling='log')
                    )
                ))
                
                # Optional batch normalization for second layer
                if hp.Boolean('batch_norm_2'):
                    model.add(keras.layers.BatchNormalization(
                        momentum=hp.Float('bn_momentum_2', min_value=0.7, max_value=0.99, default=0.9)
                    ))
                
                model.add(keras.layers.Dropout(
                    rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
                ))
            
            # Optional third hidden layer
            if hp.Boolean('third_layer'):
                model.add(keras.layers.Dense(
                    units=hp.Int('units_3', min_value=8, max_value=128, step=8),
                    activation=hp.Choice('activation_3', values=['relu', 'selu', 'elu', 'tanh']),
                    kernel_regularizer=keras.regularizers.l2(
                        hp.Float('l2_3', min_value=1e-4, max_value=1e-2, sampling='log')
                    )
                ))
                
                model.add(keras.layers.Dropout(
                    rate=hp.Float('dropout_3', min_value=0.1, max_value=0.4, step=0.1)
                ))
            
            # Output layer
            if model_type == 'binary':
                model.add(keras.layers.Dense(1, activation='sigmoid'))
            else:
                model.add(keras.layers.Dense(num_classes, activation='sigmoid'))  # multi-label
            
            # Tune learning rate for optimizer
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            
            # Choose optimizer
            optimizer_choice = hp.Choice('optimizer', values=['adam', 'adamw', 'rmsprop'])
            if optimizer_choice == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_choice == 'adamw':
                optimizer = keras.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=hp.Float('weight_decay', min_value=1e-5, max_value=1e-3, sampling='log')
                )
            else:  # rmsprop
                optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            
            # Define metrics based on model type
            metrics = [
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.F1Score(
                    name='f1_score',
                    threshold=0.5,
                    dtype=tf.float32
                )
            ]
            
            # Compile model - use binary_crossentropy for both binary and multi-label
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=metrics
            )
            
            return model
        
        # Create instance of the tuner - now using F1 Score as the objective
        tuner = kt.Hyperband(
            build_model,
            objective=kt.Objective('val_f1_score', direction='max'),  # Changed from 'val_auc' to 'val_f1_score'
            max_epochs=tuning_epochs,
            factor=3,
            directory='hyperparameter_tuning',
            project_name=f'clinical_prediction_{model_type}_{timestamp}',
            overwrite=True
        )
        
        # Define early stopping callback for the search
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_f1_score',  # Changed from 'val_loss' to 'val_f1_score'
            patience=5,
            restore_best_weights=True
        )
        
        # Add checkpoint callback like in train_models
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=f'saved_models/checkpoints/{model_type}_tuning_{timestamp}_epoch_{{epoch:02d}}.keras',
            save_best_only=False,  # Save every epoch
            save_weights_only=False,
            verbose=1
        )
        
        # Print search space summary
        tuner.search_space_summary()
        
        # Prepare sample weights if dealing with imbalanced data
        if model_type == 'binary':
            # Class weights for binary imbalanced problems
            if len(np.unique(y_train)) <= 2:  # Binary classification
                pos_count = np.sum(y_train)
                neg_count = len(y_train) - pos_count
                
                if pos_count > 0 and neg_count > 0:
                    # Calculate class weights
                    weight_ratio = neg_count / pos_count
                    class_weight = {0: 1.0, 1: min(5.0, weight_ratio)}  # Cap at 5x to prevent extreme weights
                    print(f"Using class weights: {class_weight}")
                    
                    # Search with class weights
                    tuner.search(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=tuning_epochs,
                        callbacks=[early_stopping, checkpoint_callback],  # Added checkpoint_callback
                        class_weight=class_weight,
                        verbose=1
                    )
                else:
                    # Handle edge case where one class has zero examples
                    print("Warning: One class has zero examples. Proceeding without class weights.")
                    tuner.search(
                        X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=tuning_epochs,
                        callbacks=[early_stopping, checkpoint_callback],  # Added checkpoint_callback
                        verbose=1
                    )
            else:
                # Not a binary problem, proceed without class weights
                tuner.search(
                    X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=tuning_epochs,
                    callbacks=[early_stopping, checkpoint_callback],  # Added checkpoint_callback
                    verbose=1
                )
        else:
            # For multi-label, use sample weighting
            sample_weights = np.ones(len(y_train))
            
            # Increase weight for samples with at least one positive label
            pos_samples = np.sum(y_train, axis=1) > 0
            sample_weights[pos_samples] = 3.0
            
            tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=tuning_epochs,
                callbacks=[early_stopping, checkpoint_callback],  # Added checkpoint_callback
                sample_weight=sample_weights,
                verbose=1
            )
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters(1)[0]
        print("Best hyperparameters:")
        for param, value in best_hp.values.items():
            print(f"{param}: {value}")
        
        # Build model with best hyperparameters
        best_model = build_model(best_hp)
        
        # Add reduce learning rate callback for final training
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',  # Monitor F1 score
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Create checkpoint callback for best model
        final_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=f'saved_models/checkpoints/{model_type}_best_{timestamp}_epoch_{{epoch:02d}}.keras',
            save_best_only=False,
            save_weights_only=False,
            verbose=1
        )
        
        # Train final model with best hyperparameters
        if model_type == 'binary' and len(np.unique(y_train)) <= 2:
            pos_count = np.sum(y_train)
            neg_count = len(y_train) - pos_count
            
            if pos_count > 0 and neg_count > 0:
                weight_ratio = neg_count / pos_count
                class_weight = {0: 1.0, 1: min(5.0, weight_ratio)}
                
                history = best_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=search_epochs,
                    callbacks=[
                        early_stopping, 
                        reduce_lr,
                        final_checkpoint_callback
                    ],
                    class_weight=class_weight,
                    verbose=1
                )
            else:
                history = best_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=search_epochs,
                    callbacks=[
                        early_stopping, 
                        reduce_lr,
                        final_checkpoint_callback
                    ],
                    verbose=1
                )
        else:
            # For multi-label, use sample weighting
            sample_weights = np.ones(len(y_train))
            pos_samples = np.sum(y_train, axis=1) > 0
            sample_weights[pos_samples] = 3.0
            
            history = best_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=search_epochs,
                callbacks=[
                    early_stopping, 
                    reduce_lr,
                    final_checkpoint_callback
                ],
                sample_weight=sample_weights,
                verbose=1
            )
        
        # Evaluate final model
        eval_results = best_model.evaluate(X_val, y_val)
        metrics = {name: value for name, value in zip(best_model.metrics_names, eval_results)}
        
        print("Final model evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot training history
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['f1_score'], label='Train F1')
        plt.plot(history.history['val_f1_score'], label='Validation F1')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['precision'], label='Precision')
        plt.plot(history.history['recall'], label='Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'hyperparameter_tuning/final_model_{model_type}_training_{timestamp}.png')
        plt.close()
        
        return {
            'model': best_model,
            'best_hyperparameters': best_hp.values,
            'metrics': metrics,
            'history': history.history,
            'tuner': tuner,
            'timestamp': timestamp
        }


    def enhanced_feature_selection(self, X, y, feature_names, n_features_range=(5, 50, 5), method='ensemble', 
                            n_folds=5, plot_results=True, eval_method='cv'):
        """
        Perform enhanced feature selection using multiple advanced methods and find optimal feature count
        
        Args:
            X: Input features
            y: Target values
            feature_names: List of feature names
            n_features_range: Tuple of (min_features, max_features, step) or list of feature counts to evaluate
            method: Selection method ('ensemble', 'rfe', 'boruta', 'shap', 'mutual_info')
            n_folds: Number of cross-validation folds for evaluation
            plot_results: Whether to create visualization of selected features
            eval_method: Evaluation method ('cv' for cross-validation or 'validation' to use a validation set)
            
        Returns:
            Dictionary with selected feature indices, names, importance scores and optimal feature count
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.feature_selection import mutual_info_classif, f_classif
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

        print(f"Performing enhanced feature selection using {method} method with feature count optimization...")
        
        # Verify dimensions of inputs
        print(f"X shape: {X.shape}, feature_names length: {len(feature_names)}")
        
        # Handle mismatched feature names
        if len(feature_names) != X.shape[1]:
            print(f"WARNING: Number of feature names ({len(feature_names)}) does not match X shape ({X.shape[1]})")
            print("Adjusting feature names to match X dimensions")
            if len(feature_names) > X.shape[1]:
                # Truncate feature names if there are too many
                feature_names = feature_names[:X.shape[1]]
            else:
                # Add generic names if there are too few
                additional_names = [f"feature_{i}" for i in range(len(feature_names), X.shape[1])]
                feature_names = list(feature_names) + additional_names
        
        # For multi-label targets, create a binary target for feature selection
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Create a binary target based on whether any positive class exists
            target = np.any(y == 1, axis=1).astype(int)
        else:
            # If already 1D, ensure it's flattened
            target = y.flatten() if len(y.shape) > 1 else y
        
        # Parse n_features_range parameter
        if isinstance(n_features_range, tuple) and len(n_features_range) == 3:
            min_features, max_features, step = n_features_range
            feature_counts = list(range(min_features, min(max_features+1, X.shape[1]), step))
        elif isinstance(n_features_range, list):
            feature_counts = [count for count in n_features_range if count <= X.shape[1]]
        else:
            # Default to a range of feature counts
            feature_counts = [10, 20, 30, 50, 75, 100]
            feature_counts = [count for count in feature_counts if count <= X.shape[1]]
        
        # Ensure we have at least one feature count
        if not feature_counts:
            feature_counts = [min(30, X.shape[1])]
        
        print(f"Evaluating feature counts: {feature_counts}")
        
        # Initialize results dictionary
        results = {
            'selected_indices': [],
            'selected_features': [],
            'importance_scores': {},
            'method': method,
            'evaluated_feature_counts': feature_counts,
            'feature_count_scores': {},
            'optimal_feature_count': None
        }
        
        # Create a DataFrame for feature importance visualization
        features_df = pd.DataFrame({'Feature': feature_names})
        
        # Step 1: Calculate importance scores for all features
        if method == 'ensemble':
            # Ensemble method: combine multiple feature selection techniques
            print("Using ensemble feature selection approach...")
            
            importance_methods = {
                'random_forest': None,
                'gradient_boosting': None,
                'mutual_info': None,
                'anova_f': None,
                'lasso': None
            }
            
            # 1. Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(X, target)
            importance_methods['random_forest'] = rf.feature_importances_
            features_df['RF_Importance'] = rf.feature_importances_
            
            # 2. Gradient Boosting importance
            try:
                gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                gb.fit(X, target)
                importance_methods['gradient_boosting'] = gb.feature_importances_
                features_df['GB_Importance'] = gb.feature_importances_
            except Exception as e:
                print(f"Warning: Gradient Boosting failed: {e}")
                importance_methods['gradient_boosting'] = np.zeros(X.shape[1])
                features_df['GB_Importance'] = np.zeros(X.shape[1])
            
            # 3. Mutual Information
            try:
                mi_scores = mutual_info_classif(X, target, random_state=42)
                importance_methods['mutual_info'] = mi_scores
                features_df['MI_Score'] = mi_scores
            except Exception as e:
                print(f"Warning: Mutual Information failed: {e}")
                importance_methods['mutual_info'] = np.zeros(X.shape[1])
                features_df['MI_Score'] = np.zeros(X.shape[1])
            
            # 4. ANOVA F-value
            try:
                f_scores, _ = f_classif(X, target)
                # Replace inf and NaN with 0
                f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
                # Normalize to 0-1 scale
                if np.max(f_scores) > 0:
                    f_scores = f_scores / np.max(f_scores)
                importance_methods['anova_f'] = f_scores
                features_df['F_Score'] = f_scores
            except Exception as e:
                print(f"Warning: ANOVA F-value failed: {e}")
                importance_methods['anova_f'] = np.zeros(X.shape[1])
                features_df['F_Score'] = np.zeros(X.shape[1])
            
            # 5. Lasso feature selection with cross-validation
            lasso_importance = np.zeros(X.shape[1])
            
            if X.shape[0] > 10:  # Only run if we have enough samples
                try:
                    from sklearn.linear_model import LassoCV
                    
                    # Normalize X for Lasso
                    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
                    X_scaled = np.nan_to_num(X_scaled)  # Replace NaNs
                    
                    # Run LassoCV to find optimal alpha
                    lasso = LassoCV(cv=min(5, n_folds), random_state=42, max_iter=10000)
                    lasso.fit(X_scaled, target)
                    
                    # Get coefficient importances
                    lasso_importance = np.abs(lasso.coef_)
                    
                    # Normalize to 0-1 scale
                    if np.max(lasso_importance) > 0:
                        lasso_importance = lasso_importance / np.max(lasso_importance)
                except Exception as e:
                    print(f"Warning: Lasso feature selection failed: {e}")
                    # Fill with zeros if it fails
                    lasso_importance = np.zeros(X.shape[1])
            
            importance_methods['lasso'] = lasso_importance
            features_df['Lasso_Importance'] = lasso_importance
            
            # Calculate aggregated importance scores
            ensemble_importance = np.zeros(X.shape[1])
            
            for method_name, importance in importance_methods.items():
                if importance is not None:
                    # Ensure all values are positive and normalized
                    normalized_importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.max(normalized_importance) > 0:
                        normalized_importance = normalized_importance / np.max(normalized_importance)
                    ensemble_importance += normalized_importance
            
            # Normalize the ensemble importance
            if np.max(ensemble_importance) > 0:
                ensemble_importance = ensemble_importance / np.max(ensemble_importance)
            
            # Add ensemble score to DataFrame
            features_df['Ensemble_Score'] = ensemble_importance
            
            # Sort by ensemble importance
            feature_ranking = pd.DataFrame({
                'Feature': feature_names,
                'Importance': ensemble_importance
            }).sort_values('Importance', ascending=False)
            
            # Store all feature rankings regardless of the count
            all_feature_importances = dict(zip(feature_ranking['Feature'], feature_ranking['Importance']))
            
        else:
            # Default to Random Forest importance
            print(f"Using Random Forest importance...")
            
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(X, target)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Create a dataframe of features and importances
            feature_ranking = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            features_df['RF_Importance'] = importances
            
            # Store all feature importances
            all_feature_importances = dict(zip(feature_ranking['Feature'], feature_ranking['Importance']))
        
        # Step 2: Evaluate different feature counts to find the optimal number
        print("\nEvaluating performance with different feature counts...")
        
        # Split data for validation if needed
        if eval_method == 'validation':
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.2, random_state=42)
        
        evaluation_scores = {}
        
        # Create a classifier for evaluation
        eval_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        for n_feat in feature_counts:
            print(f"Evaluating with {n_feat} features...")
            
            # Get top n_feat features
            top_features = feature_ranking.head(n_feat)
            selected_indices = [list(feature_names).index(feat) for feat in top_features['Feature']]
            
            # Use only selected features
            X_selected = X[:, selected_indices]
            
            # Evaluate using cross-validation or validation set
            if eval_method == 'cv':
                # Use stratified k-fold cross-validation
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                cv_scores = {
                    'accuracy': cross_val_score(eval_clf, X_selected, target, cv=cv, scoring='accuracy'),
                    'auc': cross_val_score(eval_clf, X_selected, target, cv=cv, scoring='roc_auc'),
                    'f1': cross_val_score(eval_clf, X_selected, target, cv=cv, scoring='f1')
                }
                
                # Average scores
                avg_scores = {metric: scores.mean() for metric, scores in cv_scores.items()}
                std_scores = {metric: scores.std() for metric, scores in cv_scores.items()}
                
                evaluation_scores[n_feat] = {
                    'avg_scores': avg_scores,
                    'std_scores': std_scores,
                    'selected_indices': selected_indices,
                    'selected_features': top_features['Feature'].tolist()
                }
                
                print(f"  CV Results - Accuracy: {avg_scores['accuracy']:.4f}, AUC: {avg_scores['auc']:.4f}, F1: {avg_scores['f1']:.4f}")
                
            else:  # validation set
                # Train on training set
                X_train_selected = X_train[:, selected_indices]
                X_val_selected = X_val[:, selected_indices]
                
                eval_clf.fit(X_train_selected, y_train)
                
                # Predict on validation set
                y_pred = eval_clf.predict(X_val_selected)
                y_pred_proba = eval_clf.predict_proba(X_val_selected)[:, 1] if hasattr(eval_clf, 'predict_proba') else None
                
                # Calculate metrics
                val_scores = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'f1': f1_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred),
                    'recall': recall_score(y_val, y_pred)
                }
                
                if y_pred_proba is not None:
                    val_scores['auc'] = roc_auc_score(y_val, y_pred_proba)
                
                evaluation_scores[n_feat] = {
                    'val_scores': val_scores,
                    'selected_indices': selected_indices,
                    'selected_features': top_features['Feature'].tolist()
                }
                
                print(f"  Val Results - Accuracy: {val_scores['accuracy']:.4f}, " + 
                    (f"AUC: {val_scores['auc']:.4f}, " if 'auc' in val_scores else "") + 
                    f"F1: {val_scores['f1']:.4f}")
        
        # Step 3: Find the optimal feature count based on F1 score
        if eval_method == 'cv':
            f1_scores = {n_feat: scores['avg_scores']['f1'] for n_feat, scores in evaluation_scores.items()}
        else:
            f1_scores = {n_feat: scores['val_scores']['f1'] for n_feat, scores in evaluation_scores.items()}
        
        # Find the optimal feature count (highest F1 score)
        optimal_n_features = max(f1_scores.items(), key=lambda x: x[1])[0]
        
        print(f"\nOptimal feature count: {optimal_n_features} with F1 score: {f1_scores[optimal_n_features]:.4f}")
        
        # Get the results for the optimal feature count
        optimal_results = evaluation_scores[optimal_n_features]
        selected_indices = optimal_results['selected_indices']
        selected_features = optimal_results['selected_features']
        
        # Create importance scores for selected features
        importance_scores = {feature: all_feature_importances.get(feature, 0.0) for feature in selected_features}
        
        # Store final results
        results['selected_indices'] = selected_indices
        results['selected_features'] = selected_features
        results['importance_scores'] = importance_scores
        results['feature_count_scores'] = f1_scores
        results['optimal_feature_count'] = optimal_n_features
        results['evaluation_scores'] = evaluation_scores
        results['all_features_df'] = features_df
        
        # Plot feature importance and count evaluation results
        if plot_results:
            # Create directory for plots
            os.makedirs('feature_selection', exist_ok=True)
            
            # 1. Plot optimal feature set importance scores
            n_to_plot = min(30, len(selected_features))
            
            # Sort features by importance for plotting
            plot_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': [importance_scores[feature] for feature in selected_features]
            }).sort_values('Importance', ascending=False).head(n_to_plot)
            
            plt.figure(figsize=(12, max(6, n_to_plot * 0.3)))
            ax = sns.barplot(x='Importance', y='Feature', data=plot_df)
            
            # Add value labels
            for i, v in enumerate(plot_df['Importance']):
                ax.text(v + 0.001, i, f"{v:.4f}", va='center')
            
            plt.title(f'Top {n_to_plot} Features from Optimal Set of {optimal_n_features} Features')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f'feature_selection/optimal_feature_set_{method}.png')
            plt.close()
            
            # 2. Plot feature count evaluation results
            plt.figure(figsize=(10, 6))
            
            # Plot F1 scores vs. feature count
            plt.plot(list(f1_scores.keys()), list(f1_scores.values()), 'o-', label='F1 Score')
            
            # Mark the optimal feature count
            plt.axvline(x=optimal_n_features, color='r', linestyle='--', 
                        label=f'Optimal: {optimal_n_features} features')
            
            # Add a point at the optimal feature count
            optimal_f1 = f1_scores[optimal_n_features]
            plt.plot(optimal_n_features, optimal_f1, 'ro', markersize=10)
            plt.annotate(f'F1={optimal_f1:.4f}', 
                        (optimal_n_features, optimal_f1),
                        xytext=(10, -20),
                        textcoords='offset points')
            
            plt.title(f'Feature Count Evaluation ({method} method)')
            plt.xlabel('Number of Features')
            plt.ylabel('F1 Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'feature_selection/feature_count_evaluation_{method}.png')
            plt.close()
            
            # 3. Plot all evaluation metrics for different feature counts
            plt.figure(figsize=(12, 6))
            
            if eval_method == 'cv':
                metrics = ['accuracy', 'auc', 'f1']
                for metric in metrics:
                    values = [scores['avg_scores'][metric] for n_feat, scores in evaluation_scores.items()]
                    plt.plot(feature_counts, values, 'o-', label=metric.upper())
            else:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                if 'auc' in evaluation_scores[feature_counts[0]]['val_scores']:
                    metrics.append('auc')
                    
                for metric in metrics:
                    values = [scores['val_scores'][metric] for n_feat, scores in evaluation_scores.items()]
                    plt.plot(feature_counts, values, 'o-', label=metric.upper())
            
            # Mark the optimal feature count
            plt.axvline(x=optimal_n_features, color='r', linestyle='--', 
                        label=f'Optimal: {optimal_n_features} features')
            
            plt.title(f'Performance Metrics vs. Feature Count ({method} method)')
            plt.xlabel('Number of Features')
            plt.ylabel('Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'feature_selection/metrics_comparison_{method}.png')
            plt.close()
        
        print(f"Selected {len(selected_indices)} features with {method} method (optimal count: {optimal_n_features})")
        print(f"Top 10 selected features:")
        
        # Print top features and their importance scores
        for i, feature in enumerate(selected_features[:10]):
            importance = importance_scores[feature]
            print(f"{i+1}. {feature} (Importance: {importance:.6f})")
        
        return results

    def feature_stability_analysis(self, X, y, feature_names, method='ensemble', 
                                n_features=50, n_runs=10, sampling_fraction=0.8):
        """
        Assess feature selection stability with bootstrapping to identify consistently important features
        
        Args:
            X: Input features
            y: Target values
            feature_names: List of feature names
            method: Selection method ('ensemble', 'rfe', 'boruta', 'shap', 'mutual_info')
            n_features: Target number of features to select
            n_runs: Number of bootstrap iterations
            sampling_fraction: Fraction of data to sample in each run
            
        Returns:
            Dictionary with stability analysis results
        """
        print(f"Performing feature stability analysis with {n_runs} iterations...")
        
        # Initialize results
        stability_results = {
            'feature_counts': {name: 0 for name in feature_names},
            'feature_ranks': {name: [] for name in feature_names},
            'selection_probability': {},
            'stability_score': 0.0,
            'consistently_selected': [],
            'run_results': []
        }
        
        # Prepare target for feature selection
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Create a binary target for feature selection
            target = np.any(y == 1, axis=1).astype(int)
        else:
            target = y.flatten() if len(y.shape) > 1 else y
        
        # Perform multiple feature selection runs with bootstrapping
        for run in range(n_runs):
            print(f"Run {run+1}/{n_runs}...")
            
            # Create bootstrap sample
            n_samples = int(X.shape[0] * sampling_fraction)
            bootstrap_indices = np.random.choice(X.shape[0], n_samples, replace=True)
            X_boot = X[bootstrap_indices]
            y_boot = target[bootstrap_indices]
            
            # Perform feature selection
            selection_result = self.enhanced_feature_selection(
                X_boot, y_boot, feature_names, 
                n_features_range=(5, 50, 5), method=method,
                plot_results=False
            )
            
            # Store results from this run
            stability_results['run_results'].append({
                'selected_features': selection_result['selected_features'],
                'selected_indices': selection_result['selected_indices'],
                'importance_scores': selection_result['importance_scores']
            })
            
            # Update feature counts and ranks
            for i, feature in enumerate(selection_result['selected_features']):
                stability_results['feature_counts'][feature] += 1
                stability_results['feature_ranks'][feature].append(i)
        
        # Calculate selection probability for each feature
        for feature, count in stability_results['feature_counts'].items():
            stability_results['selection_probability'][feature] = count / n_runs
        
        # Identify consistently selected features (selected in at least 70% of runs)
        consistently_selected = []
        for feature, prob in stability_results['selection_probability'].items():
            if prob >= 0.7:  # At least 70% selection rate
                consistently_selected.append((feature, prob))
        
        # Sort by selection probability
        consistently_selected.sort(key=lambda x: x[1], reverse=True)
        stability_results['consistently_selected'] = consistently_selected
        
        # Calculate stability score (average Jaccard similarity between feature sets)
        jaccard_sums = 0
        comparison_count = 0
        
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                set_i = set(stability_results['run_results'][i]['selected_features'])
                set_j = set(stability_results['run_results'][j]['selected_features'])
                
                # Calculate Jaccard similarity: |intersection| / |union|
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                
                if union > 0:
                    jaccard_sums += intersection / union
                    comparison_count += 1
        
        # Average Jaccard similarity
        if comparison_count > 0:
            stability_results['stability_score'] = jaccard_sums / comparison_count
        
        # Create visualization of feature stability
        plt.figure(figsize=(12, 8))
        
        # Get data for top 30 most frequently selected features
        top_features = sorted(
            [(f, stability_results['selection_probability'][f]) for f in feature_names],
            key=lambda x: x[1], 
            reverse=True
        )[:30]
        
        # Plot selection probability
        feature_names_for_plot = [f[0] for f in top_features]
        probabilities = [f[1] for f in top_features]
        
        ax = sns.barplot(y=feature_names_for_plot, x=probabilities, palette='viridis')
        
        # Add value labels
        for i, v in enumerate(probabilities):
            ax.text(v + 0.02, i, f"{v:.2f}", va='center')
        
        # Add a vertical line at 0.7 threshold
        plt.axvline(x=0.7, color='red', linestyle='--', label='Stability Threshold (70%)')
        
        plt.title(f'Feature Selection Stability Analysis ({n_runs} runs, {method} method)')
        plt.xlabel('Selection Probability')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'feature_selection/stability_analysis_{method}.png')
        plt.close()
        
        # Print stability results
        print("\nFeature Stability Analysis Results:")
        print(f"Stability Score (avg Jaccard similarity): {stability_results['stability_score']:.4f}")
        print(f"\nConsistently Selected Features (≥70% of runs):")
        
        for i, (feature, prob) in enumerate(consistently_selected[:20]):  # Show top 20
            avg_rank = np.mean(stability_results['feature_ranks'][feature]) if stability_results['feature_ranks'][feature] else np.nan
            print(f"{i+1}. {feature}: {prob*100:.1f}% (avg rank: {avg_rank:.1f})")
        
        return stability_results
    



    def balance_unified_timeline(self,filepath, random_seed=42, save_dir=None, preserve_positives=True):
        """
        Load timeline data and create a single balanced dataset that achieves good balance
        for all target variables (emergency_visit, diagnosis, medication, procedure).
        
        Args:
            filepath: Path to the timeline CSV file
            random_seed: Random seed for reproducibility (must be an integer)
            save_dir: Directory to save balanced dataset (defaults to same directory as input file)
            preserve_positives: Whether to preserve all positive examples for all targets (default True)
            
        Returns:
            Balanced DataFrame that achieves good balance for all target variables
        """
        print(f"Loading timeline data from {filepath}")
        
        try:
            # Ensure random_seed is an integer
            try:
                random_seed = int(random_seed)
            except (ValueError, TypeError):
                print(f"Warning: random_seed '{random_seed}' is not a valid integer. Using default value 42.")
                random_seed = 42
                
            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Set save directory
            if save_dir is None:
                save_dir = os.path.dirname(filepath)
                if not save_dir:  # If filepath doesn't have a directory part
                    save_dir = "."
            os.makedirs(save_dir, exist_ok=True)
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Error: File '{filepath}' not found.")
                return None
            
            # Load the CSV file
            try:
                df = pd.read_csv(filepath, low_memory=False)
            except Exception as e:
                print(f"Error loading CSV file: {str(e)}")
                return None
                
            # Store original columns to preserve at the end
            original_columns = df.columns.tolist()
            
            filename = os.path.basename(filepath).split('.')[0]
            original_row_count = len(df)
            print(f"Original data rows: {original_row_count}")
            
            # Create timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Count NaN values per row
            df['nan_count'] = df.isna().sum(axis=1)
            
            # Check how many rows have NaN values
            nan_row_count = (df['nan_count'] > 0).sum()
            print(f"Rows with at least one NaN value: {nan_row_count} ({nan_row_count/len(df)*100:.1f}%)")
            
            # Define target columns and corresponding model names
            target_columns = {
                'HAD_EMERGENCY_NEXT_MONTH': 'emergency_visit',
                'NEW_DIAGNOSES_NEXT_MONTH': 'diagnosis',
                'NEW_MEDICATIONS_NEXT_MONTH': 'medication', 
                'NEW_PROCEDURES_NEXT_MONTH': 'procedure'
            }
            
            # Verify target columns exist in dataset
            missing_columns = [col for col in target_columns.keys() if col not in df.columns]
            if missing_columns:
                print(f"Warning: The following target columns are missing from the dataset: {missing_columns}")
                # Remove missing columns from the dictionary
                for col in missing_columns:
                    del target_columns[col]
                    
            if not target_columns:
                print("Error: No valid target columns found in the dataset.")
                return None
            
            # First, drop rows where any target column is NaN (essential for model training)
            df_filtered = df.dropna(subset=list(target_columns.keys()))
            print(f"Rows after dropping target NaNs: {len(df_filtered)}")
            
            # Track new columns to remove later
            columns_to_remove = ['nan_count']
            
            # Prepare target columns for classification
            for target_column, model_name in target_columns.items():
                if model_name == 'emergency_visit':
                    # Convert emergency target to boolean
                    if df_filtered[target_column].dtype == 'object':
                        def to_bool(value):
                            if isinstance(value, bool):
                                return value
                            if isinstance(value, (int, float)):
                                return bool(value)
                            if isinstance(value, str):
                                return value.lower() in ('true', 't', 'yes', 'y', '1')
                            return False
                        
                        df_filtered[target_column] = df_filtered[target_column].apply(to_bool)
                else:
                    # For multi-label targets, create binary indicators
                    def parse_list(value):
                        try:
                            if isinstance(value, str):
                                if value.startswith('[') and value.endswith(']'):
                                    parsed = eval(value)
                                    if isinstance(parsed, list):
                                        return parsed
                            if isinstance(value, list):
                                return value
                            return []
                        except Exception:
                            return []
                    
                    list_col = f'{target_column}_list'
                    binary_col = f'{target_column}_any'
                    
                    df_filtered[list_col] = df_filtered[target_column].apply(parse_list)
                    df_filtered[binary_col] = df_filtered[list_col].apply(lambda x: len(x) > 0)
                    
                    # Track these columns to remove later
                    columns_to_remove.extend([list_col, binary_col])
            
            # Print initial distributions
            print("\n==== Initial class distributions ====")
            positives_by_target = {}
            for target_column, model_name in target_columns.items():
                binary_col = target_column if model_name == 'emergency_visit' else f'{target_column}_any'
                positive_count = df_filtered[binary_col].sum()
                negative_count = len(df_filtered) - positive_count
                positives_by_target[binary_col] = positive_count
                print(f"{model_name}: {positive_count} positive, {negative_count} negative "
                    f"({positive_count/len(df_filtered)*100:.1f}% positive)")
            
            # Create a new strategy: preserve all positive examples for all targets
            if preserve_positives:
                print("\n==== Strategy: Preserve all positive examples for all targets ====")
                
                # Create masks for positive examples for each target
                positive_masks = {}
                for target_column, model_name in target_columns.items():
                    binary_col = target_column if model_name == 'emergency_visit' else f'{target_column}_any'
                    positive_masks[binary_col] = df_filtered[binary_col] == True
                
                # Combine masks to find rows that are positive for any target
                combined_positive_mask = pd.Series(False, index=df_filtered.index)
                for mask in positive_masks.values():
                    combined_positive_mask = combined_positive_mask | mask
                
                # Rows that are positive for at least one target
                positive_rows = df_filtered[combined_positive_mask].copy()
                
                # Rows that are negative for all targets
                negative_rows = df_filtered[~combined_positive_mask].copy()
                
                print(f"Rows positive for at least one target: {len(positive_rows)}")
                print(f"Rows negative for all targets: {len(negative_rows)}")
                
                # Determine how many negatives to keep
                if len(negative_rows) <= len(positive_rows):
                    # If we have fewer negative rows than positive rows, keep all negatives
                    negative_sample = negative_rows
                else:
                    # Sort negative rows by nan_count (ascending = fewer NaNs first)
                    negative_rows = negative_rows.sort_values('nan_count')
                    
                    # Sample enough negative rows to achieve balance
                    negative_sample = negative_rows.head(len(positive_rows))
                    
                # Combine positive and negative rows
                df_balanced = pd.concat([positive_rows, negative_sample])
                
                print(f"Final balanced dataset: {len(df_balanced)} rows")
            else:
                # Fall back to original strategy if preserve_positives is False
                print("\n==== Strategy: Optimize for overall balance ====")
                
                # Create a unified balance score for each row
                df_filtered['balance_score'] = 0
                columns_to_remove.append('balance_score')
                
                # Calculate balance score for each target
                balance_weights = {}
                
                for target_column, model_name in target_columns.items():
                    # Determine the target column to use for balancing
                    balance_col = target_column if model_name == 'emergency_visit' else f'{target_column}_any'
                    
                    # Count positive and negative examples
                    positive_count = df_filtered[balance_col].sum() 
                    negative_count = len(df_filtered) - positive_count
                    
                    # Calculate ratio of positive to negative
                    if negative_count > 0:
                        pos_neg_ratio = positive_count / negative_count
                    else:
                        pos_neg_ratio = float('inf')
                        
                    # Store the weight for this target
                    balance_weights[balance_col] = pos_neg_ratio
                    
                    # Add to balance score
                    if pos_neg_ratio < 1:
                        # Need more positive examples, so increase value of positive examples
                        df_filtered.loc[df_filtered[balance_col] == True, 'balance_score'] += (1/pos_neg_ratio)
                    else:
                        # Need more negative examples, so increase value of negative examples
                        df_filtered.loc[df_filtered[balance_col] == False, 'balance_score'] += pos_neg_ratio
                
                print("\n==== Balance weights calculated ====")
                for col, weight in balance_weights.items():
                    print(f"{col}: {weight:.2f}")
                
                # Now sort by balance_score (higher value = more important for balancing)
                # When two rows have same balance score, prefer the one with fewer NaNs
                df_filtered['sort_key'] = df_filtered['balance_score'] - df_filtered['nan_count'] / df_filtered['nan_count'].max()
                columns_to_remove.append('sort_key')
                
                # Sort by our composite score
                df_sorted = df_filtered.sort_values('sort_key', ascending=False)
                
                # Determine optimal size
                min_rows_needed = 0
                for target_column, model_name in target_columns.items():
                    balance_col = target_column if model_name == 'emergency_visit' else f'{target_column}_any'
                    positive_count = df_filtered[balance_col].sum()
                    negative_count = len(df_filtered) - positive_count
                    min_rows_needed = max(min_rows_needed, 2 * min(positive_count, negative_count))
                
                # Keep the top N rows with the highest balance scores
                target_size = min(min_rows_needed, len(df_filtered))
                df_balanced = df_sorted.head(target_size)
            
            # Print final distribution
            print("\n==== Final class distributions ====")
            for target_column, model_name in target_columns.items():
                binary_col = target_column if model_name == 'emergency_visit' else f'{target_column}_any'
                positive_count = df_balanced[binary_col].sum()
                negative_count = len(df_balanced) - positive_count
                initial_positive = positives_by_target[binary_col]
                
                print(f"{model_name}: {positive_count} positive, {negative_count} negative "
                    f"({positive_count/len(df_balanced)*100:.1f}% positive)")
                
                if preserve_positives:
                    # Check if we preserved all positives
                    positive_retention = (positive_count / initial_positive) * 100
                    print(f"  - Retained {positive_count}/{initial_positive} positive examples ({positive_retention:.1f}%)")
            
            # Remove auxiliary columns used for balancing
            print(f"\nRemoving {len(columns_to_remove)} auxiliary columns added during processing")
            df_balanced = df_balanced.drop(columns=columns_to_remove)
            
            # Check if any columns were added that weren't in the original
            extra_columns = [col for col in df_balanced.columns if col not in original_columns]
            if extra_columns:
                print(f"Warning: The following columns were added during processing and will be removed: {extra_columns}")
                df_balanced = df_balanced.drop(columns=extra_columns)
            
            # Verify that final columns match original columns
            missing_columns = [col for col in original_columns if col not in df_balanced.columns]
            if missing_columns:
                print(f"Warning: The following original columns are missing from the final dataset: {missing_columns}")
            
            # Save the balanced dataset
            balanced_filepath = os.path.join(
                save_dir, f'{filename}_unified_balanced_{timestamp}.csv')
            df_balanced.to_csv(balanced_filepath, index=False)
            print(f"\nSaved unified balanced dataset to {balanced_filepath}")
            
            return df_balanced
            
        except Exception as e:
            print(f"Error processing timeline data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None