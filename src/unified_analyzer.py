"""
Unified Habitability Analysis System (UHAS)

A comprehensive package combining:
1. Physical environment analysis
2. Chemical composition analysis
3. Life prediction capabilities
4. PCA-based pattern analysis
5. Toxicity evaluation
6. Visualization suite

References:
1. Element ranges: Schlesinger & Bernhardt, "Biogeochemistry" (2020)
2. Toxic thresholds: WHO Guidelines for Drinking-water Quality (2017)
3. Compound ratios: Environmental Chemistry, Stumm & Morgan (2012)
4. Extremophile limits: Rothschild & Mancinelli, Nature Reviews (2001)
5. Biomarkers: Schwieterman et al., Astrobiology (2018)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('UHAS')


@dataclass
class ChemicalProfile:
    """
    Comprehensive chemical profile including all elements and compounds
    relevant for habitability analysis.
    """

    def __init__(self):
        # Essential elements with references
        self.essential_elements = {
            # Primary elements
            'Si': {'ideal': 27.8, 'range': (20.0, 35.0),
                  'ref': "Schlesinger 2020, p.145"},
            'Al': {'ideal': 8.1, 'range': (6.0, 10.0),
                  'ref': "Mason & Moore 1982, p.42"},
            'Fe': {'ideal': 3.8, 'range': (2.0, 5.0),
                  'ref': "Brimblecombe 2003, p.78"},
            'Ca': {'ideal': 3.2, 'range': (2.0, 4.5),
                  'ref': "Stumm & Morgan 2012, p.425"},
            'Na': {'ideal': 2.5, 'range': (1.5, 3.5),
                  'ref': "Schlesinger 2020, p.163"},
            'K': {'ideal': 2.4, 'range': (1.5, 3.5),
                  'ref': "Mason & Moore 1982, p.45"},
            'Mg': {'ideal': 1.2, 'range': (0.8, 2.0),
                  'ref': "Stumm & Morgan 2012, p.427"},
            'Ti': {'ideal': 0.4, 'range': (0.2, 0.6),
                  'ref': "Brimblecombe 2003, p.82"},
            'P': {'ideal': 0.06, 'range': (0.03, 0.10),
                  'ref': "Schlesinger 2020, p.148"},
            'N': {'ideal': 0.01, 'range': (0.005, 0.02),
                  'ref': "Stumm & Morgan 2012, p.430"},
            'C': {'ideal': 0.08, 'range': (0.05, 0.15),
                  'ref': "Schlesinger 2020, p.150"},
            # Additional elements
            'B': {'ideal': 0.003, 'range': (0.001, 0.005),
                 'ref': "Kabata-Pendias 2010, p.367"},
            'Zr': {'ideal': 0.019, 'range': (0.010, 0.030),
                  'ref': "Mason & Moore 1982, p.48"},
            'S': {'ideal': 0.05, 'range': (0.02, 0.08),
                 'ref': "Schlesinger 2020, p.152"}
        }

        # Life-supporting compounds with references
        self.life_supporting_compounds = {
            'SiO2': {'ideal': 58.23, 'range': (50.0, 65.0),
                    'ref': "Stumm & Morgan 2012, p.450"},
            'Al2O3': {'ideal': 15.12, 'range': (12.0, 18.0),
                     'ref': "Mason & Moore 1982, p.52"},
            'Fe2O3': {'ideal': 5.43, 'range': (4.0, 7.0),
                     'ref': "Schlesinger 2020, p.167"},
            'CaO': {'ideal': 4.82, 'range': (3.5, 6.0),
                   'ref': "Stumm & Morgan 2012, p.455"},
            'Na2O': {'ideal': 3.52, 'range': (2.5, 4.5),
                    'ref': "Mason & Moore 1982, p.55"},
            'K2O': {'ideal': 2.98, 'range': (2.0, 4.0),
                   'ref': "Schlesinger 2020, p.170"},
            'MgO': {'ideal': 2.12, 'range': (1.5, 3.0),
                   'ref': "Stumm & Morgan 2012, p.458"},
            'NaCl': {'ideal': 0.48, 'range': (0.3, 0.7),
                    'ref': "Mason & Moore 1982, p.58"},
            'DMS': {'ideal': 0.0001, 'range': (0.00005, 0.0002),
                   'ref': "Charlson et al., Nature 1987"}
        }

        # Toxic compounds with references
        self.toxic_compounds = {
            'As': {'threshold': 50, 'lethal': 500,
                  'ref': "WHO 2017, p.315"},
            'Hg': {'threshold': 1, 'lethal': 10,
                  'ref': "WHO 2017, p.389"},
            'Pb': {'threshold': 100, 'lethal': 1000,
                  'ref': "WHO 2017, p.398"},
            'Cd': {'threshold': 20, 'lethal': 200,
                  'ref': "WHO 2017, p.327"},
            'Cr6+': {'threshold': 50, 'lethal': 500,
                    'ref': "WHO 2017, p.340"},
            'CN-': {'threshold': 10, 'lethal': 100,
                   'ref': "WHO 2017, p.346"},
            'Cl': {'threshold': 250000, 'lethal': 500000,
                  'ref': "WHO 2017, p.334"},
            'H2S': {'threshold': 100, 'lethal': 1000,
                   'ref': "WHO 2017, p.372"}
        }

        # Biomarkers with references
        self.biomarkers = {
            'ATP': {'ref': "Schwieterman 2018, p.723"},
            'DNA': {'ref': "Schwieterman 2018, p.725"},
            'Proteins': {'ref': "Schwieterman 2018, p.727"},
            'Lipids': {'ref': "Schwieterman 2018, p.729"},
            'Chlorophyll': {'ref': "Schwieterman 2018, p.731"}
        }


class EnvironmentProfiles:
    """
    Defines environmental profiles for different location types,
    including both habitable and sterile environments.
    """

    def __init__(self):
        # Habitable environment ranges with references
        self.habitable_ranges = {
            'standard': {
                'temperature': (273, 323),  # K
                'pressure': (0.7, 1.2),     # atm
                'ph': (6.0, 8.0),
                'depth': (-100, 1000),      # m
                'humidity': (30, 70),       # %
                'radiation': (0, 1000),     # W/m²
                'ref': "Rothschild 2001"
            },
            'hot_springs': {
                'temperature': (303, 373),
                'pressure': (1.0, 1.5),
                'ph': (2.0, 9.0),
                'depth': (-50, 100),
                'humidity': (70, 100),
                'radiation': (0, 1200),
                'ref': "Brock Biology of Microorganisms 2018"
            },
            'deep_ocean': {
                'temperature': (271, 287),
                'pressure': (1.0, 1000.0),
                'ph': (7.5, 8.4),
                'depth': (-11000, -100),
                'humidity': (100, 100),
                'radiation': (0, 10),
                'ref': "Oceanography, Garrison 2012"
            },
            'high_altitude': {
                'temperature': (233, 293),
                'pressure': (0.1, 0.7),
                'ph': (5.0, 8.0),
                'depth': (3000, 9000),
                'humidity': (10, 40),
                'radiation': (1000, 1500),
                'ref': "Mountain Ecosystems, Körner 2003"
            }
        }

        # Known sterile environments with references
        self.sterile_environments = {
            'don_juan_pond': {
                'location': 'Antarctica',
                'coordinates': (-77.562778, 161.233333),
                'physical_ratio': 0.82,
                'chemical_ratio': 0.15,
                'elemental_ratio': 0.45,
                'limiting_factors': ['extreme_salinity', 'low_water_activity'],
                'ref': "Environmental Microbiology 2013"
            },
            'dallol_hotsprings': {
                'location': 'Ethiopia',
                'coordinates': (14.242889, 40.289583),
                'physical_ratio': 0.25,
                'chemical_ratio': 0.18,
                'elemental_ratio': 0.30,
                'limiting_factors': ['extreme_temperature', 'high_acidity'],
                'ref': "Nature Ecology & Evolution 2019"
            },
            'atacama_core': {
                'location': 'Chile',
                'coordinates': (-24.500000, -69.250000),
                'physical_ratio': 0.35,
                'chemical_ratio': 0.40,
                'elemental_ratio': 0.55,
                'limiting_factors': ['extreme_aridity', 'high_oxidants'],
                'ref': "Science 2003"
            }
        }


"""
Unified Habitability Analysis System - Analysis Components
Includes:
1. Physical Analysis
2. Chemical Analysis
3. PCA Analysis
4. Life Prediction
5. Visualization System
"""
# start of class Physical Analyzer


class PhysicalAnalyzer:
    """
    Analyzes physical parameters of environments for habitability assessment.
    Includes enhanced error handling and data validation.
    """

    def __init__(self, environment_profiles: EnvironmentProfiles):
        self.profiles = environment_profiles
        self.logger = logging.getLogger('UHAS.PhysicalAnalyzer')

    def analyze_physical_parameters(
        self, data: pd.Series, environment_type: str) -> Dict[str, float]:
        """Analyze physical parameters based on environment type with error handling"""
        try:
            if environment_type not in self.profiles.habitable_ranges:
                self.logger.warning(
                    f"Unknown environment type: {environment_type}. Using 'standard' ranges.")
                environment_type = 'standard'

            ranges = self.profiles.habitable_ranges[environment_type]
            scores = {}

            # Temperature analysis
            try:
                if 'temperature_c' in data:
                    temp_k = float(data['temperature_c']) + 273.15
                    scores['temperature'] = self._calculate_range_score(
                        temp_k, ranges['temperature'])
            except Exception as e:
                self.logger.error(
                    f"Error calculating temperature score: {str(e)}")
                scores['temperature'] = 0.0

            # Pressure analysis
            try:
                if 'pressure_atm' in data:
                    scores['pressure'] = self._calculate_range_score(
                        float(data['pressure_atm']), ranges['pressure'])
            except Exception as e:
                self.logger.error(
                    f"Error calculating pressure score: {str(e)}")
                scores['pressure'] = 0.0

            # pH analysis
            try:
                if 'ph' in data:
                    scores['ph'] = self._calculate_range_score(
                        float(data['ph']), ranges['ph'])
            except Exception as e:
                self.logger.error(f"Error calculating pH score: {str(e)}")
                scores['ph'] = 0.0

            # Elevation analysis
            try:
                if 'elevation_m' in data:
                    scores['elevation'] = self._calculate_range_score(
                        float(data['elevation_m']), ranges['depth'])
            except Exception as e:
                self.logger.error(
                    f"Error calculating elevation score: {str(e)}")
                scores['elevation'] = 0.0

            # Ensure we have at least some scores
            if not scores:
                raise ValueError("No physical parameters could be analyzed")

            return scores

        except Exception as e:
            self.logger.error(
                f"Error in physical parameter analysis: {str(e)}")
            # Return default scores instead of raising an error
            return {
                'temperature': 0.0,
                'pressure': 0.0,
                'ph': 0.0,
                'elevation': 0.0
            }

    def _calculate_range_score(
        self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calculate score based on value's position within range with error handling"""
        try:
            min_val, max_val = range_tuple
            if min_val <= value <= max_val:
                return 1.0
            else:
                return max(0, 1 - min(abs(value - min_val),
                           abs(value - max_val)) / (max_val - min_val))
        except Exception as e:
            self.logger.error(f"Error calculating range score: {str(e)}")
            return 0.0
# end of class Physical Analyzer
# start of class Chemical Analyzer


class ChemicalAnalyzer:
    """Analyzes chemical composition and toxicity levels with comprehensive error handling and detailed analysis of elements, compounds, and biomarkers."""
    
    def __init__(self, chemical_profile: ChemicalProfile):
        # Initialize with a chemical profile
        self.profile = chemical_profile
        self.logger = logging.getLogger('UHAS.ChemicalAnalyzer')

    def analyze_composition(self, data: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive chemical analysis of sample data.

        Parameters:
        -----------
        data : pd.Series
            Series containing chemical composition data.

        Returns:
        --------
        Dict containing analysis results for elements, compounds, toxicity, and biomarkers.
        """
        try:
            results = {
                'elements': self._analyze_elements(data),
                'compounds': self._analyze_compounds(data),
                'toxicity': self._analyze_toxicity(data),
                'biomarkers': self._analyze_biomarkers(data)
            }
            results['overall_chemical_score'] = self._calculate_overall_score(results)
            return results
        except Exception as e:
            self.logger.error(f"Error in chemical composition analysis: {str(e)}")
            return self._get_default_results()

    def _analyze_elements(self, data: pd.Series) -> Dict[str, float]:
        """
        Analyze essential element concentrations.
        """
        scores = {}
        try:
            for element, ref_data in self.profile.essential_elements.items():
                if element in data.index:
                    try:
                        value = float(data[element])
                        ideal_range = ref_data['range']
                        scores[element] = self._calculate_range_score(value, ideal_range)
                    except Exception as e:
                        self.logger.warning(f"Error analyzing element {element}: {str(e)}")
                        scores[element] = 0.0
            return scores
        except Exception as e:
            self.logger.error(f"Error in element analysis: {str(e)}")
            return {}

    def _analyze_compounds(self, data: pd.Series) -> Dict[str, float]:
        """Analyze life-supporting compound concentrations."""
        scores = {}
        try:
            for compound, ref_data in self.profile.life_supporting_compounds.items():
                if compound in data.index:
                    try:
                        value = float(data[compound])
                        ideal_range = ref_data['range']
                        scores[compound] = self._calculate_range_score(value, ideal_range)
                    except Exception as e:
                        self.logger.warning(f"Error analyzing compound {compound}: {str(e)}")
                        scores[compound] = 0.0
            return scores
        except Exception as e:
            self.logger.error(f"Error in compound analysis: {str(e)}")
            return {}

    def _analyze_toxicity(self, data: pd.Series) -> Dict[str, float]:
        """Analyze toxic compound levels"""
        scores = {}
        try:
            for compound, ref_data in self.profile.toxic_compounds.items():
                if compound in data.index:
                    try:
                        value = float(data[compound])
                        threshold = ref_data['threshold']
                        lethal = ref_data['lethal']
                        scores[compound] = self._calculate_toxicity_score(value, threshold, lethal)
                    except Exception as e:
                        self.logger.warning(f"Error analyzing toxicity of {compound}: {str(e)}")
                        scores[compound] = 0.0

            return scores
        except Exception as e:
            self.logger.error(f"Error in toxicity analysis: {str(e)}")
            return {}

    def _analyze_biomarkers(self, data: pd.Series) -> Dict[str, float]:
        """Analyze biomarker presence"""
        scores = {}
        try:
            for marker in self.profile.biomarkers:
                marker_key = f"{marker}_presence"
                if marker_key in data.index:
                    try:
                        scores[marker] = float(data[marker_key])
                    except Exception as e:
                        self.logger.warning(f"Error analyzing biomarker {marker}: {str(e)}")
                        scores[marker] = 0.0

            return scores
        except Exception as e:
            self.logger.error(f"Error in biomarker analysis: {str(e)}")
            return {}

    def _calculate_toxicity_score(self, value: float, threshold: float, lethal: float) -> float:
        """Calculate toxicity score based on thresholds"""
        try:
            if value <= threshold:
                return 1.0
            elif value >= lethal:
                return 0.0
            else:
                return max(0, 1 - ((value - threshold) / (lethal - threshold)))
        except Exception as e:
            self.logger.error(f"Error calculating toxicity score: {str(e)}")
            return 0.0

    def _calculate_range_score(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calculate score based on value's position within range"""
        try:
            min_val, max_val = range_tuple
            if min_val <= value <= max_val:
                return 1.0
            else:
                return max(0, 1 - min(abs(value - min_val), abs(value - max_val)) / (max_val - min_val))
        except Exception as e:
            self.logger.error(f"Error calculating range score: {str(e)}")
            return 0.0

    def _calculate_overall_score(self, results: Dict[str, Dict[str, float]]) -> float:
        """Calculate weighted overall chemical score"""
        try:
            weights = {
                'elements': 0.3,
                'compounds': 0.3,
                'toxicity': 0.2,
                'biomarkers': 0.2
            }

            scores = []
            weights_used = []

            for category, weight in weights.items():
                if results[category]:
                    category_score = np.mean(list(results[category].values()))
                    scores.append(category_score)
                    weights_used.append(weight)

            if not scores:
                return 0.0

            # Normalize weights
            weights_used = np.array(weights_used) / sum(weights_used)
            return float(np.average(scores, weights=weights_used))

        except Exception as e:
            self.logger.error(f"Error calculating overall chemical score: {str(e)}")
            return 0.0

    def _get_default_results(self) -> Dict[str, Any]:
        """Return default results structure when analysis fails"""
        return {
            'elements': {},
            'compounds': {},
            'toxicity': {},
            'biomarkers': {},
            'overall_chemical_score': 0.0
        }

    def get_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of chemical analysis results"""
        try:
            return {
                'overall_score': results['overall_chemical_score'],
                'element_count': len(results['elements']),
                'compound_count': len(results['compounds']),
                'toxic_compounds_detected': len(results['toxicity']),
                'biomarkers_detected': len(results['biomarkers']),
                'limiting_factors': self._identify_limiting_factors(results)
            }
        except Exception as e:
            self.logger.error(f"Error generating analysis summary: {str(e)}")
            return {}

    def _identify_limiting_factors(self, results: Dict[str, Any]) -> List[str]:
        """Identify factors that might be limiting habitability"""
        limiting_factors = []
        try:
            # Check elements
            poor_elements = [elem for elem, score in results['elements'].items() if score < 0.5]
            if poor_elements:
                limiting_factors.append(f"Deficient elements: {', '.join(poor_elements)}")

            # Check toxic compounds
            high_toxicity = [comp for comp, score in results['toxicity'].items() if score < 0.3]
            if high_toxicity:
                limiting_factors.append(f"Toxic levels: {', '.join(high_toxicity)}")

            return limiting_factors
        except Exception as e:
            self.logger.error(f"Error identifying limiting factors: {str(e)}")
            return ["Error in analysis"]

class PCAAnalyzer:
    """
    Performs PCA analysis on environmental and chemical data.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.feature_names = []
        
    def perform_pca(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA analysis on dataset"""
        # Prepare data
        numerical_data = data.select_dtypes(include=[np.number])
        self.feature_names = numerical_data.columns
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numerical_data)
        
        # Perform PCA
        transformed_data = self.pca.fit_transform(scaled_data)
        
        # Calculate explained variance
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Get component loadings
        loadings = self.pca.components_
        
        # Create loadings DataFrame
        loadings_df = pd.DataFrame(
            loadings.T,
            columns=[f'PC{i+1}' for i in range(loadings.shape[0])],
            index=self.feature_names
        )
        
        return {
            'transformed_data': transformed_data,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'loadings': loadings_df,
            'n_components': self.pca.n_components_,
            'feature_importance': self._calculate_feature_importance(loadings_df)
        }
    
    def _calculate_feature_importance(self, loadings_df: pd.DataFrame) -> pd.Series:
        """Calculate overall feature importance across components"""
        # Use absolute values of loadings
        abs_loadings = loadings_df.abs()
        
        # Weight by explained variance
        weighted_loadings = abs_loadings.multiply(
            self.pca.explained_variance_ratio_, axis=1)
        
        # Sum across components
        importance = weighted_loadings.sum(axis=1)
        return importance.sort_values(ascending=False)
#start class ReferenceData
class ReferenceData:
    """Manages reference environment data and comparisons"""
    def __init__(self):
        self.normal_environments = {
            'Amazon_Rainforest': {
                'temperature_c': 25.5,
                'pressure_atm': 1.0,
                'ph': 6.5,
                'elevation_m': 100,
                'habitability_score': 0.95,
                'primary_elements': {'C': 0.12, 'N': 0.01, 'P': 0.07},
                'reference': "NASA Earth Observatory, 2022",
                'key_characteristics': "High biodiversity, stable temperature"
            }
            # ... other normal environments
        }
        
        self.extreme_environments = {
            'Yellowstone_Geysers': {
                'temperature_c': 92.0,
                'pressure_atm': 1.2,
                'ph': 8.2,
                'elevation_m': 2357,
                'habitability_score': 0.45,
                'reference': "Yellowstone Research Centre, 2023",
                'key_characteristics': "Thermophilic bacteria present",
                'extremophile_adjustment': 0.3
            }
            # ... other extreme environments
        }

    def get_reference_data(self, environment_type: str = 'all') -> dict:
        """Get reference data based on environment type"""
        if environment_type == 'normal':
            return self.normal_environments
        elif environment_type == 'extreme':
            return self.extreme_environments
        return {**self.normal_environments, **self.extreme_environments}

    def get_comparison_data(self, location_data: dict) -> dict:
        """Get relevant comparison environments based on location characteristics"""
        # Implementation for selecting relevant comparisons
        #start class ScoringSystem
class ScoringSystem:
    """Handles scoring calculations with temperature penalties and adjustments"""
    def __init__(self):
        self.weights = {
            'physical': {'temperature': 0.35, 'pressure': 0.25, 'ph': 0.25, 'elevation': 0.15},
            'chemical': {'essential_elements': 0.4, 'compounds': 0.3, 'toxicity': 0.2, 'biomarkers': 0.1}
        }
        
    def calculate_score(self, data: dict, environment_type: str) -> dict:
        """Calculate comprehensive habitability score"""
        # Implementation for score calculation

    def apply_temperature_penalty(self, temperature: float) -> float:
        """Apply temperature-based penalties"""
        # Implementation for temperature penalties
        #start class HabitabilityVisualizer
class HabitabilityVisualizer:
    """
    Creates comprehensive visualizations for habitability analysis.
    """
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        self.logger = logging.getLogger('UHAS.HabitabilityVisualizer')
        self.reference_data = ReferenceData()
        
def create_dashboard(self, results: Dict[str, Any], save_path: Optional[str] = None):
    """Create comprehensive visualization dashboard"""
    try:
        # Get first location's data for visualization
        location_data = next((data for key, data in results.items()
                             if key != 'pca_analysis'), None)
        
        if not location_data:
            self.logger.error("No valid location data found for visualization")
            return

        # Get reference data based on environment type
        ref_data = self.reference_data.get_reference_data(
            environment_type=location_data.get('environment_type', 'standard')
        )

        fig = plt.figure(figsize=(20, 15))
        
        # Physical parameters plot
        plt.subplot(2, 3, 1)
        self._plot_physical_parameters(location_data['physical_scores'])
        
        # Chemical composition plot
        plt.subplot(2, 3, 2)
        self._plot_chemical_composition(location_data['chemical_analysis'])
        
        # PCA results plot if available
        plt.subplot(2, 3, 3)
        if 'pca_analysis' in results:
            self._plot_pca_results(results['pca_analysis'])
        else:
            plt.text(0.5, 0.5, 'PCA not available\n(requires multiple locations)',
                    ha='center', va='center')
        
        # Toxicity levels plot
        plt.subplot(2, 3, 4)
        self._plot_toxicity_levels(location_data['chemical_analysis'].get('toxicity', {}))
        
        # Overall habitability score
        plt.subplot(2, 3, 5)
        self._plot_habitability_score(location_data.get('overall_score', 0))
        
        # Feature importance from PCA if available
        plt.subplot(2, 3, 6)
        if 'pca_analysis' in results:
            self._plot_feature_importance(results['pca_analysis'].get('feature_importance', pd.Series()))
        else:
            plt.text(0.5, 0.5, 'Feature importance not available\n(requires PCA analysis)',
                    ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        self.logger.error(f"Error creating dashboard: {str(e)}")
        plt.close('all')  # Clean up any open figures

    def _plot_physical_parameters(self, scores: Dict[str, float]):
        """Create radar plot for physical parameters"""
        categories = list(scores.keys())
        values = list(scores.values())
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax = plt.gca()
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title('Physical Parameters')

    def _plot_chemical_composition(self, chemical_results: Dict[str, Any]):
        """Create bar plot for chemical composition"""
        elements = chemical_results['elements']
        plt.bar(elements.keys(), elements.values())
        plt.xticks(rotation=45)
        plt.title('Chemical Composition Scores')

    def _plot_pca_results(self, pca_results: Dict[str, Any]):
        """Create scree plot for PCA results"""
        plt.plot(range(1, len(pca_results['explained_variance']) + 1),
                pca_results['explained_variance'], 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Scree Plot')

    def _plot_toxicity_levels(self, toxicity_scores: Dict[str, float]):
        """Create horizontal bar plot for toxicity levels"""
        plt.barh(list(toxicity_scores.keys()),
                list(toxicity_scores.values()),
                color=['green' if v > 0.7 else 'yellow' if v > 0.3 else 'red'
                      for v in toxicity_scores.values()])
        plt.title('Toxicity Levels')

    def _plot_habitability_score(self, score: float):
        """Create gauge plot for overall habitability score"""
        colors = ['red', 'yellow', 'green']
        plt.pie([score, 1-score], colors=['green', 'lightgray'],
               startangle=90, counterclock=False)
        plt.title(f'Overall Habitability Score: {score:.2f}')

    def _plot_feature_importance(self, importance: pd.Series):
        """Create bar plot for feature importance"""
        importance.plot(kind='bar')
        plt.title('Feature Importance from PCA')
        plt.xticks(rotation=45)

"""
Unified Habitability Analysis System - Integration and Execution
Includes:
1. Integrated Analysis System
2. Data Handling
3. Results Management
4. Main Execution
"""

class IntegratedAnalyzer:
    """
    Main class that integrates all analysis components and manages workflow.
    """
    def __init__(self):
        self.logger = logging.getLogger('UHAS.IntegratedAnalyzer')
        self.chemical_profile = ChemicalProfile()
        self.environment_profiles = EnvironmentProfiles()
        self.physical_analyzer = PhysicalAnalyzer(self.environment_profiles)
        self.chemical_analyzer = ChemicalAnalyzer(self.chemical_profile)
        self.pca_analyzer = PCAAnalyzer()
        self.visualizer = HabitabilityVisualizer()

    def analyze_location(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive analysis on location data with enhanced error handling"""
        results = {}
        
        try:
            for idx, row in data.iterrows():
                location_name = str(row.get('location_name', f'Location_{idx}'))
                self.logger.info(f"Processing location: {location_name}")
                
                # Get environment type with fallback
                environment_type = str(row.get('environment_type', 'standard'))
                
                try:
                    # Physical analysis
                    self.logger.debug(f"Starting physical analysis for {location_name}")
                    physical_scores = self.physical_analyzer.analyze_physical_parameters(
                        row, environment_type)
                    
                    # Chemical analysis
                    self.logger.debug(f"Starting chemical analysis for {location_name}")
                    chemical_results = self.chemical_analyzer.analyze_composition(row)
                    
                    # Calculate overall score
                    physical_score = np.mean(list(physical_scores.values()))
                    chemical_score = chemical_results.get('overall_chemical_score', 0.0)
                    overall_score = 0.5 * physical_score + 0.5 * chemical_score
                    
                    # Store results
                    results[location_name] = {
                        'physical_scores': physical_scores,
                        'chemical_analysis': chemical_results,
                        'environment_type': environment_type,
                        'overall_score': overall_score,  # Add overall score
                        'physical_score': physical_score,  # Add individual scores
                        'chemical_score': chemical_score
                    }
                    self.logger.debug(f"Analysis complete for {location_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing location {location_name}: {str(e)}")
                    # Add default results for failed analysis
                    results[location_name] = {
                        'physical_scores': {},
                        'chemical_analysis': {},
                        'environment_type': environment_type,
                        'overall_score': 0.0,
                        'physical_score': 0.0,
                        'chemical_score': 0.0
                    }

            # Add PCA results if multiple locations
            if len(results) > 1:
                try:
                    pca_results = self.pca_analyzer.perform_pca(data)
                    results['pca_analysis'] = pca_results
                except Exception as e:
                    self.logger.error(f"Error in PCA analysis: {str(e)}")

            return results
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            return {}	

    def _calculate_overall_scores(self, results: Dict[str, Any]):
        """Calculate overall habitability scores with error handling"""
        for location, result in results.items():
            if location != 'pca_analysis':
                try:
                    physical_scores = result.get('physical_scores', {})
                    chemical_analysis = result.get('chemical_analysis', {})
                    
                    if physical_scores and isinstance(physical_scores, dict):
                        physical_score = np.mean([
                            float(score) for score in physical_scores.values()
                            if isinstance(score, (int, float))
                        ])
                    else:
                        physical_score = 0.0
                    
                    chemical_score = float(chemical_analysis.get('overall_chemical_score', 0.0))
                    
                    # Weighted average of physical and chemical scores
                    result['overall_score'] = 0.5 * physical_score + 0.5 * chemical_score
                    
                except Exception as e:
                    self.logger.error(f"Error calculating overall score for {location}: {str(e)}")
                    result['overall_score'] = 0.0
#start of class DataHandler
class DataHandler:
    """
    Handles data input/output and validation.
    """
    def __init__(self):
        # Define required columns and their expected data types
        self.required_columns = {
            'location_name': str,
            'environment_type': str,
            'temperature_c': float,
            'pressure_atm': float,
            'ph': float,
            'elevation_m': float
        }
        self.logger = logging.getLogger('UHAS.DataHandler')

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate input data"""
        try:
            # Check file extension and load the data
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel.")
            
            # Validate the data after loading
            data = self._validate_data(data)  # Ensure the data is valid
            data = self._fill_missing_values(data)  # Fill missing values

            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the data to ensure it has the necessary columns and correct data types"""
        try:
            # Step 1: Check if all required columns are present
            missing_columns = [col for col in self.required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

            # Step 2: Ensure 'location_name' and 'environment_type' are of type 'str' and handle missing values
            for col in ['location_name', 'environment_type']:
                if col in data.columns:
                    # Convert to string type and handle missing values
                    data[col] = data[col].fillna('Unknown').astype(str)
                else:
                    raise ValueError(f"The '{col}' column is missing in the data.")

            # Step 3: Check that other columns have the correct data types
            for column, dtype in self.required_columns.items():
                if column not in ['location_name', 'environment_type']:  # Skip already validated columns
                    try:
                        data[column] = pd.to_numeric(data[column], errors='raise')
                    except:
                        self.logger.error(f"Column '{column}' contains non-numeric values")
                        raise TypeError(f"Column '{column}' must be of type {dtype}")

            return data
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            raise

    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in specific columns with default values."""
        try:
            defaults = {
                'temperature_c': 20.0,
                'pressure_atm': 1.0,
                'ph': 7.0,
                'elevation_m': 0.0
            }
            
            for col, default in defaults.items():
                if col in data and data[col].isnull().any():
                    self.logger.warning(f"Filling missing values in '{col}' with {default}")
                    data[col].fillna(default, inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error filling missing values: {str(e)}")
            raise
#start of class ResultsManager
class ResultsManager:
    """
    Manages analysis results and exports.
    """
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger('UHAS.ResultsManager')
    
    def export_results(self, results: Dict[str, Any], output_dir: str):
        """Export analysis results to various formats"""
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Export JSON results
            self._export_json(results, output_dir)
            
            # Export CSV summary
            self._export_csv_summary(results, output_dir)
            
            # Export visualizations
            self._export_visualizations(results, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            raise
    
    def _export_json(self, results: Dict[str, Any], output_dir: str):
        """Export detailed results to JSON"""
        try:
            json_path = Path(output_dir) / f"detailed_results_{self.timestamp}.json"
            
            # Convert numpy arrays and other non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error exporting JSON: {str(e)}")
            raise
    
    def _export_csv_summary(self, results: Dict[str, Any], output_dir: str):
        """Export summary results to CSV"""
        try:
            summary_data = []
            
            for location, result in results.items():
                if location != 'pca_analysis':
                    summary_data.append({
                        'Location': location,
                        'Environment_Type': result['environment_type'],
                        'Overall_Score': result['overall_score'],
                        'Physical_Score': np.mean(list(result['physical_scores'].values())),
                        'Chemical_Score': result['chemical_analysis']['overall_chemical_score']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            csv_path = Path(output_dir) / f"summary_results_{self.timestamp}.csv"
            summary_df.to_csv(csv_path, index=False)
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {str(e)}")
            raise
    
    def _export_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Export visualization plots with error handling"""
        try:
            vis_path = Path(output_dir) / f"visualization_{self.timestamp}.png"
            visualizer = HabitabilityVisualizer()
            visualizer.create_dashboard(results, str(vis_path))
            
            # Check if we have valid results
            if not any(key != 'pca_analysis' for key in results.keys()):
                self.logger.warning("No valid location data for visualization")
                return
                
            visualizer.create_dashboard(results, str(vis_path))
            
        except Exception as e:
            self.logger.error(f"Error exporting visualizations: {str(e)}")
            self.logger.debug("Visualization error details:", exc_info=True)
    
    @staticmethod
    def _make_serializable(obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ResultsManager._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResultsManager._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj
#end of class DataHandler
def main():
    """Main execution function"""
    args = dict(input_file="example_data.csv", output="results")

    try:
        # Initialize components
        data_handler = DataHandler()
        analyzer = IntegratedAnalyzer()
        results_manager = ResultsManager()
        
        # Load and analyze data
        logger.info("Loading data...")
        data = data_handler.load_data(args['input_file'])
        
        logger.info("Performing analysis...")
        results = analyzer.analyze_location(data)
        
        # Export results
        logger.info("Exporting results...")
        results_manager.export_results(results, args['output'])
        
        logger.info(f"Analysis complete. Results saved to {args['output']}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    logger.info("Starting Unified Habitability Analysis System")
    
    # Example data
    example_data = pd.DataFrame({
        'location_name': ['Yellowstone_Geyser', 'Atacama_Desert'],
        'environment_type': ['hot_springs', 'standard'],
        'temperature_c': [92, 25],
        'pressure_atm': [1.2, 0.9],
        'ph': [8.2, 7.5],
        'elevation_m': [2357, 2408],
        'Si': [27.8, 28.1],
        'Al': [8.1, 8.3],
        'Fe': [3.8, 3.9],
        'B': [0.003, 0.004],
        'Zr': [0.019, 0.020],
        'S': [0.05, 0.06],
        'As': [25, 30],
        'Cl': [200000, 220000],
        'DMS': [0.0001, 0.00012]
    })
    
    # Save example data
    example_data.to_csv('example_data.csv', index=False)
    
    # Run analysis
    main()
