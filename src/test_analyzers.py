```python
"""
test_analyzers.py
Properly indented test script for chemical and integrated analyzers
"""

import pandas as pd
import numpy as np
from unified_analyzer import ChemicalAnalyzer, ChemicalProfile, IntegratedAnalyzer, EnvironmentProfiles

def test_chemical_analyzer():
    print("\n=== Testing Chemical Analyzer ===")
    # Create chemical profile
    profile = ChemicalProfile()
    
    # Initialize analyzer
    analyzer = ChemicalAnalyzer(profile)
    
    # Load sample data
    try:
        data = pd.read_csv('sample_data.csv')
        print(f"Loaded {len(data)} samples")
        
        # Test each sample
        for idx, row in data.iterrows():
            print(f"\nAnalyzing sample: {row['location_name']}")
            results = analyzer.analyze_composition(row)
            
            print(f"Overall Chemical Score: {results['overall_chemical_score']:.2f}")
            print("Element Scores:")
            for element, score in results['elements'].items():
                print(f"  {element}: {score:.2f}")
            
            summary = analyzer.get_analysis_summary(results)
            print("\nLimiting Factors:")
            for factor in summary.get('limiting_factors', []):
                print(f"  - {factor}")
                
    except Exception as e:
        print(f"Error in chemical analysis: {str(e)}")

def test_integrated_analyzer():
    print("\n=== Testing Integrated Analyzer ===")
    try:
        # Initialize analyzer
        analyzer = IntegratedAnalyzer()
        
        # Load sample data
        data = pd.read_csv('sample_data.csv')
        print(f"Loaded {len(data)} samples")
        
        # Perform analysis
        results = analyzer.analyze_location(data)
        
        # Print results
        for location, result in results.items():
            if location != 'pca_analysis':
                print(f"\nLocation: {location}")
                print(f"Environment Type: {result['environment_type']}")
                print(f"Overall Score: {result.get('overall_score', 0):.2f}")
                print("Physical Scores:")
                for param, score in result['physical_scores'].items():
                    print(f"  {param}: {score:.2f}")
                print(f"Chemical Score: {result['chemical_analysis']['overall_chemical_score']:.2f}")
                
    except Exception as e:
        print(f"Error in integrated analysis: {str(e)}")

if __name__ == "__main__":
    print("Starting Analyzer Tests...")
    test_chemical_analyzer()
    test_integrated_analyzer()
    print("\nTests completed.")
```
