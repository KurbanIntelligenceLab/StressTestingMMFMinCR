import os
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_scripts.base_classes import MultiModelAnnotationPipeline
import requests
import time
from typing import List, Dict, Optional, Set
import numpy as np
import random 
from llm_scripts.base_classes import Metrics, MaterialProperties, Result, Sample

# =============================
# Analysis Classes
# =============================
class MaterialAnalysis:
    """Class for material-specific analysis and metrics"""
    
    @staticmethod
    def calculate_complexity(atomic_count: int, cell_volume: float) -> float:
        """Calculate material complexity based on atomic density"""
        return atomic_count / cell_volume
    
    @staticmethod
    def calculate_similarity(material1_props: Dict, material2_props: Dict) -> float:
        """Calculate similarity between two materials based on their properties"""
        # Normalize properties for comparison
        props1 = np.array([
            material1_props['atomic_count'],
            material1_props['cell_volume'],
            material1_props['lattice_a'],
            material1_props['lattice_b'],
            material1_props['lattice_c'],
            material1_props['nn_distance'],
            material1_props['density']
        ])
        props2 = np.array([
            material2_props['atomic_count'],
            material2_props['cell_volume'],
            material2_props['lattice_a'],
            material2_props['lattice_b'],
            material2_props['lattice_c'],
            material2_props['nn_distance'],
            material2_props['density']
        ])
        
        # Calculate cosine similarity
        similarity = np.dot(props1, props2) / (np.linalg.norm(props1) * np.linalg.norm(props2))
        return float(similarity)

class RValueAnalysis:
    """Class for R-value specific analysis"""
    
    @staticmethod
    def analyze_r_progression(results: List[Result]) -> Dict:
        """Analyze how errors change with increasing R values"""
        r_values = sorted(set(r.held_out_r for r in results))
        progression = {}
        
        for r in r_values:
            r_results = [res for res in results if res.held_out_r == r]
            if r_results:
                avg_error = np.mean([res.metrics.average_error for res in r_results])
                std_error = np.std([res.metrics.average_error for res in r_results])
                progression[r] = {
                    'mean_error': float(avg_error),
                    'std_error': float(std_error)
                }
        
        return progression
    
    @staticmethod
    def calculate_interpolation_performance(results: List[Result]) -> Dict:
        """Calculate how well the model interpolates between R values"""
        r_values = sorted(set(r.held_out_r for r in results))
        interpolation = {}
        
        for i in range(len(r_values) - 2):
            r1, r2, r3 = r_values[i:i+3]
            r1_results = [res for res in results if res.held_out_r == r1]
            r2_results = [res for res in results if res.held_out_r == r2]
            r3_results = [res for res in results if res.held_out_r == r3]
            
            if r1_results and r2_results and r3_results:
                r1_error = np.mean([res.metrics.average_error for res in r1_results])
                r2_error = np.mean([res.metrics.average_error for res in r2_results])
                r3_error = np.mean([res.metrics.average_error for res in r3_results])
                
                # Check if r2 error is lower than average of r1 and r3
                expected_error = (r1_error + r3_error) / 2
                interpolation[f"{r1}-{r2}-{r3}"] = {
                    'actual_error': float(r2_error),
                    'expected_error': float(expected_error),
                    'improvement': float(expected_error - r2_error)
                }
        
        return interpolation

class Visualization:
    """Class for collecting and saving numerical data for later visualization"""
    
    @staticmethod
    def collect_error_distribution(results: List[Result], output_dir: str):
        """Collect error distribution data for each property"""
        # Collect all errors
        errors = {}
        for result in results:
            for field_name in Metrics.model_fields:  # Access from class instead of instance
                if field_name.endswith('_error') and getattr(result.metrics, field_name) is not None:
                    if field_name not in errors:
                        errors[field_name] = []
                    errors[field_name].append(getattr(result.metrics, field_name))
        
        # Calculate statistics
        error_data = {
            'property_errors': errors,
            'statistics': {
                field: {
                    'mean': float(np.mean(errors[field])),
                    'std': float(np.std(errors[field])),
                    'min': float(np.min(errors[field])),
                    'max': float(np.max(errors[field])),
                    'median': float(np.median(errors[field])),
                    'q1': float(np.percentile(errors[field], 25)),
                    'q3': float(np.percentile(errors[field], 75))
                }
                for field in errors.keys()
            }
        }
        
        # Save numerical data
        with open(os.path.join(output_dir, 'error_distribution_data.json'), 'w') as f:
            json.dump(error_data, f, indent=2)
    
    @staticmethod
    def collect_correlation_data(results: List[Result], output_dir: str):
        """Collect correlation data between properties"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all errors
        errors = {}
        for result in results:
            for field_name in Metrics.model_fields:
                if field_name.endswith('_error') and getattr(result.metrics, field_name) is not None:
                    if field_name not in errors:
                        errors[field_name] = []
                    value = getattr(result.metrics, field_name)
                    if np.isfinite(value):  # Only include finite values
                        errors[field_name].append(value)
        
        # Filter out properties with insufficient data
        valid_properties = {
            prop: values for prop, values in errors.items()
            if len(values) > 1 and all(np.isfinite(v) for v in values)
        }
        
        if not valid_properties:
            print("Warning: No valid properties found for correlation analysis")
            return
        
        # Create a matrix of error values
        property_names = list(valid_properties.keys())
        error_matrix = np.array([valid_properties[prop] for prop in property_names])
        
        # Calculate correlation matrix
        try:
            # Handle NaN and inf values
            error_matrix = np.nan_to_num(error_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate means and standard deviations
            means = np.mean(error_matrix, axis=1, keepdims=True)
            stddev = np.std(error_matrix, axis=1, keepdims=True)
            
            # Avoid division by zero and handle small values
            stddev = np.where(stddev < 1e-10, 1.0, stddev)
            
            # Center the data safely using masked arrays
            centered = np.ma.masked_invalid(error_matrix - means)
            
            # Calculate correlation matrix manually to avoid warnings
            c = np.zeros((len(property_names), len(property_names)))
            for i in range(len(property_names)):
                for j in range(len(property_names)):
                    if i != j:  # Don't correlate with self
                        # Extract scalar values from arrays
                        std_i = float(stddev[i][0])
                        std_j = float(stddev[j][0])
                        
                        # Use masked arrays for safe multiplication
                        masked_i = np.ma.masked_invalid(centered[i])
                        masked_j = np.ma.masked_invalid(centered[j])
                        
                        # Calculate correlation only on valid values
                        valid_mask = ~(masked_i.mask | masked_j.mask)
                        if np.any(valid_mask):
                            # Ensure we have enough valid values for correlation
                            numerator = np.sum(masked_i[valid_mask] * masked_j[valid_mask])
                            denominator = std_i * std_j * np.sum(valid_mask)
                            if denominator != 0:
                                c[i, j] = float(numerator / denominator)
            
            # Set diagonal to 1.0
            np.fill_diagonal(c, 1.0)
            
            # Save numerical data
            correlation_data = {
                'correlation_matrix': c.tolist(),
                'property_names': [field.replace('_error', '') for field in property_names],
                'raw_errors': {field: valid_properties[field] for field in property_names}
            }
            
            with open(os.path.join(output_dir, 'correlation_data.json'), 'w') as f:
                json.dump(correlation_data, f, indent=2)
                
        except Exception as e:
            print(f"Error calculating correlation matrix: {str(e)}")
            return
    
    @staticmethod
    def collect_material_performance(results: List[Result], output_dir: str):
        """Collect performance data across materials and R values"""
        materials = sorted(set(r.material for r in results))
        r_values = sorted(set(r.held_out_r for r in results))
        
        # Create performance matrix and collect detailed data
        performance = np.zeros((len(materials), len(r_values)))
        performance_data = {}
        
        for i, material in enumerate(materials):
            performance_data[material] = {}
            for j, r_value in enumerate(r_values):
                material_r_results = [r for r in results if r.material == material and r.held_out_r == r_value]
                if material_r_results:
                    errors = [r.metrics.average_error for r in material_r_results]
                    performance[i, j] = np.mean(errors)
                    performance_data[material][r_value] = {
                        'mean_error': float(np.mean(errors)),
                        'std_error': float(np.std(errors)),
                        'min_error': float(np.min(errors)),
                        'max_error': float(np.max(errors)),
                        'num_samples': len(errors),
                        'individual_errors': [float(e) for e in errors]
                    }
        
        # Save numerical data
        performance_data['matrix'] = performance.tolist()
        performance_data['materials'] = materials
        performance_data['r_values'] = r_values
        
        with open(os.path.join(output_dir, 'material_performance_data.json'), 'w') as f:
            json.dump(performance_data, f, indent=2)
    
    @staticmethod
    def collect_knowledge_metrics(results: List[Result], output_dir: str):
        """Collect knowledge evaluation metrics data"""
        # Collect knowledge metrics
        metrics = {
            'prediction_consistency': [],
            'physical_law_compliance': [],
            'format_faithfulness': [],
            'hallucination_score': []
        }
        
        for result in results:
            for metric in metrics.keys():
                value = getattr(result.metrics, metric)
                if value is not None:
                    metrics[metric].append(value)
        
        # Calculate statistics
        knowledge_data = {
            'metrics': {
                metric: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
                for metric, values in metrics.items()
            },
            'raw_values': metrics
        }
        
        # Save numerical data
        with open(os.path.join(output_dir, 'knowledge_metrics_data.json'), 'w') as f:
            json.dump(knowledge_data, f, indent=2)
    
    @staticmethod
    def collect_metric_correlations(results: List[Result], output_dir: str):
        """Collect correlations between different metrics"""
        # Collect all metrics
        metrics_data = {}
        for result in results:
            for field_name in Metrics.model_fields:  # Access from class instead of instance
                value = getattr(result.metrics, field_name)
                if value is not None and not isinstance(value, (dict, list)):
                    if field_name not in metrics_data:
                        metrics_data[field_name] = []
                    metrics_data[field_name].append(float(value))
        
        # Calculate correlation matrix
        metric_names = list(metrics_data.keys())
        corr_matrix = np.corrcoef([metrics_data[name] for name in metric_names])
        
        # Save numerical data
        correlation_data = {
            'correlation_matrix': corr_matrix.tolist(),
            'metric_names': metric_names,
            'raw_values': metrics_data
        }
        with open(os.path.join(output_dir, 'metric_correlations_data.json'), 'w') as f:
            json.dump(correlation_data, f, indent=2)
    
    @staticmethod
    def save_comprehensive_metrics(results: List[Result], output_dir: str):
        """Save all available metrics in a comprehensive format"""
        comprehensive_data = {
            'summary': {
                'total_samples': len(results),
                'materials': sorted(set(r.material for r in results)),
                'r_values': sorted(set(r.held_out_r for r in results)),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'material_metrics': {},
            'r_value_metrics': {},
            'property_metrics': {},
            'correlation_analysis': {},
            'interpolation_analysis': {},
            'knowledge_metrics': {},
            'raw_data': []
        }
        
        # Group results by material
        material_groups = {}
        for result in results:
            if result.material not in material_groups:
                material_groups[result.material] = []
            material_groups[result.material].append(result)
        
        # Group results by R value
        r_value_groups = {}
        for result in results:
            if result.held_out_r not in r_value_groups:
                r_value_groups[result.held_out_r] = []
            r_value_groups[result.held_out_r].append(result)
        
        # Calculate material-specific metrics
        for material, group_results in material_groups.items():
            group_metrics = Metrics.from_group([r.metrics for r in group_results], output_dir=os.path.join(output_dir, material))
            material_props = MaterialProperties.from_text(group_results[0].reference_annotation)
            
            comprehensive_data['material_metrics'][material] = {
                'average_error': float(group_metrics.average_error),
                'property_errors': {
                    field: float(getattr(group_metrics, field))
                    for field in group_metrics.model_fields
                    if field.endswith('_error') and getattr(group_metrics, field) is not None
                },
                'property_errors_std': group_metrics.property_errors_std,
                'confidence_intervals': group_metrics.confidence_interval_95,
                'complexity': float(MaterialAnalysis.calculate_complexity(
                    material_props.atomic_count,
                    material_props.cell_volume
                )) if material_props else None,
                'num_samples': len(group_results),
                'r_value_performance': {
                    r.held_out_r: {
                        'mean_error': float(np.mean([res.metrics.average_error for res in group_results if res.held_out_r == r.held_out_r])),
                        'std_error': float(np.std([res.metrics.average_error for res in group_results if res.held_out_r == r.held_out_r]))
                    }
                    for r in group_results
                }
            }
        
        # Calculate R-value specific metrics
        for r_value, group_results in r_value_groups.items():
            r_value_metrics = Metrics.from_group([r.metrics for r in group_results], output_dir=os.path.join(output_dir, r_value))
            r_progression = RValueAnalysis.analyze_r_progression(group_results)
            interpolation = RValueAnalysis.calculate_interpolation_performance(group_results)
            
            comprehensive_data['r_value_metrics'][r_value] = {
                'average_error': float(r_value_metrics.average_error),
                'property_errors': {
                    field: float(getattr(r_value_metrics, field))
                    for field in r_value_metrics.model_fields
                    if field.endswith('_error') and getattr(r_value_metrics, field) is not None
                },
                'property_errors_std': r_value_metrics.property_errors_std,
                'confidence_intervals': r_value_metrics.confidence_interval_95,
                'progression': r_progression,
                'interpolation': interpolation,
                'num_samples': len(group_results),
                'material_performance': {
                    r.material: {
                        'mean_error': float(np.mean([res.metrics.average_error for res in group_results if res.material == r.material])),
                        'std_error': float(np.std([res.metrics.average_error for res in group_results if res.material == r.material]))
                    }
                    for r in group_results
                }
            }
        
        # Calculate property-specific metrics
        for field_name in results[0].metrics.model_fields:
            if field_name.endswith('_error'):
                property_values = [getattr(r.metrics, field_name) for r in results if getattr(r.metrics, field_name) is not None]
                if property_values:
                    comprehensive_data['property_metrics'][field_name] = {
                        'mean': float(np.mean(property_values)),
                        'std': float(np.std(property_values)),
                        'min': float(np.min(property_values)),
                        'max': float(np.max(property_values)),
                        'median': float(np.median(property_values)),
                        'q1': float(np.percentile(property_values, 25)),
                        'q3': float(np.percentile(property_values, 75)),
                        'num_samples': len(property_values)
                    }
        
        # Calculate correlation analysis
        errors = {}
        for result in results:
            for field_name in result.metrics.model_fields:
                if field_name.endswith('_error') and getattr(result.metrics, field_name) is not None:
                    if field_name not in errors:
                        errors[field_name] = []
                    errors[field_name].append(getattr(result.metrics, field_name))
        
        if errors:
            corr_matrix = np.corrcoef([errors[field] for field in errors.keys()])
            comprehensive_data['correlation_analysis'] = {
                'correlation_matrix': corr_matrix.tolist(),
                'property_names': [field.replace('_error', '') for field in errors.keys()],
                'raw_errors': {field: errors[field] for field in errors.keys()}
            }
        
        # Add knowledge metrics summary
        knowledge_metrics = {
            'prediction_consistency': [],
            'physical_law_compliance': [],
            'format_faithfulness': [],
            'hallucination_score': []
        }
        
        for result in results:
            for metric in knowledge_metrics.keys():
                value = getattr(result.metrics, metric)
                if value is not None:
                    knowledge_metrics[metric].append(value)
        
        comprehensive_data['knowledge_metrics'] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'num_samples': len(values)
            }
            for metric, values in knowledge_metrics.items()
        }
        
        # Save raw data for each result
        for result in results:
            comprehensive_data['raw_data'].append({
                'material': result.material,
                'held_out_r': result.held_out_r,
                'rotation_index': result.rotation_index,
                'metrics': {
                    field: float(getattr(result.metrics, field))
                    for field in result.metrics.model_fields
                    if field.endswith('_error') and getattr(result.metrics, field) is not None
                },
                'average_error': float(result.metrics.average_error),
                'knowledge_metrics': {
                    metric: float(getattr(result.metrics, metric))
                    for metric in knowledge_metrics.keys()
                    if getattr(result.metrics, metric) is not None
                }
            })
        
        # Save to file
        with open(os.path.join(output_dir, 'comprehensive_metrics.json'), 'w') as f:
            json.dump(comprehensive_data, f, indent=2)

    @staticmethod
    def save_model_outputs(results: List[Result], output_dir: str):
        """Save model outputs (generated annotations) in a separate JSON file"""
        outputs_data = {
            'summary': {
                'total_samples': len(results),
                'materials': sorted(set(r.material for r in results)),
                'r_values': sorted(set(r.held_out_r for r in results)),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'outputs': []
        }
        
        # Group results by material and R value
        for material in sorted(set(r.material for r in results)):
            material_data = {
                'material': material,
                'r_values': {}
            }
            
            for r_value in sorted(set(r.held_out_r for r in results if r.material == material)):
                r_value_results = [r for r in results if r.material == material and r.held_out_r == r_value]
                if r_value_results:
                    material_data['r_values'][r_value] = {
                        'rotations': {
                            str(r.rotation_index): {
                                'generated_annotation': r.generated_annotation,
                                'reference_annotation': r.reference_annotation,
                                'metrics': {
                                    field: float(getattr(r.metrics, field))
                                    for field in r.metrics.model_fields
                                    if field.endswith('_error') and getattr(r.metrics, field) is not None
                                },
                                'average_error': float(r.metrics.average_error)
                            }
                            for r in r_value_results
                        }
                    }
            
            outputs_data['outputs'].append(material_data)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'model_outputs_{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump(outputs_data, f, indent=2)

# =============================
# Helper: Build leave-one-radius-out samples
# =============================
def build_leave_one_radius_out_samples(input_dir: str, material: str, r_values: Set[float]) -> List[Sample]:
    """
    Build leave-one-radius-out samples for a specific material.
    
    Args:
        input_dir: Base directory containing material subdirectories
        material: Material name
        r_values: Set of R values for this material
        
    Returns:
        List of Sample objects for leave-one-radius-out validation
    """
    samples = []
    material_dir = os.path.join(input_dir, material)
    
    # For each R value, create a sample where that R value is held out
    for test_r in r_values:
        # Get all other R values for training
        train_r_values = r_values - {test_r}
        
        # Build training samples
        train_samples = []
        for train_r in train_r_values:
            r_dir = os.path.join(material_dir, f"R{train_r}")
            
            # Get image files
            image_dir = os.path.join(r_dir, "images")
            if not os.path.exists(image_dir):
                print(f"Warning: No images directory found in {r_dir}")
                continue
                
            image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"Warning: No image files found in {image_dir}")
                continue
                
            # Get annotation files
            annotation_dir = os.path.join(r_dir, "annotations")
            if not os.path.exists(annotation_dir):
                print(f"Warning: No annotations directory found in {r_dir}")
                continue
                
            annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
            if not annotation_files:
                print(f"Warning: No annotation files found in {annotation_dir}")
                continue
                
            # Get XYZ files
            xyz_dir = os.path.join(r_dir, "xyz")
            if not os.path.exists(xyz_dir):
                print(f"Warning: No xyz directory found in {xyz_dir}")
                continue
                
            xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith('.xyz')]
            if not xyz_files:
                print(f"Warning: No xyz files found in {xyz_dir}")
                continue
            
            # Sort files by rotation number
            image_files.sort(key=lambda x: int(x.split('rot')[1].split('.')[0]))
            annotation_files.sort(key=lambda x: int(x.split('rot')[1].split('_')[0]))
            xyz_files.sort(key=lambda x: int(x.split('rot')[1].split('.')[0]))
            
            # Add all rotations as training samples
            for rot_idx in range(len(image_files)):
                train_samples.append({
                    'material': material,
                    'r': f"R{train_r}",
                    'annotation': os.path.join(annotation_dir, annotation_files[rot_idx]),
                    'image_path': os.path.join(image_dir, image_files[rot_idx]),
                    'xyz_path': os.path.join(xyz_dir, xyz_files[rot_idx])
                })
        
        # Build test samples for each rotation
        test_r_dir = os.path.join(material_dir, f"R{test_r}")
        
        # Get test image files
        test_image_dir = os.path.join(test_r_dir, "images")
        if not os.path.exists(test_image_dir):
            print(f"Warning: No images directory found in {test_r_dir}")
            continue
            
        test_image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not test_image_files:
            print(f"Warning: No image files found in {test_image_dir}")
            continue
            
        # Get test annotation files
        test_annotation_dir = os.path.join(test_r_dir, "annotations")
        if not os.path.exists(test_annotation_dir):
            print(f"Warning: No annotations directory found in {test_annotation_dir}")
            continue
            
        test_annotation_files = [f for f in os.listdir(test_annotation_dir) if f.endswith('.txt')]
        if not test_annotation_files:
            print(f"Warning: No annotation files found in {test_annotation_dir}")
            continue
            
        # Get test XYZ files
        test_xyz_dir = os.path.join(test_r_dir, "xyz")
        if not os.path.exists(test_xyz_dir):
            print(f"Warning: No xyz directory found in {test_xyz_dir}")
            continue
            
        test_xyz_files = [f for f in os.listdir(test_xyz_dir) if f.endswith('.xyz')]
        if not test_xyz_files:
            print(f"Warning: No xyz files found in {test_xyz_dir}")
            continue
        
        # Sort files by rotation number
        test_image_files.sort(key=lambda x: int(x.split('rot')[1].split('.')[0]))
        test_annotation_files.sort(key=lambda x: int(x.split('rot')[1].split('_')[0]))
        test_xyz_files.sort(key=lambda x: int(x.split('rot')[1].split('.')[0]))
        
        # Create samples for each rotation
        for rot_idx in range(len(test_image_files)):
            sample = Sample(
                material=material,
                held_out_r=f"R{test_r}",
                rotation_index=rot_idx,
                held_out_annotation=os.path.join(test_annotation_dir, test_annotation_files[rot_idx]),
                held_out_image=os.path.join(test_image_dir, test_image_files[rot_idx]),
                held_out_xyz=os.path.join(test_xyz_dir, test_xyz_files[rot_idx]),
                context=train_samples
            )
            samples.append(sample)
    
    return samples

# =============================
# Custom pipeline for leave-one-radius-out
# =============================
class LeaveOneRadiusOutPipeline(MultiModelAnnotationPipeline):
    def create_messages(self, held_out_image, held_out_xyz, context, material, held_out_r):
        """
        Build a prompt with context from other R values and the query for the held-out R value.
        Uses all available examples for maximum context, without XYZ files to reduce input size.
        """
        # Sort context by R value for consistent ordering
        sorted_context = sorted(context, key=lambda x: x['r'])
        
        # Build context string using all available examples
        context_str = ""
        context_images = []  # List to store context images
        for c in sorted_context:
            with open(c['annotation'], 'r') as f:
                ann = f.read()
            context_str += f"Material: {c['material']}\n"
            context_str += f"R value: {c['r']}\n"
            context_str += f"Annotation:\n{ann}\n"
            context_str += f"---\n"
            
            # Add context image if available
            if os.path.exists(c['image_path']):
                context_images.append(c['image_path'])
            
        # Read held-out xyz and format coordinates to 2 decimal places
        with open(held_out_xyz, 'r') as f:
            xyz_content = f.read()
            lines = xyz_content.split('\n')
            # Skip the first line (atomic count) and second line (comment)
            coords = lines[2:]  # Process coordinate lines
            formatted_coords = []
            for line in coords:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:  # Ensure we have atom type and coordinates
                        atom = parts[0]
                        x, y, z = map(lambda x: f"{float(x):.2f}", parts[1:4])
                        formatted_coords.append(f"{atom} {x} {y} {z}")
            xyz_content = '\n'.join(formatted_coords)
            
        system_content = (
            "You are a materials science expert specializing in analyzing crystal structures. "
            "You are given context about different R values for the same material. "
            "Your task is to generate a detailed annotation for the provided structure, "
            "using the context from other R values as guidance. "
            "The annotation must strictly follow the format below, with ALL numerical values formatted to exactly 4 decimal places:\n\n"
            "Material: [Material]\n"
            "R value: [R]\n"
            "Atomic Count: [Total atoms]\n"
            "Cell Volume: [Cell volume] Å³\n"
            "Lattice Parameters: a = [a] Å, b = [b] Å, c = [c] Å\n"
            "Average NN Distance: [NN distance] Å\n"
            "Rotation Axis: [Rotation Axis]\n"
            "Density: [Density] g/cm³\n\n"
            "Primitive Cell Information:\n"
            "Primitive Cell Parameters: a = [primitive_a] Å, b = [primitive_b] Å, c = [primitive_c] Å\n"
            "Primitive Cell Angles: α = [primitive_alpha]°, β = [primitive_beta]°, γ = [primitive_gamma]°\n"
            "Space Group: [space_group]\n\n"
            "Summary:\n"
            "This [Material] crystal structure at radius [R] contains [Total atoms] atoms and has a cell volume of [Cell volume] Å³. "
            "The primitive unit cell has parameters a = [primitive_a] Å, b = [primitive_b] Å, c = [primitive_c] Å with angles α = [primitive_alpha]°, β = [primitive_beta]°, γ = [primitive_gamma]°, "
            "indicating a [geometry] geometry in space group [space_group]. "
            "The supercell (R[R]) is characterized by parameters a = [a] Å, b = [b] Å, c = [c] Å. "
            "The average nearest neighbor distance is [NN distance] Å, and the density is estimated to be [Density] g/cm³.\n\n"
            "IMPORTANT: You MUST include the Primitive Cell Information section with ALL primitive cell parameters and angles. "
            "This information is crucial for understanding the fundamental crystal structure. "
            "Only output the annotation as specified above without any additional text. "
            "Remember to format ALL numerical values to exactly 4 decimal places."
        )
        user_text = (
            f"Material: {material}\n"
            f"Target R value: {held_out_r}\n"
            f"\nContext from other R values:\n{context_str}\n"
            "Here is the XYZ structural data for the target material:\n"
            "\n" + xyz_content + "\n\n"
            "Based on the context and the structural data for the target material, generate the annotation following the exact format provided."
        )

        # Only include image for models that support multimodal input
        multimodal_models = {
            "mistralai/mistral-medium-3",
            "x-ai/grok-2-vision-1212",
            "meta-llama/llama-4-maverick",
            "openai/gpt-4.1-mini",
            "google/gemini-2.5-flash-preview-05-20",
            "anthropic/claude-opus-4",
            "anthropic/claude-sonnet-4",
        }

        if self.model_name in multimodal_models:
            # Create content array with text and context images
            content = [{"type": "text", "text": user_text}]
            
            # Add context images
            selected_images = random.sample(context_images, min(2, len(context_images))) if context_images else []
            for img_path in selected_images:
                try:
                    image_b64 = self._encode_image(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "low"}
                    })
                except Exception as e:
                    print(f"Error encoding image {img_path}: {e}")
                    continue
            
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        return messages

    def compute_metrics(self, reference: str, generated: str) -> Metrics:
        """
        Compute metrics between reference and generated annotations using Pydantic models.
        """
        try:
            ref_props = MaterialProperties.from_text(reference)
            gen_props = MaterialProperties.from_text(generated)
            
            if ref_props is None or gen_props is None:
                return Metrics(average_error=float('inf'))
            
            return Metrics.from_properties(ref_props, gen_props)
            
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            return Metrics(average_error=float('inf'))

    def generate_annotation(self, held_out_image, held_out_xyz, context, material, held_out_r, max_retries=3):
        messages = self.create_messages(held_out_image, held_out_xyz, context, material, held_out_r)
        for attempt in range(max_retries):
            try:
                if self.use_openrouter:
                    payload = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": 6500,
                        "temperature": 0.2
                    }
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers=self.headers,
                        json=payload
                    )
                    if response.status_code != 200:
                        print(f"OpenRouter Error Response: {response.text}")
                        raise Exception(f"OpenRouter API Error: {response.json()}")
                    response_json = response.json()
                    if 'error' in response_json:
                        raise Exception(f"OpenRouter API Error: {response_json['error']}")
                    return response_json['choices'][0]['message']['content']
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.2
                    )
                    return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)
                continue

# =============================
# Main execution
# =============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Leave-one-radius-out validation for LLMs.")
    parser.add_argument('--dataset_dir', type=str, default="multimodal_dataset", help='Path to the dataset base directory (default: multimodal_dataset)')
    args = parser.parse_args()

    dataset_base_dir = args.dataset_dir

    # Set your API keys here
    api_keys = {
        'openrouter': os.getenv('OPENROUTER_API_KEY'),
    }

    models_to_test = [
        {'name': "deepseek/deepseek-chat", 'use_openrouter': True},
        {'name': "x-ai/grok-2-vision-1212", 'use_openrouter': True},
        {'name': "x-ai/grok-2-1212", 'use_openrouter': True},
        {'name': "meta-llama/llama-4-maverick", 'use_openrouter': True},
        {'name': "mistralai/mistral-medium-3", 'use_openrouter': True},
        {'name': "openai/gpt-4.1-mini", 'use_openrouter': True},
        {'name': "google/gemini-2.5-flash-preview-05-20", 'use_openrouter': True},
        {'name': "anthropic/claude-opus-4", 'use_openrouter': True},
        {'name': "anthropic/claude-sonnet-4", 'use_openrouter': True}
    ]

    # Get all materials and their R values
    material_r_values = {}
    for material_dir in os.listdir(dataset_base_dir):
        material_path = os.path.join(dataset_base_dir, material_dir)
        if os.path.isdir(material_path) and not material_dir.startswith('.'):
            r_values = set()
            for r_dir in os.listdir(material_path):
                if r_dir.startswith('R') and os.path.isdir(os.path.join(material_path, r_dir)):
                    try:
                        r_value = int(r_dir[1:])  # Extract number after 'R' as integer
                        r_values.add(r_value)
                    except ValueError:
                        continue
            if r_values:
                material_r_values[material_dir] = r_values

    # Build leave-one-radius-out samples for each material
    all_samples = []
    for material, r_values in material_r_values.items():
        material_samples = build_leave_one_radius_out_samples(dataset_base_dir, material, r_values)
        all_samples.extend(material_samples)
    
    def process_samples_loro(pipeline, samples, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        results = []
        model_name = pipeline.model_name
        
        # Group samples by material and held_out_r
        grouped_samples = {}
        for sample in samples:
            key = (sample.material, sample.held_out_r)
            if key not in grouped_samples:
                grouped_samples[key] = []
            grouped_samples[key].append(sample)
        
        # Process each group
        for (material, held_out_r), group_samples in grouped_samples.items():
            print(f"\nProcessing {model_name} for {material} with {held_out_r} as held-out")
            
            # Sort samples by rotation index to ensure we process rot0-rot4 in order
            group_samples.sort(key=lambda x: x.rotation_index)
            
            # Select only the first 5 rotations (0-4)
            group_samples = [s for s in group_samples if s.rotation_index < 5]
            
            if not group_samples:
                print(f"Warning: No samples found for {material} {held_out_r}")
                continue
                
            group_results = []
            start_time = time.time()  # Start timing for this held-out group
            
            for sample in tqdm(group_samples, desc=f"Processing {model_name} - {material} - {held_out_r} (rotations 0-4)", position=0, leave=True):
                try:
                    sample_start_time = time.time()  # Start timing for this sample
                    generated_annotation = pipeline.generate_annotation(
                        sample.held_out_image,
                        sample.held_out_xyz,
                        sample.context,
                        sample.material,
                        sample.held_out_r
                    )
                    with open(sample.held_out_annotation, 'r', encoding='utf-8') as f:
                        reference_annotation = f.read()
                    metrics = pipeline.compute_metrics(reference_annotation, generated_annotation)
                    sample_end_time = time.time()  # End timing for this sample
                    
                    result = Result(
                        material=sample.material,
                        held_out_r=sample.held_out_r,
                        rotation_index=sample.rotation_index,
                        reference_annotation=reference_annotation,
                        generated_annotation=generated_annotation,
                        metrics=metrics
                    )
                    # Add timing information to the result
                    result.processing_time = sample_end_time - sample_start_time
                    group_results.append(result)
                except Exception as e:
                    print(f"Error processing sample {sample} for model {model_name}: {e}")
            
            end_time = time.time()  # End timing for this held-out group
            total_time = end_time - start_time
            
            # Calculate group-level metrics
            if group_results:
                group_metrics = Metrics.from_group([r.metrics for r in group_results], output_dir=os.path.join(output_dir, material, held_out_r))
                
                # Calculate material-specific metrics
                material_props = MaterialProperties.from_text(reference_annotation)
                if material_props:
                    complexity = MaterialAnalysis.calculate_complexity(
                        material_props.atomic_count,
                        material_props.cell_volume
                    )
                
                # Calculate R-value progression
                r_progression = RValueAnalysis.analyze_r_progression(group_results)
                
                # Create visualizations
                group_dir = os.path.join(output_dir, material, held_out_r)
                os.makedirs(group_dir, exist_ok=True)
                
                Visualization.collect_error_distribution(group_results, group_dir)
                Visualization.collect_correlation_data(group_results, group_dir)
                
                # Save detailed results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name_safe = pipeline.model_name.replace('/', '_')
                output_file = os.path.join(group_dir, f"loro_annotations_{model_name_safe}_{timestamp}.json")
                with open(output_file, 'w') as f:
                    json.dump({
                        'results': [r.model_dump() for r in group_results],
                        'group_metrics': group_metrics.model_dump(),
                        'summary': {
                            'material': material,
                            'held_out_r': held_out_r,
                            'num_samples': len(group_results),
                            'average_error': group_metrics.average_error,
                            'property_errors_std': group_metrics.property_errors_std,
                            'confidence_intervals': group_metrics.confidence_interval_95,
                            'error_correlations': group_metrics.error_correlations,
                            'material_complexity': complexity if material_props else None,
                            'r_value_progression': r_progression,
                            'timing': {
                                'total_processing_time': total_time,
                                'average_time_per_sample': total_time / len(group_results),
                                'individual_sample_times': {
                                    f'rotation_{r.rotation_index}': r.processing_time
                                    for r in group_results
                                }
                            }
                        }
                    }, f, indent=2)
            
            results.extend(group_results)
        
        # Create overall visualizations
        Visualization.collect_material_performance(results, output_dir)
        
        # After processing all samples, save comprehensive metrics and model outputs
        if results:
            Visualization.save_comprehensive_metrics(results, output_dir)
            Visualization.save_model_outputs(results, output_dir)
        
        return results

    def process_model_loro(model_config, samples, api_keys):
        try:
            print(f"Processing model: {model_config['name']}")
            pipeline = LeaveOneRadiusOutPipeline(
                api_keys=api_keys,
                model_name=model_config['name'],
                use_openrouter=model_config['use_openrouter']
            )
            model_output_dir = os.path.join("loro_annotation_results", model_config['name'].replace('/', '_'))
            results = process_samples_loro(pipeline, samples, model_output_dir)
            print(f"Completed processing model: {model_config['name']}")
            return (model_config['name'], results)
        except Exception as e:
            print(f"Error processing model {model_config['name']}: {e}")
            return (model_config['name'], None)

    # Process each model in parallel
    with ThreadPoolExecutor(max_workers=len(models_to_test)) as executor:
        future_to_model = {
            executor.submit(process_model_loro, model_config, all_samples, api_keys): model_config
            for model_config in models_to_test
        }
        for future in as_completed(future_to_model):
            model_config = future_to_model[future]
            try:
                model_name, results = future.result()
                print(f"Model {model_name} finished with {len(results) if results is not None else 'no'} results.")
            except Exception as e:
                print(f"Exception raised for model {model_config['name']}: {e}")

def validate_leave_one_radius_out(
    model_name: str,
    input_dir: str,
    output_dir: str,
    use_openrouter: bool = False,
    openrouter_api_key: Optional[str] = None
) -> List[Result]:
    """
    Perform leave-one-radius-out validation.
    
    Args:
        model_name: Name of the model to use
        input_dir: Directory containing the input files
        output_dir: Directory to save the results
        use_openrouter: Whether to use OpenRouter API
        openrouter_api_key: OpenRouter API key if using OpenRouter
        
    Returns:
        List of validation results
    """
    pipeline = LeaveOneRadiusOutPipeline(
        model_name=model_name,
        use_openrouter=use_openrouter,
        openrouter_api_key=openrouter_api_key
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all materials and their R values
    material_r_values = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.cif'):
            material = filename.split('_')[0]
            r_value = float(filename.split('_')[1].replace('r', ''))
            if material not in material_r_values:
                material_r_values[material] = set()
            material_r_values[material].add(r_value)
    
    results = []
    for material, r_values in material_r_values.items():
        print(f"\nValidating material: {material}")
        
        # Build samples for this material
        samples = build_leave_one_radius_out_samples(input_dir, material, r_values)
        
        # Process each sample
        for sample in samples:
            try:
                # Generate annotation
                generated = pipeline.generate_annotation(
                    sample.held_out_image,
                    sample.held_out_xyz,
                    sample.context,
                    sample.material,
                    sample.held_out_r
                )
                
                # Read reference annotation
                with open(sample.reference_annotation, 'r') as f:
                    reference = f.read()
                
                # Compute metrics
                metrics = pipeline.compute_metrics(reference, generated)
                
                # Save generated annotation
                output_file = os.path.join(
                    output_dir,
                    f"{material}_r{sample.held_out_r}_generated.txt"
                )
                with open(output_file, 'w') as f:
                    f.write(generated)
                
                # Create result
                result = Result(
                    material=sample.material,
                    held_out_r=sample.held_out_r,
                    reference_annotation=sample.reference_annotation,
                    generated_annotation=output_file,
                    metrics=metrics
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {sample.material} at R={sample.held_out_r}: {str(e)}")
                continue
    
    return results

def build_leave_one_radius_out_samples(input_dir: str, material: str, r_values: Set[float]) -> List[Sample]:
    """
    Build samples for leave-one-radius-out validation.
    
    Args:
        input_dir: Directory containing the input files
        material: Material to validate
        r_values: Set of R values for this material
        
    Returns:
        List of samples for validation
    """
    samples = []
    
    for held_out_r in r_values:
        # Get corresponding files
        held_out_file = f"{material}_r{held_out_r}.cif"
        held_out_image = os.path.join(input_dir, held_out_file.replace('.cif', '.png'))
        held_out_xyz = os.path.join(input_dir, held_out_file.replace('.cif', '.xyz'))
        reference_annotation = os.path.join(input_dir, held_out_file.replace('.cif', '.txt'))
        
        # Check if all required files exist
        if not all(os.path.exists(f) for f in [held_out_image, held_out_xyz, reference_annotation]):
            print(f"Skipping {held_out_file} - missing required files")
            continue
        
        # Build context from other R values
        context = []
        for r in r_values:
            if r != held_out_r:
                annotation = os.path.join(input_dir, f"{material}_r{r}.txt")
                if os.path.exists(annotation):
                    context.append({
                        'material': material,
                        'r': r,
                        'annotation': annotation
                    })
        
        # Create sample
        sample = Sample(
            material=material,
            held_out_r=str(held_out_r),
            rotation_index=0,
            held_out_annotation=reference_annotation,
            held_out_image=held_out_image,
            held_out_xyz=held_out_xyz,
            context=context
        )
        samples.append(sample)
    
    return samples 