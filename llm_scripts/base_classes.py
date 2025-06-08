"""
Base classes for multimodal foundation model interactions in crystallographic reasoning.

This module provides core classes and utilities for interacting with multimodal foundation models
to analyze crystal structures and generate detailed annotations. It includes comprehensive
error handling, retry mechanisms, and metric computation for evaluation.
"""

import base64
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Union, Any, TypeVar
from pathlib import Path
import os 
import json
import numpy as np
import requests
from PIL import Image
from rouge_score import rouge_scorer
from rouge_score.scoring import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI
from pydantic import BaseModel, Field, validator

class Sample(BaseModel):
    material: str
    held_out_r: str
    rotation_index: int
    held_out_annotation: str
    held_out_image: str
    held_out_xyz: str
    context: List[Dict[str, str]]

# Type variables for generic type hints
T = TypeVar('T')
ModelT = TypeVar('ModelT', bound=BaseModel)

class ModelAPIError(Exception):
    """Custom exception for model API-related errors."""
    pass

class ImageProcessingError(Exception):
    """Custom exception for image processing-related errors."""
    pass

class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

@dataclass
class AnnotationMetrics:
    """
    Data class for storing annotation evaluation metrics.
    
    Attributes:
        bleu: BLEU score for text similarity
        rouge1: ROUGE-1 F1 score
        rouge2: ROUGE-2 F1 score
        rougeL: ROUGE-L F1 score
        numerical: Dictionary of numerical property comparisons
        reference_values: Dictionary of reference property values
        generated_values: Dictionary of generated property values
    """
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float
    numerical: Dict[str, Any]
    reference_values: Dict[str, Any]
    generated_values: Dict[str, Any]

class MaterialProperties(BaseModel):
    """
    Data model for material properties extracted from crystal structure annotations.
    
    This model represents both supercell and primitive cell properties of a crystal structure,
    with validation to ensure physical consistency of the values.
    """
    atomic_count: int = Field(..., description="Number of atoms in the structure")
    cell_volume: float = Field(..., description="Cell volume in Å³")
    lattice_a: float = Field(..., description="Lattice parameter a in Å")
    lattice_b: float = Field(..., description="Lattice parameter b in Å")
    lattice_c: float = Field(..., description="Lattice parameter c in Å")
    nn_distance: float = Field(..., description="Average nearest neighbor distance in Å")
    density: float = Field(..., description="Density in g/cm³")
    
    # Primitive cell properties
    primitive_a: Optional[float] = Field(None, description="Primitive cell parameter a in Å")
    primitive_b: Optional[float] = Field(None, description="Primitive cell parameter b in Å")
    primitive_c: Optional[float] = Field(None, description="Primitive cell parameter c in Å")
    primitive_alpha: Optional[float] = Field(None, description="Primitive cell angle alpha in degrees")
    primitive_beta: Optional[float] = Field(None, description="Primitive cell angle beta in degrees")
    primitive_gamma: Optional[float] = Field(None, description="Primitive cell angle gamma in degrees")
    space_group: Optional[str] = Field(None, description="Space group of the primitive cell")

    @validator('atomic_count')
    def validate_atomic_count(cls, v: int) -> int:
        """Validate that atomic count is positive."""
        if v <= 0:
            raise ValidationError("Atomic count must be positive")
        return v

    @validator('cell_volume', 'lattice_a', 'lattice_b', 'lattice_c', 'nn_distance', 'density')
    def validate_positive_values(cls, v: float) -> float:
        """Validate that physical quantities are positive."""
        if v <= 0:
            raise ValidationError("Value must be positive")
        return v

    @validator('primitive_alpha', 'primitive_beta', 'primitive_gamma')
    def validate_angles(cls, v: Optional[float]) -> Optional[float]:
        """Validate that angles are between 0 and 180 degrees."""
        if v is not None and (v <= 0 or v >= 180):
            raise ValidationError("Angle must be between 0 and 180 degrees")
        return v

    @classmethod
    def from_text(cls, text: str) -> Optional['MaterialProperties']:
        """
        Parse material properties from annotation text.
        
        Args:
            text: Annotation text to parse
            
        Returns:
            MaterialProperties object if parsing successful, None otherwise
            
        Raises:
            ValidationError: If required properties are missing or invalid
        """
        try:
            # Define regex patterns for property extraction
            patterns = {
                # Supercell properties
                'atomic_count': (r"Atomic Count:\s*(\d+)", int),
                'cell_volume': (r"Cell Volume:\s*([\d.]+)\s*Å³", float),
                'lattice_a': (r"Lattice Parameters:\s*a\s*=\s*([\d.]+)\s*Å", float),
                'lattice_b': (r"Lattice Parameters:.*b\s*=\s*([\d.]+)\s*Å", float),
                'lattice_c': (r"Lattice Parameters:.*c\s*=\s*([\d.]+)\s*Å", float),
                'nn_distance': (r"Average NN Distance:\s*([\d.]+)\s*Å", float),
                'density': (r"Density:\s*([\d.]+)\s*g/cm³", float),
                
                # Primitive cell properties (optional)
                'primitive_a': (r"Primitive Cell Parameters:\s*a\s*=\s*([\d.]+)\s*Å", float),
                'primitive_b': (r"Primitive Cell Parameters:.*b\s*=\s*([\d.]+)\s*Å", float),
                'primitive_c': (r"Primitive Cell Parameters:.*c\s*=\s*([\d.]+)\s*Å", float),
                'primitive_alpha': (r"Primitive Cell Angles:\s*α\s*=\s*([\d.]+)°", float),
                'primitive_beta': (r"Primitive Cell Angles:.*β\s*=\s*([\d.]+)°", float),
                'primitive_gamma': (r"Primitive Cell Angles:.*γ\s*=\s*([\d.]+)°", float),
                'space_group': (r"Space Group:\s*([^'\n]+)", str)
            }
            
            values = {}
            missing_required = False
            
            # First pass: check required fields
            for key, (pattern, converter) in patterns.items():
                if not key.startswith('primitive_') and key != 'space_group':  # Required fields
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if not match:
                        missing_required = True
                        continue
                    try:
                        values[key] = converter(match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ValidationError(f"Error parsing {key}: {str(e)}")
            
            if missing_required:
                raise ValidationError("Missing required properties")
            
            # Second pass: get optional fields
            for key, (pattern, converter) in patterns.items():
                if key.startswith('primitive_') or key == 'space_group':  # Optional fields
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        try:
                            values[key] = converter(match.group(1))
                        except (ValueError, IndexError):
                            values[key] = None
            
            # Validate numeric values
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    if not np.isfinite(value) or value <= 0:
                        if not key.startswith('primitive_') and key != 'space_group':
                            raise ValidationError(f"Invalid value for {key}")
                        values[key] = None
            
            return cls(**values)
        except Exception as e:
            raise ValidationError(f"Error parsing text: {str(e)}")

    def has_primitive_cell(self) -> bool:
        """
        Check if the material has complete primitive cell information.
        
        Returns:
            bool: True if all primitive cell parameters are available
        """
        return all(
            getattr(self, field) is not None
            for field in ['primitive_a', 'primitive_b', 'primitive_c', 
                         'primitive_alpha', 'primitive_beta', 'primitive_gamma']
        )

    def get_primitive_cell_volume(self) -> Optional[float]:
        """
        Calculate primitive cell volume if all parameters are available.
        
        Returns:
            Optional[float]: Primitive cell volume in Å³ if calculable, None otherwise
        """
        if not self.has_primitive_cell():
            return None
            
        try:
            # Convert angles to radians
            alpha = np.radians(self.primitive_alpha)
            beta = np.radians(self.primitive_beta)
            gamma = np.radians(self.primitive_gamma)
            
            # Calculate volume using the formula for triclinic cell
            volume = (self.primitive_a * self.primitive_b * self.primitive_c * 
                     np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 
                            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)))
            
            return float(volume) if np.isfinite(volume) else None
        except Exception:
            return None

class MultiModelAnnotationPipeline:
    """
    A robust pipeline for generating and evaluating crystal structure annotations using multimodal foundation models.
    
    This class handles the interaction with various multimodal foundation models (OpenAI, OpenRouter)
    to analyze crystal structures and generate detailed annotations. It includes comprehensive
    error handling, retry mechanisms, and metric computation for evaluation.
    """

    def __init__(
        self,
        api_keys: Dict[str, str],
        model_name: str = "gpt-4-vision",
        use_openrouter: bool = False,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ) -> None:
        """
        Initialize the annotation evaluation pipeline.

        Args:
            api_keys: Dictionary containing API keys for different services
            model_name: Name of the model to use
            use_openrouter: Whether to use OpenRouter API instead of OpenAI
            site_url: Optional site URL for OpenRouter
            site_name: Optional site name for OpenRouter
            
        Raises:
            ValueError: If required API keys are missing
        """
        self.model_name = model_name
        self.use_openrouter = use_openrouter
        self.api_keys = api_keys

        if use_openrouter:
            if 'openrouter' not in api_keys:
                raise ValueError("OpenRouter API key is required when use_openrouter is True")
            self.headers = {
                "Authorization": f"Bearer {api_keys['openrouter']}",
                "Content-Type": "application/json"
            }
        else:
            if 'openai' not in api_keys:
                raise ValueError("OpenAI API key is required when use_openrouter is False")
            self.client = OpenAI(api_key=api_keys['openai'])

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Convert image to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string of the image

        Raises:
            ImageProcessingError: If image processing fails
            FileNotFoundError: If image file does not exist
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            raise ImageProcessingError(f"Error encoding image {image_path}: {e}")

    def _format_xyz_data(self, xyz_path: Union[str, Path]) -> str:
        """
        Read and format XYZ file content.

        Args:
            xyz_path: Path to the XYZ file

        Returns:
            Formatted XYZ content

        Raises:
            FileNotFoundError: If XYZ file cannot be found
            ValueError: If XYZ file format is invalid
        """
        try:
            xyz_path = Path(xyz_path)
            if not xyz_path.exists():
                raise FileNotFoundError(f"XYZ file not found: {xyz_path}")
                
            with open(xyz_path, 'r') as f:
                xyz_content = f.read()
                # Skip first two lines (header)
                xyz_content = xyz_content.split('\n', 2)[2]
            return xyz_content
        except Exception as e:
            raise ValueError(f"Error reading XYZ file {xyz_path}: {e}")

    def create_messages(
        self,
        image_path: Union[str, Path],
        xyz_path: Union[str, Path],
        material: str,
        structure: str,
        is_original: bool
    ) -> List[Dict[str, Any]]:
        """
        Create messages for the model API to generate an annotation.

        Args:
            image_path: Path to the structure image
            xyz_path: Path to the XYZ file
            material: Material name
            structure: Structure prototype
            is_original: Whether this is the original structure

        Returns:
            List of message dictionaries for the API
            
        Raises:
            ImageProcessingError: If image processing fails
            ValueError: If XYZ file processing fails
        """
        image_b64 = self._encode_image(image_path)
        xyz_content = self._format_xyz_data(xyz_path)

        system_content = (
            "You are a materials science expert specializing in analyzing crystal structures. "
            "Your task is to generate a detailed annotation for the provided structure based on both visual and atomic coordinate data. "
            "The annotation must strictly follow the format below:\n\n"
            "Material: [Material]\n"
            "Structure Prototype: [Structure]\n"
            "Atomic Count: [Total atoms]\n"
            "Cell Volume: [Cell volume] Å³\n"
            "Lattice Parameters: a = [a] Å, b = [b] Å, c = [c] Å\n"
            "Average NN Distance: [NN distance] Å\n"
            "Rotation Axis: [Rotation Axis]\n"
            "Density: [Density] g/cm³\n\n"
            "Summary:\n"
            "This [Material] crystal structure with prototype [Structure] contains [Total atoms] atoms and has a cell volume of [Cell volume] Å³. "
            "Its lattice is characterized by parameters a = [a] Å, b = [b] Å, and c = [c] Å, suggesting an orthogonal geometry. "
            "The average nearest neighbor distance is [NN distance] Å, and the density is estimated to be [Density] g/cm³.\n\n"
            "Only output the annotation as specified above without any additional text."
        )

        user_text = (
            f"Analyze this crystal structure.\n"
            f"Material: {material}\n"
            f"Structure Prototype: {structure}\n"
            f"This is {'the original structure (no rotation)' if is_original else 'a rotated structure'}.\n\n"
            "Here is the XYZ structural data:\n"
            "```\n"
            f"{xyz_content}\n"
            "```\n\n"
            "Based on the structural data and the image, generate the annotation following the exact format provided."
        )

        return [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "low"}}
                ]
            }
        ]

    def generate_annotation(
        self,
        image_path: Union[str, Path],
        xyz_path: Union[str, Path],
        material: str,
        structure: str,
        is_original: bool,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """
        Generate an annotation using the selected model.

        Args:
            image_path: Path to the structure image
            xyz_path: Path to the XYZ file
            material: Material name
            structure: Structure prototype
            is_original: Whether this is the original structure
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds

        Returns:
            Generated annotation text

        Raises:
            ModelAPIError: If model API calls fail after all retries
            ImageProcessingError: If image processing fails
            ValueError: If XYZ file processing fails
        """
        messages = self.create_messages(image_path, xyz_path, material, structure, is_original)

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
                        json=payload,
                        timeout=30  # Add timeout
                    )
                    response.raise_for_status()  # Raise exception for non-200 status codes
                    response_json = response.json()
                    if 'error' in response_json:
                        raise ModelAPIError(f"OpenRouter API Error: {response_json['error']}")
                    return response_json['choices'][0]['message']['content']
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.2
                    )
                    return response.choices[0].message.content
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise ModelAPIError(f"Failed to generate annotation after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ModelAPIError(f"Failed to generate annotation after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (2 ** attempt))
                continue

    def extract_properties(self, text: str) -> Dict[str, Union[int, float, str, None]]:
        """
        Extract numerical properties from an annotation text.

        Args:
            text: Annotation text to parse

        Returns:
            Dictionary of extracted properties
            
        Raises:
            ValueError: If property extraction fails
        """
        props = {}
        try:
            # Define regex patterns for property extraction
            patterns = {
                'atomic_count': (r'Atomic Count:\s*(\d+)', int),
                'cell_volume': (r'Cell Volume:\s*([\d.]+)\s*Å³', float),
                'lattice_a': (r'Lattice Parameters:\s*a\s*=\s*([\d.]+)\s*Å', float),
                'lattice_b': (r'b\s*=\s*([\d.]+)\s*Å', float),
                'lattice_c': (r'c\s*=\s*([\d.]+)\s*Å', float),
                'nn_distance': (r'Average NN Distance:\s*([\d.]+)\s*Å', float),
                'density': (r'Density:\s*([\d.]+)\s*g/cm³', float),
                'material': (r'Material:\s*(\S+)', str),
                'structure': (r'Structure Prototype:\s*(\S+)', str)
            }

            for key, (pattern, type_cast) in patterns.items():
                match = re.search(pattern, text)
                if match:
                    try:
                        props[key] = type_cast(match.group(1))
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Error converting {key} value: {e}")

        except Exception as e:
            raise ValueError(f"Error extracting properties: {e}")
        return props

    def compute_metrics(self, reference: str, candidate: str) -> AnnotationMetrics:
        """
        Compute text similarity and numerical differences between reference and candidate annotations.

        Args:
            reference: Reference annotation text
            candidate: Candidate annotation text

        Returns:
            AnnotationMetrics object containing computed metrics

        Raises:
            ValueError: If metric computation fails
        """
        try:
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothing)
            rouge_scores = self.rouge_scorer.score(reference, candidate)

            ref_props = self.extract_properties(reference)
            cand_props = self.extract_properties(candidate)

            numerical_metrics = {}
            for key in ['atomic_count', 'cell_volume', 'lattice_a', 'lattice_b', 'lattice_c', 'nn_distance', 'density']:
                if key in ref_props and key in cand_props and ref_props[key] is not None and cand_props[key] is not None:
                    diff = abs(ref_props[key] - cand_props[key])
                    perc = diff / ref_props[key] * 100 if ref_props[key] != 0 else None
                    numerical_metrics[key] = {
                        'reference': ref_props[key],
                        'candidate': cand_props[key],
                        'absolute_error': diff,
                        'percent_error': perc
                    }
                else:
                    numerical_metrics[key] = None

            material_match = (ref_props.get('material', '').lower() == cand_props.get('material', '').lower())
            structure_match = (ref_props.get('structure', '').lower() == cand_props.get('structure', '').lower())
            numerical_metrics['material_match'] = material_match
            numerical_metrics['structure_match'] = structure_match

            return AnnotationMetrics(
                bleu=bleu_score,
                rouge1=rouge_scores['rouge1'].fmeasure,
                rouge2=rouge_scores['rouge2'].fmeasure,
                rougeL=rouge_scores['rougeL'].fmeasure,
                numerical=numerical_metrics,
                reference_values=ref_props,
                generated_values=cand_props
            )
        except Exception as e:
            raise ValueError(f"Error computing metrics: {e}")

class Metrics(BaseModel):
    atomic_count_error: Optional[float] = Field(None, description="Relative error in atomic count")
    cell_volume_error: Optional[float] = Field(None, description="Relative error in cell volume")
    lattice_a_error: Optional[float] = Field(None, description="Relative error in lattice parameter a")
    lattice_b_error: Optional[float] = Field(None, description="Relative error in lattice parameter b")
    lattice_c_error: Optional[float] = Field(None, description="Relative error in lattice parameter c")
    nn_distance_error: Optional[float] = Field(None, description="Relative error in nearest neighbor distance")
    density_error: Optional[float] = Field(None, description="Relative error in density")
    
    # Primitive cell error metrics
    primitive_a_error: Optional[float] = Field(None, description="Relative error in primitive cell parameter a")
    primitive_b_error: Optional[float] = Field(None, description="Relative error in primitive cell parameter b")
    primitive_c_error: Optional[float] = Field(None, description="Relative error in primitive cell parameter c")
    primitive_alpha_error: Optional[float] = Field(None, description="Absolute error in primitive cell angle alpha")
    primitive_beta_error: Optional[float] = Field(None, description="Absolute error in primitive cell angle beta")
    primitive_gamma_error: Optional[float] = Field(None, description="Absolute error in primitive cell angle gamma")
    space_group_match: Optional[bool] = Field(None, description="Whether space group matches exactly")
    
    average_error: float = Field(..., description="Average of all valid errors")
    
    property_errors_std: Optional[Dict[str, float]] = Field(None, description="Standard deviation of errors for each property")
    confidence_interval_95: Optional[Dict[str, tuple]] = Field(None, description="95% confidence interval for each property")
    error_correlations: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlations between property errors")
    absolute_errors: Optional[Dict[str, float]] = Field(None, description="Absolute errors for each property")
    
    prediction_consistency: Optional[float] = Field(None, description="Consistency of predictions across rotations")
    physical_law_compliance: Optional[float] = Field(None, description="Compliance with physical laws")
    format_faithfulness: Optional[float] = Field(None, description="Faithfulness to reference format")
    hallucination_score: Optional[float] = Field(None, description="Score indicating potential hallucinations")
    
    @classmethod
    def from_properties(cls, reference: MaterialProperties, generated: MaterialProperties) -> 'Metrics':
        errors = {}
        absolute_errors = {}
        
        # Calculate relative and absolute errors for each property
        for field_name in MaterialProperties.model_fields:
            ref_value = getattr(reference, field_name)
            gen_value = getattr(generated, field_name)
            
            if ref_value is not None and gen_value is not None:
                try:
                    if field_name.startswith('primitive_alpha') or field_name.startswith('primitive_beta') or field_name.startswith('primitive_gamma'):
                        # For angles, use absolute error
                        abs_error = abs(gen_value - ref_value)
                        errors[f"{field_name}_error"] = abs_error
                        absolute_errors[field_name] = abs_error
                    elif field_name == 'space_group':
                        # For space group, use exact match
                        errors['space_group_match'] = gen_value == ref_value
                    elif ref_value != 0:
                        # For other properties, use relative error
                        rel_error = abs(gen_value - ref_value) / abs(ref_value)
                        abs_error = abs(gen_value - ref_value)
                        errors[f"{field_name}_error"] = rel_error
                        absolute_errors[field_name] = abs_error
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
        
        # Calculate average error (excluding space group match)
        error_values = [v for k, v in errors.items() if k != 'space_group_match']
        if error_values:
            average_error = sum(error_values) / len(error_values)
        else:
            average_error = float('inf')
        
        # Calculate physical law compliance
        physical_law_compliance = cls._calculate_physical_law_compliance(reference, generated)
        
        # Calculate format faithfulness
        format_faithfulness = cls._calculate_format_faithfulness(reference, generated)
        
        # Calculate hallucination score
        hallucination_score = cls._calculate_hallucination_score(reference, generated)
        
        return cls(
            **errors,
            average_error=average_error,
            absolute_errors=absolute_errors,
            physical_law_compliance=physical_law_compliance,
            format_faithfulness=format_faithfulness,
            hallucination_score=hallucination_score
        )
    
    @staticmethod
    def _calculate_physical_law_compliance(reference: MaterialProperties, generated: MaterialProperties) -> float:
        """
        Calculate physical law compliance score based on relative deviations.
        
        Scoring rules:
        - ≤10% deviation = 1.0
        - >10% and ≤25% = 0.5
        - >25% = 0.0
        
        The final score is averaged over all valid checks.
        
        Args:
            reference: Reference material properties
            generated: Generated material properties
            
        Returns:
            float: Compliance score between 0 and 1
        """
        if reference is None or generated is None:
            return 0.0
            
        scores = []
        
        # Check density relationship
        try:
            if reference.density > 0 and generated.density > 0:
                rel = abs(generated.density - reference.density) / reference.density
                if rel <= 0.10:
                    scores.append(1.0)
                elif rel <= 0.25:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
        except Exception:
            scores.append(0.0)
        
        # Check lattice parameter ratios
        try:
            for p1, p2 in [('lattice_b', 'lattice_a'), ('lattice_c', 'lattice_a')]:
                ref_p1 = getattr(reference, p1)
                ref_p2 = getattr(reference, p2)
                gen_p1 = getattr(generated, p1)
                gen_p2 = getattr(generated, p2)
                
                if ref_p2 > 0 and gen_p2 > 0:
                    ref_ratio = ref_p1 / ref_p2
                    gen_ratio = gen_p1 / gen_p2
                    rel = abs(gen_ratio - ref_ratio) / ref_ratio if ref_ratio != 0 else 1.0
                    
                    if rel <= 0.10:
                        scores.append(1.0)
                    elif rel <= 0.25:
                        scores.append(0.5)
                    else:
                        scores.append(0.0)
        except Exception:
            scores.append(0.0)
        
        # Check primitive cell ratios
        try:
            if all(getattr(reference, f'primitive_{p}') is not None for p in ['a','b','c']) and \
               all(getattr(generated, f'primitive_{p}') is not None for p in ['a','b','c']):
                for p1, p2 in [('primitive_b', 'primitive_a'), ('primitive_c', 'primitive_a')]:
                    ref_p1 = getattr(reference, p1)
                    ref_p2 = getattr(reference, p2)
                    gen_p1 = getattr(generated, p1)
                    gen_p2 = getattr(generated, p2)
                    
                    if ref_p2 > 0 and gen_p2 > 0:
                        ref_ratio = ref_p1 / ref_p2
                        gen_ratio = gen_p1 / gen_p2
                        rel = abs(gen_ratio - ref_ratio) / ref_ratio if ref_ratio != 0 else 1.0
                        
                        if rel <= 0.10:
                            scores.append(1.0)
                        elif rel <= 0.25:
                            scores.append(0.5)
                        else:
                            scores.append(0.0)
        except Exception:
            scores.append(0.0)
        
        return float(np.mean(scores)) if scores else 0.0
    
    @staticmethod
    def _calculate_format_faithfulness(reference: MaterialProperties, generated: MaterialProperties) -> float:
        """Calculate faithfulness to reference format"""
        faithfulness_scores = []
        
        # Check if all required fields are present
        required_fields = ['atomic_count', 'cell_volume', 'lattice_a', 'lattice_b', 'lattice_c', 'nn_distance', 'density']
        ref_fields = all(getattr(reference, field) is not None for field in required_fields)
        gen_fields = all(getattr(generated, field) is not None for field in required_fields)
        faithfulness_scores.append(1.0 if ref_fields == gen_fields else 0.0)
        
        # Check primitive cell fields consistency
        primitive_fields = ['primitive_a', 'primitive_b', 'primitive_c', 'primitive_alpha', 'primitive_beta', 'primitive_gamma', 'space_group']
        ref_prim = all(getattr(reference, field) is not None for field in primitive_fields)
        gen_prim = all(getattr(generated, field) is not None for field in primitive_fields)
        faithfulness_scores.append(1.0 if ref_prim == gen_prim else 0.0)
        
        # Check value ranges
        for field in MaterialProperties.model_fields:
            ref_value = getattr(reference, field)
            gen_value = getattr(generated, field)
            if ref_value is not None and gen_value is not None:
                if isinstance(ref_value, (int, float)):
                    # Check if values are in similar ranges
                    if ref_value != 0:
                        ratio = abs(gen_value / ref_value)
                        if 0.1 <= ratio <= 10:  # Allow for reasonable range
                            faithfulness_scores.append(1.0)
                        else:
                            faithfulness_scores.append(0.0)
        
        return float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
    
    @staticmethod
    def _calculate_hallucination_score(reference: MaterialProperties, generated: MaterialProperties) -> float:
        """
        Calculate hallucination score based on physical plausibility and relative deviations.
        
        Scoring rules:
        - Impossible values (≤0) or >25% error = 1.0
        - >10% and ≤25% error = 0.5
        - ≤10% error = 0.0
        
        The final score is averaged over all checked properties.
        
        Args:
            reference: Reference material properties
            generated: Generated material properties
            
        Returns:
            float: Hallucination score between 0 and 1
        """
        if reference is None or generated is None:
            return 1.0
            
        scores = []
        
        # Check all relevant properties
        properties_to_check = [
            'density', 'cell_volume', 'atomic_count',
            'lattice_a', 'lattice_b', 'lattice_c',
            'nn_distance', 'primitive_a', 'primitive_b', 'primitive_c'
        ]
        
        for prop in properties_to_check:
            gen_value = getattr(generated, prop)
            ref_value = getattr(reference, prop)
            
            # Check for impossible values
            if gen_value is not None and isinstance(gen_value, (int, float)) and gen_value <= 0:
                scores.append(1.0)
                continue
                
            # Check relative error if both values exist and reference is non-zero
            if ref_value is not None and gen_value is not None and isinstance(ref_value, (int, float)) and ref_value != 0:
                rel = abs(gen_value - ref_value) / abs(ref_value)
                if rel <= 0.10:
                    scores.append(0.0)
                elif rel <= 0.25:
                    scores.append(0.5)
                else:
                    scores.append(1.0)
        
        return float(np.mean(scores)) if scores else 0.0
    
    @classmethod
    def from_group(cls, results: List['Metrics'], output_dir: Optional[str] = None) -> 'Metrics':
        """Calculate group-level metrics from a list of individual results"""
        if not results:
            return cls(average_error=float('inf'))
            
        # Collect all errors for each property
        property_errors = {}
        for result in results:
            for field_name in cls.model_fields:
                if field_name.endswith('_error') and getattr(result, field_name) is not None:
                    if field_name not in property_errors:
                        property_errors[field_name] = []
                    value = getattr(result, field_name)
                    if np.isfinite(value):  # Only include finite values
                        property_errors[field_name].append(value)
        
        # Calculate standard deviations
        property_errors_std = {}
        for prop, errors in property_errors.items():
            if len(errors) > 1:
                std = np.std(errors)
                if np.isfinite(std):
                    property_errors_std[prop] = float(std)
        
        # Calculate 95% confidence intervals
        confidence_interval_95 = {}
        for prop, errors in property_errors.items():
            if len(errors) > 1:
                mean = np.mean(errors)
                std = np.std(errors)
                if np.isfinite(mean) and np.isfinite(std):
                    ci = (mean - 1.96 * std / np.sqrt(len(errors)),
                          mean + 1.96 * std / np.sqrt(len(errors)))
                    if all(np.isfinite(x) for x in ci):
                        confidence_interval_95[prop] = tuple(map(float, ci))
        
        # Calculate error correlations
        error_correlations = {}
        valid_properties = {
            prop: errors for prop, errors in property_errors.items()
            if len(errors) > 1 and all(np.isfinite(v) for v in errors)
        }
        
        if valid_properties:
            property_names = list(valid_properties.keys())
            error_matrix = np.array([valid_properties[prop] for prop in property_names])
            
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
                                if np.sum(valid_mask) >= 2:  # Need at least 2 points for correlation
                                    numerator = np.sum(masked_i[valid_mask] * masked_j[valid_mask])
                                    denominator = std_i * std_j * np.sum(valid_mask)
                                    if denominator != 0:
                                        c[i, j] = float(numerator / denominator)
                
                # Set diagonal to 1.0
                np.fill_diagonal(c, 1.0)
                
                # Check if we have any valid correlations
                if not np.any(c != 0):
                    print("Warning: No valid correlations could be calculated")
                    return
                
                # Save correlation data if output_dir is provided
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
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
        
        # Calculate average errors for each property
        avg_errors = {}
        for prop, errors in property_errors.items():
            if errors:  # Only calculate if we have valid errors
                mean = np.mean(errors)
                if np.isfinite(mean):
                    avg_errors[prop] = float(mean)
        
        # Calculate overall average error
        valid_errors = [r.average_error for r in results if np.isfinite(r.average_error)]
        overall_avg_error = float(np.mean(valid_errors)) if valid_errors else float('inf')
        
        # Calculate prediction consistency
        prediction_consistency = 1.0  # Default value
        try:
            # Group results by rotation index
            rotation_groups = {}
            for result in results:
                if not hasattr(result, 'rotation_index'):
                    continue
                if result.rotation_index not in rotation_groups:
                    rotation_groups[result.rotation_index] = []
                rotation_groups[result.rotation_index].append(result)
            
            if len(rotation_groups) > 1:  # Need at least 2 rotations to calculate consistency
                # Calculate mean error for each rotation
                rotation_means = []
                for rot_idx, rot_results in rotation_groups.items():
                    valid_errors = [r.average_error for r in rot_results if np.isfinite(r.average_error)]
                    if valid_errors:
                        mean_error = np.mean(valid_errors)
                        if np.isfinite(mean_error):
                            rotation_means.append(mean_error)
                
                if len(rotation_means) > 1:
                    # Calculate standard deviation of mean errors across rotations
                    std_means = np.std(rotation_means)
                    mean_of_means = np.mean(rotation_means)
                    
                    if np.isfinite(std_means) and np.isfinite(mean_of_means) and mean_of_means > 0:
                        # Normalize by the mean of means to get a relative measure
                        prediction_consistency = 1.0 - min(std_means / mean_of_means, 1.0)
                    else:
                        prediction_consistency = 1.0  # Default to perfect consistency if calculation fails
                else:
                    prediction_consistency = 1.0  # Default to perfect consistency if not enough rotations
            else:
                prediction_consistency = 1.0  # Default to perfect consistency if only one rotation
                
        except Exception as e:
            print(f"Error calculating prediction consistency: {str(e)}")
            prediction_consistency = 1.0  # Default to perfect consistency on error
        
        # Calculate average knowledge metrics
        knowledge_metrics = {}
        for metric in ['physical_law_compliance', 'format_faithfulness', 'hallucination_score']:
            values = [getattr(r, metric) for r in results if getattr(r, metric) is not None and np.isfinite(getattr(r, metric))]
            if values:
                knowledge_metrics[metric] = float(np.mean(values))
        
        # Create metrics instance with all calculated values
        metrics = cls(
            property_errors_std=property_errors_std,
            confidence_interval_95=confidence_interval_95,
            error_correlations=error_correlations,
            average_error=overall_avg_error,
            prediction_consistency=prediction_consistency,
            **knowledge_metrics
        )
        
        # Set individual property errors
        for prop, value in avg_errors.items():
            setattr(metrics, prop, value)
        
        return metrics
   
class Result(BaseModel):
    material: str
    held_out_r: str
    rotation_index: int
    reference_annotation: str
    generated_annotation: str
    metrics: Metrics
    processing_time: Optional[float] = Field(None, description="Processing time for this result in seconds")