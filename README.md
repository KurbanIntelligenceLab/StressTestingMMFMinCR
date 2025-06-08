# Stress-Testing Multimodal Foundation Models for Crystallographic Reasoning

This repository contains the code and analysis scripts for the paper "Stress-Testing Multimodal Foundation Models for Crystallographic Reasoning" submitted to ACL 2025 Workshop on Towards Knowledgeable Foundation Models. The project evaluates the performance of various multimodal foundation models on crystallographic reasoning tasks, with a particular focus on Minimum Critical Radius (MinCR) calculations.

## Dataset

The dataset used in this project is available at: [Dataset Link](https://figshare.com/s/4704f61d44a1f2ca63c5)

## Project Structure

```
.
├── analysis_scripts/
│   ├── transfer_degration.py      # Analysis of knowledge transfer degradation across materials
│   ├── correlation_calculation.py # Correlation analysis between model predictions
│   ├── latency_calculation       # Performance and latency analysis of model responses
│   └── hallucination_and_compliance.py # Analysis of model hallucinations and compliance with crystallographic constraints
├── llm_scripts/
│   ├── base_parser.py            # Base parser for multimodal model interactions
│   ├── leave_one_radius_out_validation.py # Leave-one-radius-out validation for MinCR predictions
│   └── leave_one_material_out_validation.py # Leave-one-material-out validation for material-specific reasoning
└── requirements.txt              # Project dependencies
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd StressTestingMMFMinCR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Scripts Overview

### Analysis Scripts

- `transfer_degration.py`: Analyzes how model performance degrades when transferring knowledge between different materials and radii, measuring the robustness of crystallographic reasoning.
- `correlation_calculation.py`: Calculates correlations between different models' predictions to understand agreement in crystallographic reasoning.
- `latency_calculation`: Measures and analyzes model response times for real-world applicability assessment.
- `hallucination_and_compliance.py`: Evaluates model hallucinations and compliance with crystallographic constraints and physical laws.

### LLM Scripts

- `base_parser.py`: Core functionality for interacting with various multimodal foundation model APIs.
- `leave_one_radius_out_validation.py`: Implements leave-one-radius-out cross-validation to test model generalization.
- `leave_one_material_out_validation.py`: Implements leave-one-material-out cross-validation to assess material-specific reasoning capabilities.

## Usage

1. First, ensure you have the dataset downloaded from the provided Figshare link.

2. Run the validation scripts:
```bash
python llm_scripts/leave_one_material_out_validation.py
python llm_scripts/leave_one_radius_out_validation.py
```

3. Analyze the results:
```bash
python analysis_scripts/transfer_degration.py
python analysis_scripts/correlation_calculation.py
python analysis_scripts/hallucination_and_compliance.py
```

## Dependencies

- numpy
- pandas
- requests
- Pillow
- openai
- rouge-score
- nltk
- tqdm
- pydantic

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{stress_testing_mmf_2025,
  title={Stress-Testing Multimodal Foundation Models for Crystallographic Reasoning},
  author={[Authors]},
  journal={ACL 2025 Workshop on Towards Knowledgeable Foundation Models},
  year={2025}
}
```

## License

This project is licensed under the terms included in the LICENSE file.