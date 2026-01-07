"""Core A/B test logic (ported from the original TypeScript project)."""

from .statistics import (
    ABTestResult,
    PowerAnalysisResult,
    z_test_proportions,
    calculate_power,
    calculate_required_sample_size,
    simulate_power_analysis,
    generate_recommendation,
)
from .data_generator import generate_experiment_data, export_to_csv
