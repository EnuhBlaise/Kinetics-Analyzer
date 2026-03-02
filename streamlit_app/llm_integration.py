"""
LLM Integration module for AI-powered result interpretation.

Uses Hugging Face Inference API to analyze fitting results and provide
genuine, dynamic AI-generated recommendations for parameter adjustments.

This module calls actual LLM models for analysis - the AI generates its own
interpretations based on the scientific context provided, NOT pre-written rules.
"""

import os
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LLMAnalyzer:
    """
    Analyzes kinetic parameter fitting results using genuine LLM inference.
    
    The LLM generates its own analysis based on the data - not template-based rules.
    """
    
    # Available models in order of preference
    AVAILABLE_MODELS = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "google/gemma-2-9b-it",
    ]
    
    def __init__(self, api_token: str = None, model_id: str = None):
        """
        Initialize the LLM analyzer.
        
        Args:
            api_token: Hugging Face API token
            model_id: Model ID to use for inference
        """
        self.api_token = api_token or os.getenv("HF_API_TOKEN", "")
        self.model_id = model_id or os.getenv("HF_MODEL_ID", self.AVAILABLE_MODELS[0])
        self.client = None
        self.last_error = None
        
        if HF_AVAILABLE and self.api_token:
            try:
                self.client = InferenceClient(token=self.api_token)
            except Exception as e:
                self.last_error = str(e)
                print(f"Warning: Could not initialize HF client: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None and bool(self.api_token)
    
    def _try_chat_completion(self, messages: List[Dict[str, str]], model: str) -> Optional[str]:
        """
        Try to get a chat completion from a specific model.
        
        Returns the generated text or None if failed.
        """
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=2000,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.last_error = f"{model}: {str(e)}"
            return None
    
    def _try_text_generation(self, prompt: str, model: str) -> Optional[str]:
        """
        Fallback to text generation API if chat completion fails.
        
        Returns the generated text or None if failed.
        """
        try:
            response = self.client.text_generation(
                prompt,
                model=model,
                max_new_tokens=2000,
                temperature=0.7,
                do_sample=True
            )
            return response
        except Exception as e:
            self.last_error = f"{model} (text_gen): {str(e)}"
            return None
    
    def analyze_results(
        self,
        parameters: Dict[str, float],
        statistics: Dict[str, float],
        confidence_intervals: Dict[str, Dict[str, float]] = None,
        model_type: str = "dual_monod_lag",
        substrate_name: str = "Unknown",
        predictions_df: pd.DataFrame = None,
        experimental_df: pd.DataFrame = None,
        conditions: List[str] = None
    ) -> str:
        """
        Analyze fitting results using genuine LLM inference.
        
        The LLM generates its own analysis - this is NOT template-based.
        
        Args:
            parameters: Fitted parameter values
            statistics: Fit statistics (R², RMSE, etc.)
            confidence_intervals: Parameter confidence intervals
            model_type: Type of model used
            substrate_name: Name of substrate
            predictions_df: Model predictions DataFrame
            experimental_df: Experimental data DataFrame
            conditions: List of experimental conditions
            
        Returns:
            LLM-generated analysis and recommendations
        """
        model_used = "No LLM Available"
        analysis = None
        
        if self.is_available():
            # Build the comprehensive prompt for genuine LLM analysis
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                parameters, statistics, confidence_intervals, model_type, substrate_name,
                predictions_df, experimental_df, conditions
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Try each model until one works
            for model in [self.model_id] + self.AVAILABLE_MODELS:
                if analysis:
                    break
                    
                # Try chat completion first (preferred)
                analysis = self._try_chat_completion(messages, model)
                if analysis:
                    model_used = model
                    break
                
                # Fallback to text generation
                full_prompt = self._build_instruct_prompt(system_prompt, user_prompt)
                analysis = self._try_text_generation(full_prompt, model)
                if analysis:
                    model_used = f"{model} (text generation)"
                    break
        
        # If no LLM available, provide minimal guidance to set up API
        if not analysis:
            analysis = self._get_setup_guide(
                parameters, statistics, model_type, substrate_name,
                predictions_df, experimental_df, conditions
            )
            model_used = "Setup Guide (LLM not configured)"
        
        # Add header with model info
        header = f"""## 🤖 AI Analysis Report

**Substrate:** {substrate_name}  
**Kinetic Model:** {self._format_model_name(model_type)}  
**Analysis Engine:** {model_used}

---

"""
        
        # Add disclaimer at the end
        disclaimer = """

---

⚠️ **Important Disclaimer:** This analysis was generated by an AI Large Language Model (LLM). 

**LLMs can make mistakes**, including:
- Misinterpreting parameter values or their significance
- Providing recommendations that may not apply to your specific experimental conditions
- Hallucinating information that sounds plausible but is incorrect
- Missing important context about your particular system

**Always:**
1. Verify AI-generated insights with domain expertise
2. Cross-check recommendations against published literature
3. Validate suggestions through experimental testing
4. Consult qualified researchers familiar with microbial kinetics before making critical decisions

The analysis above reflects the model's interpretation and should be used as a starting point for discussion, not as definitive guidance.
"""
        
        return header + analysis + disclaimer
    
    def _format_model_name(self, model_type: str) -> str:
        """Format model type for display."""
        model_names = {
            'single_monod': 'Single Monod',
            'single_haldane': 'Single Monod (Haldane)',
            'dual_monod': 'Dual Monod',
            'dual_haldane': 'Dual Monod (Haldane)',
            'dual_monod_lag': 'Dual Monod + Lag',
            'dual_haldane_lag': 'Dual Monod + Lag (Haldane)',
        }
        return model_names.get(model_type, model_type)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines the AI's expertise and behavior."""
        return """You are an expert scientist specializing in microbial kinetics, bioprocess engineering, and mathematical modeling of biological systems. You have deep knowledge of:

- Monod kinetics and its extensions (substrate inhibition, dual substrate models)
- Parameter estimation and statistical analysis
- Microbial growth dynamics and metabolism
- Bioprocess optimization and scale-up

Your task is to analyze parameter fitting results from a kinetic modeling experiment and provide your genuine scientific interpretation. Be thorough, insightful, and practical.

IMPORTANT GUIDELINES:
1. Provide YOUR OWN analysis based on the data - do not use generic templates
2. Be specific about what the numbers tell you about this particular system
3. Identify any concerning patterns or values that warrant attention
4. Give actionable recommendations tailored to the observed results
5. Be honest about uncertainty - if something is unclear, say so
6. Connect the parameters to biological meaning where possible
7. Consider what the confidence intervals tell you about parameter identifiability

Format your response with clear markdown sections. Be verbose and thorough - the user wants detailed insights, not brief summaries."""

    def _build_user_prompt(
        self,
        parameters: Dict[str, float],
        statistics: Dict[str, float],
        confidence_intervals: Dict[str, Dict[str, float]],
        model_type: str,
        substrate_name: str,
        predictions_df: pd.DataFrame = None,
        experimental_df: pd.DataFrame = None,
        conditions: List[str] = None
    ) -> str:
        """Build the user prompt with all the fitting results data including predictions."""
        
        # Format parameters with CIs
        param_lines = []
        for name, value in parameters.items():
            ci = confidence_intervals.get(name, {}) if confidence_intervals else {}
            std_err = ci.get('std_error')
            lower = ci.get('lower_bound')
            upper = ci.get('upper_bound')
            
            line = f"  {name}: {value:.6f}"
            if std_err and std_err == std_err:  # not NaN
                line += f" ± {std_err:.6f} (SE)"
                if lower is not None and upper is not None:
                    line += f" [95% CI: {lower:.6f} to {upper:.6f}]"
            param_lines.append(line)
        
        params_text = "\n".join(param_lines)
        
        # Model descriptions
        model_info = {
            'single_monod': """Single Monod Model with Substrate Inhibition
dS/dt = -qmax * (S / (Ks + S)) * (Ki / (Ki + S)) * X
dX/dt = Y * qmax * (S / (Ks + S)) * (Ki / (Ki + S)) * X - b_decay * X
Parameters: qmax (max specific uptake), Ks (half-saturation), Ki (inhibition constant), Y (yield), b_decay (decay rate)""",
            
            'dual_monod': """Dual Monod Model (Substrate + Oxygen Limitation)
dS/dt = -qmax * (S / (Ks + S)) * (Ki / (Ki + S)) * (O2 / (K_o2 + O2)) * X
dX/dt = Y * qmax * (S / (Ks + S)) * (Ki / (Ki + S)) * (O2 / (K_o2 + O2)) * X - b_decay * X
dO2/dt = kLa * (O2_sat - O2) - Y_o2 * qmax * (...) * X
Parameters: qmax, Ks, Ki, Y, b_decay, K_o2 (oxygen half-sat), Y_o2 (oxygen yield)""",
            
            'dual_monod_lag': """Dual Monod Model with Lag Phase
Same as Dual Monod, but with a lag function: growth_factor = 1 - exp(-t / lag_time)
This accounts for microbial adaptation before exponential growth begins.
Parameters: qmax, Ks, Ki, Y, b_decay, K_o2, Y_o2, lag_time"""
        }
        
        # Statistics formatting
        r_squared = statistics.get('R_squared', 0)
        rmse = statistics.get('RMSE', 0)
        aic = statistics.get('AIC', 0)
        bic = statistics.get('BIC', 0)
        
        # Build experimental data summary
        exp_data_summary = self._summarize_experimental_data(experimental_df, substrate_name, conditions)
        
        # Build predictions summary and residual analysis
        predictions_summary = self._summarize_predictions(predictions_df, experimental_df, substrate_name, conditions)
        
        prompt = f"""Please analyze these kinetic parameter fitting results:

## SUBSTRATE INFORMATION
Substrate being modeled: {substrate_name}

## KINETIC MODEL USED
{model_info.get(model_type, f'Model type: {model_type}')}

## FITTED PARAMETERS (with uncertainty estimates)
{params_text}

## FIT STATISTICS
- R² (coefficient of determination): {r_squared:.6f}
- RMSE (root mean square error): {rmse:.6f}
- AIC (Akaike Information Criterion): {aic:.2f}
- BIC (Bayesian Information Criterion): {bic:.2f}

{exp_data_summary}

{predictions_summary}

## WHAT I NEED FROM YOU

Please provide a comprehensive analysis covering:

### 1. Overall Fit Quality
Evaluate the R² and RMSE values. What do they tell you about how well this model captures the experimental data? Are there any red flags? Look at the residual analysis provided.

### 2. Data and Prediction Analysis
Based on the experimental data and model predictions provided:
- How well does the model capture the substrate consumption dynamics?
- Are there systematic deviations at certain time points or conditions?
- Do any conditions show noticeably worse fits than others?

### 3. Parameter Interpretation
For each fitted parameter:
- Is the value biologically reasonable for {substrate_name} metabolism?
- What does this specific value tell us about the microbial system?
- Are there any parameters that seem unusual or concerning?

### 4. Uncertainty Analysis
Looking at the standard errors and confidence intervals:
- Which parameters are well-constrained by the data?
- Which parameters have high uncertainty, and what does this mean?
- Are any parameters potentially unidentifiable from this data?

### 5. Model Suitability
Is this model (the {model_type}) appropriate for this system? Would a different model potentially work better based on the residual patterns?

### 6. Practical Recommendations
Based on your analysis:
- What specific steps could improve the fit?
- Should the user try different parameter bounds?
- Any suggestions for additional experiments or data collection?
- What are the key takeaways for using these parameters in practice?

Be thorough and specific to this data - I want your genuine scientific assessment, not generic advice."""

        return prompt
    
    def _summarize_experimental_data(
        self,
        df: pd.DataFrame,
        substrate_name: str,
        conditions: List[str] = None
    ) -> str:
        """Summarize experimental data for the LLM."""
        if df is None or df.empty:
            return "## EXPERIMENTAL DATA\nNo experimental data provided."
        
        lines = ["## EXPERIMENTAL DATA"]
        
        # Get time column
        time_col = None
        for col in df.columns:
            if 'time' in col.lower():
                time_col = col
                break
        
        if time_col:
            time_data = df[time_col].dropna()
            lines.append(f"\n**Time Range:** {time_data.min():.3f} to {time_data.max():.3f} days")
            lines.append(f"**Number of Time Points:** {len(time_data)}")
        
        # Summarize each condition
        if conditions:
            lines.append(f"\n**Experimental Conditions:** {', '.join(conditions)}")
            lines.append("\n**Condition-wise Data Summary:**")
            
            for cond in conditions:
                # Find substrate and biomass columns for this condition
                sub_col = None
                bio_col = None
                for col in df.columns:
                    if cond in col:
                        if 'biomass' in col.lower():
                            bio_col = col
                        elif '(mg/l)' in col.lower() or '(mm)' in col.lower():
                            sub_col = col
                
                if sub_col and sub_col in df.columns:
                    sub_data = df[sub_col].dropna()
                    lines.append(f"\n  **{cond}:**")
                    lines.append(f"    - Initial {substrate_name}: {sub_data.iloc[0]:.2f} mg/L")
                    lines.append(f"    - Final {substrate_name}: {sub_data.iloc[-1]:.2f} mg/L")
                    lines.append(f"    - Consumption: {sub_data.iloc[0] - sub_data.iloc[-1]:.2f} mg/L ({((sub_data.iloc[0] - sub_data.iloc[-1])/sub_data.iloc[0]*100):.1f}%)")
                    
                    if bio_col and bio_col in df.columns:
                        bio_data = df[bio_col].dropna()
                        lines.append(f"    - Initial Biomass: {bio_data.iloc[0]:.4f} OD")
                        lines.append(f"    - Final Biomass: {bio_data.iloc[-1]:.4f} OD")
                        lines.append(f"    - Growth: {bio_data.iloc[-1] - bio_data.iloc[0]:.4f} OD ({((bio_data.iloc[-1] - bio_data.iloc[0])/bio_data.iloc[0]*100):.1f}%)")
        
        return "\n".join(lines)
    
    def _summarize_predictions(
        self,
        predictions_df: pd.DataFrame,
        experimental_df: pd.DataFrame,
        substrate_name: str,
        conditions: List[str] = None
    ) -> str:
        """Summarize model predictions and calculate residuals for the LLM."""
        if predictions_df is None or predictions_df.empty:
            return "## MODEL PREDICTIONS\nNo prediction data available."
        
        lines = ["## MODEL PREDICTIONS AND RESIDUAL ANALYSIS"]
        
        # Check what columns are available
        if 'Time' in predictions_df.columns:
            lines.append(f"\n**Simulation Time Points:** {len(predictions_df['Time'].unique())} points")
        
        if 'Condition' in predictions_df.columns:
            pred_conditions = predictions_df['Condition'].unique()
            lines.append(f"**Conditions Simulated:** {', '.join(str(c) for c in pred_conditions)}")
            
            # Analyze each condition
            lines.append("\n**Condition-wise Prediction Summary:**")
            
            for cond in pred_conditions:
                cond_pred = predictions_df[predictions_df['Condition'] == cond]
                
                lines.append(f"\n  **{cond}:**")
                
                # Substrate predictions
                if 'Substrate' in cond_pred.columns:
                    sub_pred = cond_pred['Substrate']
                    lines.append(f"    - Predicted Initial Substrate: {sub_pred.iloc[0]:.2f} mg/L")
                    lines.append(f"    - Predicted Final Substrate: {sub_pred.iloc[-1]:.2f} mg/L")
                    lines.append(f"    - Predicted Consumption: {sub_pred.iloc[0] - sub_pred.iloc[-1]:.2f} mg/L")
                
                # Biomass predictions
                if 'Biomass' in cond_pred.columns:
                    bio_pred = cond_pred['Biomass']
                    lines.append(f"    - Predicted Initial Biomass: {bio_pred.iloc[0]:.4f}")
                    lines.append(f"    - Predicted Final Biomass: {bio_pred.iloc[-1]:.4f}")
                    lines.append(f"    - Predicted Growth: {bio_pred.iloc[-1] - bio_pred.iloc[0]:.4f}")
            
            # Calculate residuals if experimental data is available
            if experimental_df is not None and not experimental_df.empty:
                lines.append("\n**Residual Analysis (Model - Experimental):**")
                residual_stats = self._calculate_residuals(predictions_df, experimental_df, conditions)
                if residual_stats:
                    for cond, stats in residual_stats.items():
                        lines.append(f"\n  **{cond}:**")
                        if 'substrate' in stats:
                            s = stats['substrate']
                            lines.append(f"    - Substrate Mean Residual: {s['mean']:.2f} mg/L")
                            lines.append(f"    - Substrate Max Absolute Residual: {s['max_abs']:.2f} mg/L")
                            lines.append(f"    - Substrate Residual Std Dev: {s['std']:.2f} mg/L")
                        if 'biomass' in stats:
                            b = stats['biomass']
                            lines.append(f"    - Biomass Mean Residual: {b['mean']:.4f}")
                            lines.append(f"    - Biomass Max Absolute Residual: {b['max_abs']:.4f}")
        
        return "\n".join(lines)
    
    def _calculate_residuals(
        self,
        predictions_df: pd.DataFrame,
        experimental_df: pd.DataFrame,
        conditions: List[str] = None
    ) -> Dict:
        """Calculate residuals between predictions and experimental data."""
        residuals = {}
        
        if conditions is None:
            return residuals
        
        # Get time column from experimental data
        time_col = None
        for col in experimental_df.columns:
            if 'time' in col.lower():
                time_col = col
                break
        
        if time_col is None:
            return residuals
        
        for cond in conditions:
            cond_residuals = {}
            
            # Find experimental columns for this condition
            exp_sub_col = None
            exp_bio_col = None
            for col in experimental_df.columns:
                if cond in col:
                    if 'biomass' in col.lower():
                        exp_bio_col = col
                    elif '(mg/l)' in col.lower() or '(mm)' in col.lower():
                        exp_sub_col = col
            
            # Get predictions for this condition
            if 'Condition' in predictions_df.columns:
                cond_pred = predictions_df[predictions_df['Condition'] == cond]
            else:
                continue
            
            if cond_pred.empty:
                continue
            
            # Calculate substrate residuals
            if exp_sub_col and 'Substrate' in cond_pred.columns:
                try:
                    exp_times = experimental_df[time_col].values
                    exp_vals = experimental_df[exp_sub_col].values
                    
                    # Interpolate predictions to experimental time points
                    pred_times = cond_pred['Time'].values
                    pred_vals = cond_pred['Substrate'].values
                    
                    interp_pred = np.interp(exp_times, pred_times, pred_vals)
                    res = interp_pred - exp_vals
                    
                    # Filter out NaN
                    res = res[~np.isnan(res)]
                    if len(res) > 0:
                        cond_residuals['substrate'] = {
                            'mean': float(np.mean(res)),
                            'std': float(np.std(res)),
                            'max_abs': float(np.max(np.abs(res)))
                        }
                except Exception:
                    pass
            
            # Calculate biomass residuals
            if exp_bio_col and 'Biomass' in cond_pred.columns:
                try:
                    exp_times = experimental_df[time_col].values
                    exp_vals = experimental_df[exp_bio_col].values
                    
                    pred_times = cond_pred['Time'].values
                    pred_vals = cond_pred['Biomass'].values
                    
                    interp_pred = np.interp(exp_times, pred_times, pred_vals)
                    res = interp_pred - exp_vals
                    
                    res = res[~np.isnan(res)]
                    if len(res) > 0:
                        cond_residuals['biomass'] = {
                            'mean': float(np.mean(res)),
                            'std': float(np.std(res)),
                            'max_abs': float(np.max(np.abs(res)))
                        }
                except Exception:
                    pass
            
            if cond_residuals:
                residuals[cond] = cond_residuals
        
        return residuals
    
    def _build_instruct_prompt(self, system: str, user: str) -> str:
        """Build an instruct-format prompt for text generation API."""
        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST]"""

    def _get_setup_guide(
        self,
        parameters: Dict[str, float],
        statistics: Dict[str, float],
        model_type: str,
        substrate_name: str,
        predictions_df: pd.DataFrame = None,
        experimental_df: pd.DataFrame = None,
        conditions: List[str] = None
    ) -> str:
        """Provide setup guide when LLM is not configured."""
        r_squared = statistics.get('R_squared', 0)
        rmse = statistics.get('RMSE', 0)
        
        # Build basic data summary
        data_summary = ""
        if conditions:
            data_summary = f"\n**Conditions Analyzed:** {', '.join(conditions)}"
        
        return f"""### ⚙️ LLM Not Configured

To get AI-powered analysis of your results, please configure a Hugging Face API token:

1. **Get a free API token:**
   - Go to [huggingface.co](https://huggingface.co) and create an account
   - Navigate to Settings → Access Tokens
   - Create a new token with "Read" permission

2. **Enter your token:**
   - Paste the token in the "Hugging Face API Token" field in the sidebar
   - The AI analysis will automatically activate

---

### 📊 Basic Results Summary

**Substrate:** {substrate_name}
**Model:** {self._format_model_name(model_type)}{data_summary}

**Quick Statistics:**
- R² = {r_squared:.4f} {'✅ Good fit' if r_squared > 0.8 else '⚠️ May need improvement' if r_squared > 0.5 else '❌ Poor fit'}
- RMSE = {rmse:.4f}

**Fitted Parameters:**
{chr(10).join(f'- {k}: {v:.4f}' for k, v in parameters.items())}

---

*Configure an API token above to receive detailed AI interpretation including analysis of your experimental data, model predictions, and residuals.*"""
    
def get_llm_analyzer(api_token: str = None) -> LLMAnalyzer:
    """
    Factory function to get an LLM analyzer instance.
    
    Args:
        api_token: Optional Hugging Face API token
        
    Returns:
        LLMAnalyzer instance
    """
    return LLMAnalyzer(api_token=api_token)

