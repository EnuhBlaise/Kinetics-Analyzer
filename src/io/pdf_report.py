"""
PDF report generation for kinetic parameter estimation results.

Generates a single PDF containing all summary text, tables, and figures
produced during a fitting run. Uses matplotlib's PdfPages backend — no
extra dependencies beyond matplotlib.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import textwrap
import numpy as np


def _wrap_text(text: str, width: int = 95) -> str:
    """Wrap long lines for PDF text rendering."""
    lines = text.split('\n')
    wrapped = []
    for line in lines:
        if len(line) > width and not line.startswith('=') and not line.startswith('-'):
            wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent='  '))
        else:
            wrapped.append(line)
    return '\n'.join(wrapped)


def _render_text_page(pdf: PdfPages, text: str, title: str = None,
                      fontsize: float = 8.5) -> None:
    """
    Render a block of monospaced text onto one or more PDF pages.

    Long text is automatically split across pages.
    """
    max_lines_per_page = 58  # fits A4 with margins at fontsize ~8.5

    lines = _wrap_text(text).split('\n')

    # If a title is provided, prepend it
    if title:
        lines = [title, '=' * len(title), ''] + lines

    # Paginate
    for page_start in range(0, len(lines), max_lines_per_page):
        page_lines = lines[page_start:page_start + max_lines_per_page]
        page_text = '\n'.join(page_lines)

        fig = plt.figure(figsize=(8.5, 11))  # US letter
        fig.text(
            0.05, 0.95, page_text,
            transform=fig.transFigure,
            fontsize=fontsize,
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='left',
        )
        fig.patch.set_facecolor('white')
        plt.axis('off')
        pdf.savefig(fig, facecolor='white')
        plt.close(fig)


def _render_title_page(pdf: PdfPages, title: str, subtitle: str,
                       metadata: Dict[str, str]) -> None:
    """Render a cover / title page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.7, title,
             fontsize=22, fontweight='bold',
             ha='center', va='center',
             fontfamily='serif')

    fig.text(0.5, 0.62, subtitle,
             fontsize=14, ha='center', va='center',
             fontfamily='serif', color='#555555')

    # Metadata block
    meta_lines = [f'{k}: {v}' for k, v in metadata.items()]
    meta_text = '\n'.join(meta_lines)
    fig.text(0.5, 0.45, meta_text,
             fontsize=10, ha='center', va='center',
             fontfamily='monospace', color='#333333',
             linespacing=1.8)

    fig.text(0.5, 0.05,
             f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             fontsize=8, ha='center', color='#999999')

    plt.axis('off')
    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def _embed_figure(pdf: PdfPages, figure_path: Path, caption: str = None) -> None:
    """Embed an existing figure image (PNG/PDF) into the report."""
    if not figure_path.exists():
        return

    try:
        img = plt.imread(str(figure_path))
    except Exception:
        return

    # Determine aspect ratio
    h, w = img.shape[:2]
    aspect = w / h
    fig_width = 7.5
    fig_height = fig_width / aspect

    # Cap height to leave room for caption
    max_height = 9.5
    if fig_height > max_height:
        fig_height = max_height
        fig_width = fig_height * aspect

    fig = plt.figure(figsize=(8.5, min(fig_height + 1.5, 11)))
    fig.patch.set_facecolor('white')

    # Add caption at top
    if caption:
        fig.text(0.5, 0.97, caption,
                 fontsize=10, fontweight='bold',
                 ha='center', va='top', fontfamily='serif')

    ax = fig.add_axes([
        (8.5 - fig_width) / (2 * 8.5),   # left
        0.05,                               # bottom
        fig_width / 8.5,                    # width
        fig_height / (fig_height + 1.5)     # height
    ])
    ax.imshow(img)
    ax.axis('off')

    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def _embed_matplotlib_figure(pdf: PdfPages, fig: plt.Figure,
                              caption: str = None) -> None:
    """Embed a live matplotlib Figure object directly into the report."""
    if caption:
        fig.suptitle(caption, fontsize=10, fontweight='bold', y=0.99)
    pdf.savefig(fig, facecolor='white', bbox_inches='tight')


# ── Public API ──────────────────────────────────────────────────────

def generate_workflow_report(
    output_dir: Path,
    summary_text: str,
    model_type: str,
    substrate_name: str,
    statistics: Dict[str, Any],
    parameters: Dict[str, float],
    confidence_intervals: Dict[str, Dict[str, float]] = None,
    figure_paths: List[Path] = None,
    extra_sections: Dict[str, str] = None,
    filename: str = "results_report.pdf"
) -> Path:
    """
    Generate a PDF report for a base workflow (global/individual fit).

    Args:
        output_dir: Directory to save the PDF
        summary_text: Full text summary (from WorkflowResult.summary())
        model_type: Model identifier string
        substrate_name: Name of the substrate
        statistics: Fit statistics dictionary
        parameters: Fitted parameter values
        confidence_intervals: Parameter CIs (optional)
        figure_paths: Paths to figure files to embed
        extra_sections: Additional named text sections
        filename: Output PDF filename

    Returns:
        Path to the generated PDF
    """
    output_path = Path(output_dir) / filename
    figure_paths = figure_paths or []
    extra_sections = extra_sections or {}

    with PdfPages(str(output_path)) as pdf:
        # ── Title page ──
        _render_title_page(
            pdf,
            title='Kinetic Parameter Estimation Report',
            subtitle=f'{model_type} — {substrate_name}',
            metadata={
                'Model': model_type,
                'Substrate': substrate_name,
                'Parameters': str(len(parameters)),
                'R² (combined)': f"{statistics.get('R_squared', 'N/A'):.4f}" if isinstance(statistics.get('R_squared'), (int, float)) else 'N/A',
            }
        )

        # ── Summary text ──
        _render_text_page(pdf, summary_text, title='Fit Summary')

        # ── Extra sections (e.g. robust fit summary, bootstrap) ──
        for section_title, section_text in extra_sections.items():
            _render_text_page(pdf, section_text, title=section_title)

        # ── Figures ──
        for fig_path in figure_paths:
            fig_path = Path(fig_path)
            if fig_path.exists() and fig_path.suffix.lower() == '.png':
                caption = fig_path.stem.replace('_', ' ').title()
                _embed_figure(pdf, fig_path, caption=caption)

    return output_path


def generate_individual_condition_report(
    output_dir: Path,
    summary_text: str,
    model_type: str,
    substrate_name: str,
    condition_results: Dict[str, Any],
    parameter_summary: Dict[str, Dict[str, float]],
    global_parameters: Optional[Dict[str, float]] = None,
    global_loss: Optional[float] = None,
    figure_paths: List[Path] = None,
    filename: str = "results_report.pdf"
) -> Path:
    """
    Generate a PDF report for individual condition fitting.

    Args:
        output_dir: Directory to save the PDF
        summary_text: Full text from IndividualConditionResult.summary()
        model_type: Model identifier string
        substrate_name: Name of the substrate
        condition_results: Dict of ConditionResult objects
        parameter_summary: Cross-condition parameter statistics
        global_parameters: Global parameter estimates (optional)
        global_loss: Global cost function value (optional)
        figure_paths: Paths to figure files to embed
        filename: Output PDF filename

    Returns:
        Path to the generated PDF
    """
    output_path = Path(output_dir) / filename
    figure_paths = figure_paths or []

    n_conditions = len(condition_results)
    n_successful = sum(1 for r in condition_results.values()
                       if getattr(r, 'success', True))

    with PdfPages(str(output_path)) as pdf:
        # ── Title page ──
        meta = {
            'Model': model_type,
            'Substrate': substrate_name,
            'Conditions': str(n_conditions),
            'Successful fits': f'{n_successful}/{n_conditions}',
        }
        if global_loss is not None:
            meta['Global loss'] = f'{global_loss:.6f}'

        _render_title_page(
            pdf,
            title='Individual Condition Fitting Report',
            subtitle=f'{model_type} — {substrate_name}',
            metadata=meta
        )

        # ── Full summary text (paginated) ──
        _render_text_page(pdf, summary_text, title='Complete Results Summary')

        # ── Figures ──
        for fig_path in figure_paths:
            fig_path = Path(fig_path)
            if fig_path.exists() and fig_path.suffix.lower() == '.png':
                caption = fig_path.stem.replace('_', ' ').title()
                _embed_figure(pdf, fig_path, caption=caption)

    return output_path


def generate_robust_fit_report(
    output_dir: Path,
    summary_text: str,
    substrate_name: str,
    figure_paths: List[Path] = None,
    bootstrap_info: Dict[str, Any] = None,
    filename: str = "results_report.pdf"
) -> Path:
    """
    Generate a PDF report for a robust fitting run.

    Args:
        output_dir: Directory to save the PDF
        summary_text: Full text summary from RobustFitResult.summary()
        substrate_name: Name of the substrate
        figure_paths: Paths to figure files to embed
        bootstrap_info: Bootstrap metadata (n_iterations, success_rate, etc.)
        filename: Output PDF filename

    Returns:
        Path to the generated PDF
    """
    output_path = Path(output_dir) / filename
    figure_paths = figure_paths or []
    bootstrap_info = bootstrap_info or {}

    with PdfPages(str(output_path)) as pdf:
        # ── Title page ──
        meta = {
            'Substrate': substrate_name,
            'Method': 'Robust (Two-Stage + Bootstrap)',
        }
        if bootstrap_info:
            meta['Bootstrap iterations'] = str(bootstrap_info.get('n_iterations', '?'))
            meta['Success rate'] = f"{bootstrap_info.get('success_rate', 0):.1%}"

        _render_title_page(
            pdf,
            title='Robust Parameter Estimation Report',
            subtitle=substrate_name,
            metadata=meta
        )

        # ── Summary ──
        _render_text_page(pdf, summary_text, title='Robust Fit Summary')

        # ── Figures ──
        for fig_path in figure_paths:
            fig_path = Path(fig_path)
            if fig_path.exists() and fig_path.suffix.lower() == '.png':
                caption = fig_path.stem.replace('_', ' ').title()
                _embed_figure(pdf, fig_path, caption=caption)

    return output_path
