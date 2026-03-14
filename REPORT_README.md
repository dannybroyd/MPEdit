# MPEdit Final Project Report

This directory contains the LaTeX source for the MPEdit final project report.

## Files

- `report.tex` - Main LaTeX document (6-page report + appendix)
- `references.bib` - Bibliography file with citations
- `scripts/inference/logs_experiments/figures/` - Directory containing experimental figures

## Compiling the Report

### Prerequisites

You need a LaTeX distribution installed:
- **Linux**: `sudo apt-get install texlive-full`
- **macOS**: Install MacTeX from https://www.tug.org/mactex/
- **Windows**: Install MiKTeX from https://miktex.org/

### Compilation

```bash
# Compile the PDF (run twice for references)
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex

# Or use latexmk for automatic compilation
latexmk -pdf report.tex
```

### Output

The compiled PDF will be `report.pdf`.

## Report Structure

### Main Body (6 pages, 12pt font, single column)

1. **Abstract** - Overview of MPEdit approach and key results
2. **Introduction** - Motivation, problem statement, and contributions
3. **Background** - MPD and SDEdit foundations
4. **Methodology** - SDEdit for motion planning algorithm and implementation
5. **Experiments** - Five comprehensive experiments with figure placeholders:
   - Experiment 1: Noise level sweep (success vs. faithfulness tradeoff)
   - Experiment 2: Baseline comparison (SDEdit vs. Full MPD vs. RRTConnect)
   - Experiment 3: Sketch-to-path quality analysis
   - Experiment 4: Inference speed analysis
   - Experiment 5: Output diversity analysis
6. **Results and Discussion** - Key findings and limitations
7. **Conclusion and Future Work** - Summary and research directions

### Appendix (Additional pages beyond main 6)

- **Sketch Mode Examples** - Interactive GUI demonstrations and qualitative results
- Includes placeholders for:
  - Path Sketcher GUI screenshot
  - Sketch refinement examples (3-panel figure)
  - Before-after visualization
  - Denoising process animations
  - Code examples

## Figure Placeholders

The report includes placeholders for the following experimental figures. Replace these with actual figures from your experiments:

### Main Body Figures

1. **Figure 1 (fig2_t0_sweep.pdf)** - Line plot showing success rate and Fréchet distance vs. noise level t₀
2. **Table 1 (table1_baselines.tex)** - Already included! Baseline comparison table
3. **Figure 2 (fig_baselines_comparison.pdf)** - Grouped bar chart comparing methods across scenarios
4. **Figure 3 (fig4_sketch_heatmap.pdf)** - Heatmap of sketch quality (σ) vs. noise level (t₀)
5. **Figure 4 (fig5_speed.pdf)** - Inference speed comparison vs. t₀
6. **Figure 5 (fig6_diversity.pdf)** - Output diversity analysis

### Appendix Figures

7. **Path Sketcher GUI** - Screenshot of the interactive sketching interface
8. **Sketch Refinement Examples** - 3-panel comparison of different sketch qualities
9. **Before-After Visualization** - 3-panel denoising process visualization
10. **Denoising Animation Reference** - Reference to MP4 videos

## Adding Actual Figures

To replace the placeholders with actual figures:

1. **Make sure figures exist** in `scripts/inference/logs_experiments/figures/`
2. **Edit report.tex** and replace the placeholder `\fbox{...}` blocks with:
   ```latex
   \includegraphics[width=0.9\textwidth]{scripts/inference/logs_experiments/figures/fig2_t0_sweep.pdf}
   ```
3. **For the table**, the actual LaTeX table is already included via:
   ```latex
   \input{scripts/inference/logs_experiments/figures/table1_baselines.tex}
   ```
4. **Recompile** the PDF

Example replacement for Figure 1:
```latex
% BEFORE (placeholder):
\begin{figure}[ht]
\centering
\fbox{\parbox{0.9\textwidth}{\centering
\vspace{1.5in}
\textbf{[FIGURE PLACEHOLDER]}
...
}}
\caption{...}
\end{figure}

% AFTER (actual figure):
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{scripts/inference/logs_experiments/figures/fig2_t0_sweep.pdf}
\caption{Noise level sweep: Success rate and Fréchet distance vs. $t_0$. The optimal operating point (marked with a star) balances both objectives.}
\label{fig:t0_sweep}
\end{figure}
```

## Customization

### Adjusting Page Count

The report is designed to fit approximately 6 pages for the main body plus appendix. If you need to adjust:

- **Reduce content**: Remove or condense sections, reduce figure sizes
- **Add content**: The appendix can be expanded indefinitely without violating the 6-page main body limit

### Font Size

The document uses 12pt font as required. Do not change the `\documentclass[12pt,...]` parameter.

### Margins

Current margins are 1 inch (standard). To adjust: `\usepackage[margin=1in]{geometry}`

### Column Format

The report uses single-column format as preferred. To switch to two-column, change:
```latex
\documentclass[12pt,twocolumn,letterpaper]{article}
```

## Running Experiments

To generate the actual experimental figures referenced in the report:

```bash
# From the repository root
source set_env_variables.sh
conda activate mpd-splines-public
cd scripts/inference

# Run all experiments (takes several hours)
python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_2d.yaml

# Generate figures
python ../experiments/plot_results.py --results_dir logs_experiments
```

This will populate `scripts/inference/logs_experiments/figures/` with all required figures.

## Project Structure Reference

```
MPEdit/
├── report.tex                           # Main LaTeX document (THIS FILE)
├── references.bib                       # Bibliography
├── REPORT_README.md                     # This README
├── README.md                            # Main project README
├── scripts/
│   ├── inference/
│   │   ├── inference_sdedit.py         # Batch SDEdit inference
│   │   ├── inference_sdedit_interactive.py  # Interactive GUI mode
│   │   └── logs_experiments/
│   │       └── figures/                # Experimental figures directory
│   │           ├── fig2_t0_sweep.pdf
│   │           ├── table1_baselines.tex  ← Already exists!
│   │           ├── fig_baselines_comparison.pdf
│   │           ├── fig4_sketch_heatmap.pdf
│   │           ├── fig5_speed.pdf
│   │           └── fig6_diversity.pdf
│   └── experiments/
│       ├── run_experiment.py           # Main experiment runner
│       ├── plot_results.py             # Figure generation
│       └── cfgs/
│           └── experiment_2d.yaml      # Experiment configuration
└── mpd/                                # Core library
    ├── interactive/                    # GUI editors
    ├── plotting/sdedit_plots.py       # Visualization functions
    └── metrics/sdedit_metrics.py      # Evaluation metrics
```

## Academic Integrity Note

This report is for a Master's Computer Science final project. Ensure that:
- All code is your own work or properly attributed
- Experimental results are genuine and reproducible
- Citations are complete and accurate
- The work represents your individual contribution

## Contact

For questions about the implementation or experiments, see the main `README.md` or open an issue at:
https://github.com/dannybroyd/MPEdit/issues
