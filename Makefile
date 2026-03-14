# Makefile for MPEdit Final Project Report

.PHONY: all clean view help

# Default target
all: report.pdf

# Main compilation target
report.pdf: report.tex references.bib
	@echo "Compiling LaTeX report..."
	@pdflatex -interaction=nonstopmode report.tex > /dev/null
	@bibtex report > /dev/null 2>&1 || true
	@pdflatex -interaction=nonstopmode report.tex > /dev/null
	@pdflatex -interaction=nonstopmode report.tex > /dev/null
	@echo "Success! Generated report.pdf"

# Quick compilation (single pass, no bibtex)
quick: report.tex
	@echo "Quick compilation (single pass)..."
	@pdflatex -interaction=nonstopmode report.tex > /dev/null
	@echo "Success! Generated report.pdf (references may be incomplete)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -f report.aux report.log report.out report.bbl report.blg report.toc report.pdf
	@echo "Done."

# Clean everything including PDF
cleanall: clean
	@rm -f report.pdf

# View the PDF (platform-specific)
view: report.pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open report.pdf; \
	elif command -v open > /dev/null; then \
		open report.pdf; \
	else \
		echo "Please open report.pdf manually"; \
	fi

# Show word count estimate (for main body, excluding appendix)
wordcount: report.tex
	@echo "Estimated word count (main body only, excluding appendix):"
	@sed -n '1,/\\clearpage/p' report.tex | \
	    grep -v '^%' | \
	    sed 's/\\[a-zA-Z]*{//g' | \
	    sed 's/}//g' | \
	    wc -w
	@echo "(Note: This is approximate and includes LaTeX commands)"

# Check if figures exist
check-figures:
	@echo "Checking for experimental figures..."
	@for fig in fig2_t0_sweep.pdf fig_baselines_comparison.pdf fig4_sketch_heatmap.pdf fig5_speed.pdf fig6_diversity.pdf table1_baselines.tex; do \
	    if [ -f "scripts/inference/logs_experiments/figures/$$fig" ]; then \
	        echo "  ✓ $$fig"; \
	    else \
	        echo "  ✗ $$fig (MISSING)"; \
	    fi \
	done

# Help message
help:
	@echo "MPEdit LaTeX Report Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Compile the full report with bibliography"
	@echo "  make quick        - Quick compilation (single pass, no bibtex)"
	@echo "  make clean        - Remove auxiliary files (keep PDF)"
	@echo "  make cleanall     - Remove all generated files including PDF"
	@echo "  make view         - Open the PDF in the default viewer"
	@echo "  make wordcount    - Estimate word count (main body only)"
	@echo "  make check-figures - Check if all experimental figures exist"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (texlive or similar LaTeX distribution)"
	@echo "  - bibtex (usually included with LaTeX)"
	@echo ""
	@echo "To install LaTeX:"
	@echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
	@echo "  macOS:         Install MacTeX from https://www.tug.org/mactex/"
	@echo "  Windows:       Install MiKTeX from https://miktex.org/"
	@echo ""
