#!/bin/bash
# Compile the MPEdit LaTeX report

set -e  # Exit on error

echo "============================================"
echo "Compiling MPEdit Final Project Report"
echo "============================================"
echo ""

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found!"
    echo ""
    echo "Please install a LaTeX distribution:"
    echo "  - Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  - macOS: Install MacTeX from https://www.tug.org/mactex/"
    echo "  - Windows: Install MiKTeX from https://miktex.org/"
    echo ""
    exit 1
fi

# Check if bibtex is installed
if ! command -v bibtex &> /dev/null; then
    echo "ERROR: bibtex not found!"
    echo "Please install a complete LaTeX distribution."
    exit 1
fi

# Check if required files exist
if [ ! -f "report.tex" ]; then
    echo "ERROR: report.tex not found!"
    exit 1
fi

if [ ! -f "references.bib" ]; then
    echo "ERROR: references.bib not found!"
    exit 1
fi

if [ ! -f "scripts/inference/logs_experiments/figures/table1_baselines.tex" ]; then
    echo "WARNING: Experimental table not found at scripts/inference/logs_experiments/figures/table1_baselines.tex"
    echo "Some content may be missing from the report."
fi

echo "Step 1/4: Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1 || {
    echo "ERROR: First pdflatex pass failed!"
    echo "Check report.log for details"
    exit 1
}

echo "Step 2/4: Running bibtex..."
bibtex report > /dev/null 2>&1 || {
    echo "WARNING: bibtex encountered issues (this is often normal)"
}

echo "Step 3/4: Running pdflatex (second pass)..."
pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1 || {
    echo "ERROR: Second pdflatex pass failed!"
    echo "Check report.log for details"
    exit 1
}

echo "Step 4/4: Running pdflatex (third pass)..."
pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1 || {
    echo "ERROR: Third pdflatex pass failed!"
    echo "Check report.log for details"
    exit 1
}

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f report.aux report.log report.out report.bbl report.blg report.toc

echo ""
echo "============================================"
echo "SUCCESS! Report compiled to report.pdf"
echo "============================================"
echo ""
echo "Page count:"
pdfinfo report.pdf 2>/dev/null | grep "Pages:" || echo "  (pdfinfo not available - install poppler-utils to see page count)"
echo ""
echo "File size:"
ls -lh report.pdf | awk '{print "  " $5}'
echo ""
echo "To view the report:"
echo "  - Linux: xdg-open report.pdf"
echo "  - macOS: open report.pdf"
echo "  - Windows: start report.pdf"
echo ""
