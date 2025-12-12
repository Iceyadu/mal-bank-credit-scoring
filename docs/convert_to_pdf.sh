#!/bin/bash
# Script to convert credit_risk_summary.md to PDF
# Requires pandoc and a LaTeX distribution

echo "Converting credit_risk_summary.md to PDF..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed."
    echo "Install with: brew install pandoc (macOS) or apt-get install pandoc (Linux)"
    exit 1
fi

# Convert markdown to PDF
pandoc docs/credit_risk_summary.md \
    -o docs/credit_risk_summary.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    --toc-depth=1 \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue

if [ $? -eq 0 ]; then
    echo "✓ PDF created successfully: docs/credit_risk_summary.pdf"
else
    echo "✗ Conversion failed. Trying alternative method..."
    
    # Alternative: Use wkhtmltopdf if available
    if command -v wkhtmltopdf &> /dev/null; then
        pandoc docs/credit_risk_summary.md -o docs/credit_risk_summary.html
        wkhtmltopdf docs/credit_risk_summary.html docs/credit_risk_summary.pdf
        rm docs/credit_risk_summary.html
        echo "✓ PDF created using wkhtmltopdf: docs/credit_risk_summary.pdf"
    else
        echo "Please install pandoc and a LaTeX distribution (e.g., MacTeX, TeX Live)"
        echo "Or use one of the alternative methods in the README"
    fi
fi

