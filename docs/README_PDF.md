# Generating credit_risk_summary.pdf

## Quick Methods

### Method 1: Pandoc (Command Line) - Recommended

```bash
# Install pandoc (if needed)
# macOS: brew install pandoc basictex
# Linux: apt-get install pandoc texlive-latex-base

# Convert
cd /Users/mac/Documents/DigitalBank/docs
pandoc credit_risk_summary.md -o credit_risk_summary.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article
```

### Method 2: VS Code / Cursor Extension

1. Install "Markdown PDF" extension
2. Open `docs/credit_risk_summary.md`
3. Right-click → "Markdown PDF: Export (pdf)"
4. PDF will be saved in same directory

### Method 3: Online Converter

1. Copy content from `credit_risk_summary.md`
2. Use online tool:
   - https://www.markdowntopdf.com/
   - https://dillinger.io/ (export as PDF)
   - https://www.markdowntopdf.com/

### Method 4: Python Script

```python
# Install: pip install markdown pdfkit
import markdown
import pdfkit

with open('credit_risk_summary.md', 'r') as f:
    md_content = f.read()

html = markdown.markdown(md_content)
pdfkit.from_string(html, 'credit_risk_summary.pdf')
```

### Method 5: Google Docs / Word

1. Copy markdown content
2. Paste into Google Docs or Word
3. Format as needed
4. Export/Print to PDF

## File Location

The generated PDF should be saved as:
```
docs/credit_risk_summary.pdf
```

## Verification

After generation, verify:
- ✓ File exists: `docs/credit_risk_summary.pdf`
- ✓ File size: ~50-200 KB (typical for 1-2 pages)
- ✓ All sections present (A through F)
- ✓ Formatting looks professional

