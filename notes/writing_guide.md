# ICML 2025 Paper Writing Guide

## File Structure

```
ICML2025_Template/
├── paper.tex          <- ICML wrapper (title, authors, abstract)
├── paper_body.md      <- Your content in Markdown (Introduction onward)
├── paper_body.tex     <- Generated from paper_body.md (don't edit directly)
├── paper.bib          <- Your references (BibTeX format)
├── figures/           <- Your figures (.pdf for vector, .png for raster)
└── [template files]   <- icml2025.sty, icml2025.bst, etc.
```

---

## Step-by-Step Workflow

### Step 1: Write Content in Markdown

Edit `paper_body.md` with your paper content starting from Introduction.

**Headings** - No manual numbering (LaTeX numbers automatically):
```markdown
## Introduction        (correct)
## 1 Introduction      (wrong - will double-number)
```

**Math** - Use LaTeX syntax:
```markdown
Inline: $a^2 + b^2 = c^2$

Display:
\[
\mathcal{L}(\theta) = \mathbb{E}_{(x,y)} \left[ \ell(f_\theta(x), y) \right]
\]
```

**Citations** - Use pandoc syntax:
```markdown
[@Smith2020]                    -> (Smith, 2020)
[@Smith2020; @Jones2021]        -> (Smith, 2020; Jones, 2021)
As shown by @Smith2020, ...     -> As shown by Smith (2020), ...
```

**Figures**:
```markdown
![Caption text here.](figures/my_figure.pdf){width=90%}
```

**Tables** (simple - no caption in markdown, add in LaTeX if needed):
```markdown
| Method | Accuracy | F1 Score |
|--------|----------|----------|
| Baseline | 0.85 | 0.82 |
| **Ours** | **0.92** | **0.89** |
```

*Note: For complex tables with captions, write directly in LaTeX in paper_body.tex after conversion.*

---

### Step 2: Edit Metadata in paper.tex

Open `paper.tex` and customize:

1. **Title** (line ~60):
   ```latex
   \icmltitle{Your Paper Title Here}
   ```

2. **Running title** (line ~52):
   ```latex
   \icmltitlerunning{Short Running Title}
   ```

3. **Authors** (lines ~67-75):
   ```latex
   \begin{icmlauthorlist}
   \icmlauthor{Your Name}{equal,aff1}
   \end{icmlauthorlist}

   \icmlaffiliation{aff1}{Department, University, City, Country}
   \icmlcorrespondingauthor{Your Name}{email@domain.edu}
   ```

4. **Abstract** (lines ~86-89):
   ```latex
   \begin{abstract}
   Your abstract here. Single paragraph, 4-6 sentences.
   \end{abstract}
   ```

5. **Keywords** (line ~77):
   ```latex
   \icmlkeywords{Machine Learning, ICML, keyword1, keyword2}
   ```

---

### Step 3: Add References to paper.bib

```bibtex
@inproceedings{Smith2020,
  author    = {Smith, John and Doe, Jane},
  title     = {Paper Title with {Important} {Words}},
  booktitle = {Proceedings of ICML},
  year      = {2020},
  pages     = {1--10},
}
```

**Tips:**
- Protect capitals with braces: `{B}ayesian`, `{LSTM}`, `{GPT}`
- Include page numbers when available
- Use consistent author name formatting

---

### Step 4: Convert Markdown to LaTeX

```bash
pandoc paper_body.md \
  --from markdown \
  --to latex \
  --natbib \
  --wrap=none \
  -o paper_body.tex
```

---

### Step 5: Compile to PDF

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or as a single command:
```bash
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## Submission Checklist

### For Anonymous Submission (Double-Blind)

- [ ] `\usepackage{icml2025}` (no `[accepted]` option)
- [ ] No author names visible in PDF
- [ ] Acknowledgements section commented out
- [ ] No identifying information in text
- [ ] Self-citations in third person
- [ ] Main body <= 8 pages (excluding references/appendix)
- [ ] Total file size < 10MB

### For Camera-Ready (After Acceptance)

- [ ] `\usepackage[accepted]{icml2025}`
- [ ] Real author names and affiliations
- [ ] Acknowledgements section included
- [ ] Main body <= 9 pages
- [ ] Code/data URLs included (if applicable)

---

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| Citations not showing | Run `bibtex paper` then `pdflatex` twice |
| Double section numbers | Remove manual numbers from Markdown headings |
| Build breaks with hyperref | Use `\usepackage[nohyperref]{icml2025}` |
| Figures not found | Check path is relative to paper.tex location |
| Type-3 font errors | Use `pdflatex` instead of `latex` + `dvips` |

---

## Quick Reference Commands

```bash
# Navigate to template folder
cd "/path/to/ICML2025_Template"

# Convert markdown to latex
pandoc paper_body.md --from markdown --to latex --natbib --wrap=none -o paper_body.tex

# Full build
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

# Clean auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc
```
