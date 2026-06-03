# Thesis — Detection of Externally-Influenced Routing via Inverse Markov-Chain Methods

Master's project report for the MEng Design Engineering programme at Imperial College London.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Top-level LaTeX document. Includes the section files in order. |
| `sections/` | Section source files. `main.tex` is the source of truth for the compiled order; older background/methodology drafts remain in the folder but are not included in the submitted build. |
| `references.bib` | BibTeX bibliography in IEEEtran numeric style. |
| `figures/` | Folder for any figures referenced by the sections (currently a stub). |
| `.vscode/settings.json` | LaTeX Workshop build recipe for VS Code. |

## How to build (VS Code)

1. Install the **LaTeX Workshop** extension by James Yu.
2. Install a LaTeX distribution that provides `pdflatex` and `bibtex`. On macOS the easiest path is MacTeX (`brew install --cask mactex`) or BasicTeX (smaller, then `tlmgr install <pkg>` as needed).
3. Open this `thesis/` folder in VS Code (`File → Open Folder…` and pick `thesis/`). The settings file in `.vscode/` will load automatically.
4. Open `main.tex`. Use `Cmd-Shift-P → LaTeX Workshop: Build LaTeX project` (or just save the file with auto-build on).
5. The compiled `main.pdf` will appear next to `main.tex`. To open it inline: `Cmd-Shift-P → LaTeX Workshop: View LaTeX PDF file → in new tab`.

The build recipe is `pdflatex → bibtex → pdflatex → pdflatex` (so citations and cross-references resolve correctly on the second pass).

## How to build from a terminal

```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Word count / page limit

The module limits the main body to 35 pages at 12pt (`Module descriptor and assessment brief | Design-Engineering-Masters-Project.html`). Front matter (abstract, ToC), back matter (references), and appendices do **not** count. The submitted source is set to 12pt in `main.tex`.

Current status (sections shipped, revised through the Labour-Day Xuancheng stress test and 2-step audit):
1. Notation
2. Introduction
3. Tier 1: Synthetic Reproduction
4. Tier 2: Controlled Validation on SUMO
5. Tier 3: Real-World Validation on Xuancheng
6. Discussion
7. Conclusion
8. Project Management and Reflection

Plus appendices covering repository structure, use of AI, build instructions, and a metric/method primer.
