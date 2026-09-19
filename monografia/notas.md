# Editorial Consistency Checklist

**1. Visuals and Tabular Formatting**

* **Color Mapping:** Enforce invariant color assignments for all models and data samples across all plots and scatter plots.
* **Ordinal Sequences:** Maintain a fixed, consistent ordering for models and datasets in all tables.
* **Layout Elements:** Standardize the formatting and caption placement (e.g., above tables, below figures) for all tables, figures, algorithms, and code listings.

**2. LaTeX and Typographical Conventions**

* **Font Macros:** Systematize the application of `\textbf{}` and `\texttt{}` for packages, datasets, programming languages, and model names.
* **Cross-References:** Standardize label prefixes (e.g., `sec:`, `subsec:`, `fig:`).
* **Cleveref Usage:** Apply `\Cref{}` exclusively for sentence-initial references and `\cref{}` elsewhere. Remove redundant nouns preceding references (e.g., prefer `\cref{eq:x}` over `the equation \cref{eq:x}`).

**3. Nomenclature and Textual Style**

* **Codebase Alignment:** Ensure exact capitalization and spelling in the text mirror the source code (e.g., `MicroTEDAClus`).
* **Dataset Nomenclature:** Standardize the naming conventions for all datasets.
* **Numeric Representation:** Enforce a strict rule for numbers (e.g., spelling out integers strictly less than 10, using numerals for values $\geq 10$ and for all mathematical or parametric values).
* **Voice and Perspective:** Consistently utilize the active voice and the first-person plural (e.g., "we evaluated", "we propose").

**4. Repository Coherence**

* **Structural Synchronization:** Ensure complete terminological and parametric parity between the experimental framework and the compiled manuscript artifacts.

**5. Refinement: Universal Macro Application**

* **Macro Precedence:** Predefined named LaTeX macros must be exclusively utilized for the entity names (models, datasets) throughout the entire document body.
* **Historical and Source Consistency:** Macros must be used even when discussing historical usage, literature reviews, or referencing the original paper. Hardcoded textual representations of these entities are discouraged to prevent terminological drift.
