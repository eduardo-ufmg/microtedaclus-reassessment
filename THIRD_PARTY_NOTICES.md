# Third-party material

The licenses of this repository (`LICENSE` and `monografia/LICENSE.md`) do not cover the
material below, which keeps the terms of its authors.

## Code derived from evolving_clustering

The original implementation of MicroTEDAClus, by Carlos Severiano, in
`src/models/evolving/EvolvingClustering.py` of https://github.com/cseveriano/evolving_clustering,
is the base of:

- `experimentos/proposed_methodology/models/MicroTEDAClus/MicroTEDAClus.py`;
- `experimentos/preprocessing/MicroTEDAClus/MicroTEDAClus.py`;
- `experimentos/multiple_runs/MicroTEDAClus/MicroTEDAClus.py`.

Its license:

```
The MIT License (MIT)
Copyright (c) 2018, Carlos Severiano

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Datasets

`experimentos/modern_methodology/data/cluto-t7-10k.arff` and `cluto-t8-8k.arff` are the datasets
DS3 and DS4 of the evaluation of Chameleon (Karypis, Han and Kumar, 1999), as distributed by
https://github.com/deric/clustering-benchmark, from which the other experiments download them. The
other streams are generated or downloaded when the experiments run.

## Figures of other works

On the branch `main`, `monografia/MethodologyReassessment/figures/original_curves.png` reproduces
the published curves of Maia et al. (2020), which the thesis credits in its caption.
