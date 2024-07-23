# Problematic Subgroup Identification on Text

### Explainable and Trustworthy AI Course
### Politecnico di Torino - 2023/2024
**Reference Teachers:** Eliana Pastor, Salvatore Greco

---

## Overview

This research project aims to propose new methods for identifying and mitigating disadvantaged subgroups in text classifiers. Identifying subgroups where a model performs differently from its overall behavior enables understanding at the subgroup level and investigates the fairness and robustness of the model. Existing solutions primarily focus on tabular and speech data, with limited work addressing textual data. This project aims to fill that gap by exploring interpretable representations of textual data and leveraging existing approaches for identifying problematic subgroups.

## Goal

The primary goal of this project is to identify interpretable representations of textual data to enable the understanding and evaluation of model performance at the subgroup level. This involves:
- Using Large Language Model (LLM) prompts to derive categories similar to concept-based approaches.
- Defining a hierarchy of categories to further enhance understanding.
- Leveraging existing techniques for problematic subgroup identification.


## Required Analysis, Implementation, and Evaluation

### Literature Review
Conduct a systematic review of existing Natural Language Processing (NLP) fairness metrics and techniques for identifying and evaluating subgroups in textual data.

### Identification of Research Gaps
Identify key research gaps for evaluating subgroup performance and, optionally, for mitigating performance disparities.

### Implementation
1. Select a specific research gap to address.
2. Propose and implement a methodology to identify and evaluate subgroup performance. This may involve:
    - Using LLM-prompting in zero or few-shot learning for annotating metadata in texts.
    - Deriving categories (e.g., concept-based approaches).
3. Propose or leverage existing techniques to mitigate subgroup performance disparities.

### Evaluation
Assess the effectiveness and applicability of the newly implemented approach using at least two datasets.

## Usage

### Running the Program

To run the program, use the following command:

```bash
python main.py --dataset_type <dataset_type> --debug <debug_mode> --percentage <percentage>
```

Arguments

    --dataset_type: Type of the dataset to load. Options are emotion or sarcasm. Default is emotion.
    --debug: Enable debug mode to print additional information. Options are True or False. Default is False.
    --percentage: Percentage of the dataset to use. For example, use 0.1 for 10% of the dataset. Default is 100.0.

## References

1. Yeounoh Chung et al. “Slice finder: Automated data slicing for model validation”. In: 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE. 2019, pp. 1550–1553.
2. Alkis Koudounas et al. “Exploring subgroup performance in end-to-end speech models”. In: ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE. 2023, pp. 1–5.
3. Eliana Pastor, Luca De Alfaro, and Elena Baralis. “Looking for trouble: Analyzing classifier behavior via pattern divergence”. In: Proceedings of the 2021 International Conference on Management of Data. 2021, pp. 1400–1412.
4. Svetlana Sagadeeva and Matthias Boehm. “Sliceline: Fast, linear-algebra-based slice finding for ML model debugging”. In: Proceedings of the 2021 International Conference on Management of Data. 2021, pp. 2290–2299.
5. Nima Shahbazi et al. “Representation bias in data: a survey on identification and resolution techniques”. In: ACM Computing Surveys 55.13s (2023), pp. 1–39.
6. Tony Sun et al. “Mitigating gender bias in natural language processing: Literature review”. In: arXiv preprint arXiv:1906.08976 (2019).

---