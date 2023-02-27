# Machine learning for non-invasive sensing of hypoglycemia while driving in people with diabetes

This repository contains source code to reproduce the results and display items for the following journal article:
> Lehmann, V., Zueger, T., Maritsch, M., Kraus, M., Albrecht, C., Bérubé, C., Feuerriegel, S., Wortmann, F., Kowatsch, T., Styger, N., Lagger, S., Laimer, M., Fleisch, E. and Stettler, C. (2023), Machine learning for non-invasive sensing of hypoglycemia while driving in people with diabetes. Diabetes Obes Metab. https://doi.org/10.1111/dom.15021

## Article Abstract
### Aims
Hypoglycemia is one of the most dangerous acute complications of diabetes mellitus and is associated with an increased risk of driving mishaps. Current approaches to detect hypoglycemia are limited by invasiveness, availability, costs, and technical restrictions. In this work, we developed and evaluated the concept of a non-invasive machine learning (ML) approach detecting hypoglycemia based exclusively on combined driving (CAN) and eye tracking (ET) data.

### Materials and Methods
We first developed and tested our ML approach in pronounced hypoglycemia, and, then, we applied it to mild hypoglycemia to evaluate its early warning potential. For this, we conducted two consecutive, interventional studies in individuals with type 1 diabetes mellitus. In study 1 (_n_ = 18), we collected CAN and ET data in a driving simulator during eu- and pronounced hypoglycemia (blood glucose [BG] 2.0 – 2.5 mmol/L). In study 2 (_n_ = 9), we collected CAN and ET data in the same simulator but in eu- and mild hypoglycemia (BG 3.0 – 3.5 mmol/L).

### Results
Here, we show that our ML approach detects pronounced and mild hypoglycemia with high accuracy (area under the receiver operating characteristics curve [AUROC] 0.88 ± 0.10 and 0.83 ± 0.11, respectively).

### Conclusions
Our findings suggest that an ML approach based on CAN and ET data, exclusively, allows for detection of hypoglycemia while driving. This provides a promising concept for alternative and non-invasive detection of hypoglycemia.

## Repository Content
This repo consists of two major parts: (i) Python code to train ML models and make predictions from driving data and (ii) a Jupyter notebook to produce display items from predictions.


> **Prerequisites**: We recommend to use Python 3.8 and to install dependencies via `pip install -U -r requirements.txt`

- `main.py`: main script to train an ML model and make classify of hypoglycemia.
- `plot.ipynb`: Notebook to reproduce display items from stored pickle files.

⚠️ This repository does not include raw study data needed to actually reproduce results.

## Citation
Please cite the following journal article in work that uses any of these resources.
```
@article{HypoglycemiaDetectionWhileDriving,
    author = {Lehmann, Vera and Zueger, Thomas and Maritsch, Martin and Kraus, Mathias and Albrecht, Caroline and Bérubé, Caterina and Feuerriegel, Stefan and Wortmann, Felix and Kowatsch, Tobias and Styger, Naïma and Lagger, Sophie and Laimer, Markus and Fleisch, Elgar and Stettler, Christoph},
    title = {Machine learning for non-invasive sensing of hypoglycemia while driving in people with diabetes},
    journal = {Diabetes, Obesity and Metabolism},
    year = {2023},
    doi = {https://doi.org/10.1111/dom.15021}
}
```
 