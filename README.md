# Project overview

  Many mathematical frameworks such as WolframAlpha provide a tool to conduct symbolic calculation of mathematical expressions. For example, such a tool can take a mathematical expression as input and output the differentiated or integrated forms. The original paper by Lample and Charton, ,  shows that an RNN model can also achieve this task of symbolic function integration and differentiation, which is analogous to building a translation machine for the mathematical language.
  
  This project aims to replicate the integration task of the original paper and uses the datasets published by Facebook research available on the github page. The dataset contains the series of functions expressed as symbols and operations (e.g. “a pow 2 mult b add c” to express a^2*b+c) as input and the final output functions after the integration.
    
    The directory contains three key files:
    1. The final report: projectSummaryReport.pdf
    2. Solved jupyter notebook: model_train_test_notebook_fin.ipynb
    3. Main model layer: model.py
    4. Encoder/Decoder/Attention layer: layers.py
    5. Model experiment class: run_model_train.py
    6. User defined hyperparameters to test during model experimentation: model_params.py

## Links to the model and data:

* Best model saved: https://drive.google.com/drive/folders/1dGIF2mrqy7CGFfQ-9BkiUzhH6VZPWfSd?usp=share_link
* Dataset saved: https://drive.google.com/drive/folders/1RFoR_2947tPvuVhW5eblxAVqXSo9ie6K?usp=share_link


## Note

Please note the final model training was done on google colab with GPU. Users can reset the path to the dataset when running the jupyter notebook.

# Organization of this directory

data and models folder not included because of the size; the links to the data and the models are included in the final report (at the end of the report, section 8)
```
.
├── .gitignore
├── projectSummaryReport.pdf
├── README.md
├── layers.py
├── model.py
├── model_params.py
├── model_train_test_notebook_fin.ipynb
└── run_model_train.py

0 directories, 8 files
```
