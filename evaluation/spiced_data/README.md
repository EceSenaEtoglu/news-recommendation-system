# SPICED Dataset

This directory contains the SPICED (Similarity Detection in News Articles) dataset and related utilities.

## Files

- `spiced.csv` - The main SPICED dataset with 977 article pairs
- `spiced_loader.py` - Python utilities for loading and processing SPICED data
- `requirements.txt` - Python dependencies for SPICED processing
- `README.md` - This documentation file

## Dataset Description

The SPICED dataset contains news article pairs from 7 different topics:
- Politics
- Crimes  
- Disasters
- Science
- Economics
- Sports
- Culture

Each pair includes:
- `text_1` and `text_2` - The article texts
- `URL_1` and `URL_2` - Source URLs
- `Type` - Topic category

## Usage

```python
from evaluation.spiced_data.spiced_loader import load_combined, load_intertopic, load_intratopic_and_hard_examples

# Load combined dataset
train_data, test_data = load_combined('train'), load_combined('test')

# Load intertopic pairs
intertopic_train = load_intertopic('train')
intertopic_test = load_intertopic('test')

# Load intratopic pairs and hard examples
intratopic_train, hard_train = load_intratopic_and_hard_examples('train')
intratopic_test, hard_test = load_intratopic_and_hard_examples('test')
```

## Citation

If you use this dataset, please cite the original SPICED paper:

```
@inproceedings{spiced2024,
  title={SPICED: Similarity Detection in News Articles},
  author={Authors},
  booktitle={Proceedings of LREC},
  year={2024}
}
```