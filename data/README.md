# Data

This folder contains sample data and instructions for obtaining the datasets used in the paper.

## Cancer signaling network

The cancer signaling network used in the paper was constructed from:

1. **Cancer Gene Census** (https://cancer.sanger.ac.uk/census)
   - Curated list of cancer-associated genes
   - Contains 379 proteins after filtering

2. **STRING database** (https://string-db.org)
   - Protein-protein interaction scores
   - Filtered for high-confidence interactions (score > 700)
   - Results in 3,498 interactions

## Downloading the data

### Option 1: Use provided scripts

```bash
# Download and process network data
python scripts/download_data.py
```

### Option 2: Manual download

1. Download cancer genes from Cancer Gene Census
2. Query STRING API for interactions:
   ```python
   import requests

   proteins = ["TP53", "BRCA1", "EGFR", ...]  # Your protein list
   url = "https://string-db.org/api/tsv/network"
   params = {
       "identifiers": "%0d".join(proteins),
       "species": 9606,  # Human
       "required_score": 700
   }
   response = requests.get(url, params=params)
   ```

3. Save as edgelist format:
   ```
   PROTEIN1 PROTEIN2
   TP53 MDM2
   BRCA1 BARD1
   ...
   ```

## Data format

### `protein_network.edgelist`
```
PROTEIN1 PROTEIN2
TP53 MDM2
BRCA1 BARD1
...
```

### `protein_annotations.csv`
```csv
protein,category,category_id
TP53,Cell cycle,0
BRCA1,DNA repair,2
...
```

## PubMed abstracts

Retrieved from NCBI E-utilities API:
- https://www.ncbi.nlm.nih.gov/books/NBK25501/

Example query:
```python
from Bio import Entrez

Entrez.email = "your@email.com"
handle = Entrez.esearch(db="pubmed", term="TP53 cancer")
```

## License

- STRING data: CC BY 4.0
- Cancer Gene Census: Academic use only
- PubMed abstracts: Public domain
