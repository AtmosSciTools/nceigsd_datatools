# nceigsd

A Python library for downloading and processing Global Summary of the Day (GSD) data from NCEI-NOAA.

## Installation

```bash
pip install .
```

## Usage

```python
from nceigsd import NCEIGSDProcessor

processor = NCEIGSDProcessor(
    start_year=2010,
    end_year=2012,
    area=[18, 20, 105, 107],
    output_dir="output/test_output"
)
processor.run()
```