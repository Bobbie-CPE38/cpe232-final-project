# Predictive Modeling of Ridesharing Prices in Boston


## Set Up

### 1. Create virtual environment
```bash
python -m venv .venv
```

Activate:
- Window
```bash
.venv\Scripts\activate
```

- Mac/Linux
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Choose Kernal (VS Code)
Go to .ipynb file > Look at top right > Select Another Kernel > Python Environments > .venv


## Running Pipelines
```bash
python -m src.pipelines.<pipeline_name>
```

Example src/pipeline/data_prep.py:
```bash
python -m src.pipelines.data_prep
```