# GISR
This repository contains code for Graph-Informed Symbolic Regression: a framework that jointly trains a GNN-based classifier to detect clusters within a hierarchical dataset and performs symbolic regression on the predicted clusters. 

## Installing dependencies

Create a conda environment using `conda create --name <env> --file requirements.txt`

## (Optional) Drug interaction data pre-processing

If you would like to run GISR on the graph-structured data ensemble constructed from [DrugComb repository](https://drugcomb.fimm.fi) and [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/), follow the steps below.

### Step 1: Install Reservoir data lake

We use the Reservoir data lake to retrieve dose-response data from DrugComb. Follow the Reservoir [installation instructions](https://github.com/RECOVERcoalition/Reservoir/tree/main).

Make sure that the `reservoir` folder is in the GISR directory.

### Step 2: Data pre-processing

Run `preprocessing.py` to load the DrugComb and PrimeKG dataset. Below are the arguments you can pass to the script along with their descriptions:

#### Arguments:

1. **`--data_dir`**: Path to directory for loading and saving data. 
   - **Type**: `str`
   - **Note**: This argument is mandatory. 

2. **`--combo_measurements`**: Retrieve all experiment blocks that have this many combo measurements.
   - **Type**: `int`
   - **Default**: `9`

3. **`--mono_row_measurements`**: Retrieve all experiment blocks that have this many mono measurements for the row drug.
   - **Type**: `int`
   - **Default**: `4`

4. **`--mono_col_measurements`**: Retrieve all experiment blocks that have this many mono measurements for the column drug.
   - **Type**: `int`
   - **Default**: `4`

5. **`--cell_line`**: Retrieve all experiment blocks from this cell line.
   - **Type**: `str`
   - **Default**: `None` (If not specified, all cell lines will be considered.)

6. **`--study`**: Retrieve all experiment blocks from this study.
   - **Type**: `str`
   - **Default**: `None` (If not specified, all studies will be considered.)


## Running GISR

Run `train.py` to run GISR. Below are the arguments you can pass to the script along with their descriptions:

### Arguments:

1. **`--data_dir`**: Path to directory for loading and saving data. 
   - **Type**: `str`
   - **Default**: `data/`

2. **`--k`**: Number of clusters to search for.
   - **Type**: `int`
   - **Default**: `3`

3. **`--max_iters`**: Maximum number of GISR training iterations.
   - **Type**: `int`
   - **Default**: `100`

4. **`--classif_epochs`**: Number of classifier training epochs per iteration.
   - **Type**: `int`
   - **Default**: `100`

5. **`--learning_rate`**: Initial learning rate for classifier training.
   - **Type**: `float`
   - **Default**: `0.001`

6. **`--hidden_dim`**: Hidden layer dimension for classifier.
   - **Type**: `int`
   - **Default**: `32`

7. **`--dataset`**: Name of dataset.
   - **Type**: `str`
   - **Choices**: `synthetic`, `real`
   - **Default**: `synthetic`

8. **`--cluster_init`**: Type of label initialization.
   - **Type**: `str`
   - **Choices**: `random`, `similar`
   - **Default**: `random`

9. **`--noise_std`**: Standard deviation of added Gaussian noise to the synthetic dataset.
   - **Type**: `float`
   - **Default**: `0.0`

10. **`--test_size`**: Portion of dataset that is in the test set.
   - **Type**: `float`
   - **Default**: `0.1`
