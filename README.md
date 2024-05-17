# OneDiff Speedup Test

This repository contains a simple test to measure the speedup provided by the OneDiff algorithm.

## How to Use

Follow these steps to set up and run the test:

1. **Set up a Python 3.10 Conda environment**

   Create a new Conda environment named `onediff-test` with Python 3.10:

   ```bash
   conda env create -n onediff-test python=3.10

   ```

2. **Activate the environment**

   Switch to the new environment:

   ```bash
   conda activate onediff-test

   ```

3. **Install the OneDiff library**

   Run the provided script to install the OneDiff library with CUDA 11.2 support:

   ```bash
   bash install_one_diff_cuda121.bash
   ```

   This script follows the [official OneDiff installation instructions](https://github.com/siliconflow/onediff#1-install-oneflow).

4. **Verify the OneDiff installation**

   Verify that the OneDiff library is successfully installed by importing it in Python:

   ```bash
   python -c "import onediff"
   ```

5. **Run the test**

   Execute the test script:

   ```bash
   python onediff_test.py
   ```

   This script uses the OneDiff library to perform text-to-image inference using a pre-trained model.
# onediff_test
