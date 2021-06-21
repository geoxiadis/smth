# Create an environemnt with the requirements
conda env create -f ~/Desktop/aer_surr/env.yml
conda activate aeroelastic_surrogate

# Example of loading and evaluating one of the surrogates
cd ~/Desktop/aer_surr/
python example.py