# Preparation

- Install clang (preferred) or gcc

- Set up and activate a fresh Python virtual environment (Python >= 3.7 should work)

- Install the required packages listed in `requirements.txt`, either manually or by running

        pip install -r requirements.txt


- If you want to use pycombina for comparison, install the dependencies listed at https://pycombina.readthedocs.io/en/latest/install.html#install-on-ubuntu-18-04-debian-10, then clone and build pycombina by running:


        git clone https://github.com/adbuerger/pycombina.git
        cd pycombina
        git submodule init
        git submodule update
        python setup.py install 


- Create the system and NLP files and libraries by running

        python nlpsetup.py


# Solving the MINLP

- `solve_minlp.py` solves the problem using GN-MIQP and CIA for comparison
- `solve_minlp_voronoi.py` shows how the Voronoi cuts can enter the setup, using the (yet) dummy-class from voronoi.py

# Plotting

The Jupyter notebook `plot_results.ipynb` can produce some figures from the saved pickle files.
