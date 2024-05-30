# Deep Neural Network Analysis SuSe2024
## Fairness - (Calibrated) Equal odds with AIF360
This repository is from the 2024 course "Deep Neural Network Analysis" at the university of Osnabrück, held by Lukas Niehaus.
The topic of this repo is Restrict Discrimination and Enhance Fairness in ML models.
We present two postprocessing methods, equalized odds, and calibrated equalized odds, via their implementation in the [AIF360 toolkit](https://github.com/Trusted-AI/AIF360).

How they work in detail is explained in the presentation pdf found in the course.

# Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/HenningSte/fairness_equal_odds.git
   ```

2. Navigate to the project directory:
   ```bash
   cd fairness_equal_odds
   ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
        
4. Download the model checkpoints folder from [github.com/lucasld/neural_network_analysis](https://github.com/lucasld/neural_network_analysis/tree/main/).

5. Put the folder in the same directory as this repository:

    ```bash
    /parent_folder
        /neural_network_analysis
        /fairness_equal_odds
    ```

6. AIF360 doesn't come with the raw dataset files needed for it's load dataset methods. As described in the notebook ([or here](https://github.com/Trusted-AI/AIF360/blob/main/aif360/data/README.md)), you will have to download the files yourself and place them **in the corresponding folder in your aif360 install in your python environment folder** after setting up your python environment. 

    Using for example a virtual environment and the german dataset, the file structure should look like this:

    ```bash
    \fairness_equal_odds\.venv\Lib\site-packages\aif360\data\raw\german\german.data
    ```

    The files can be found here:
    - [Compas](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv)
    - [German](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
    - [Adult](https://archive.ics.uci.edu/dataset/2/adult)


# Usage
## Repository structure
We have implemented both Equal odds and calibrated odds in 2 notebooks for different classification models. In the AIF360 demo notebook we present both methods performance using a simple regression model trained on well suited datasets for fairness analysis, such as german credit score data or the COMPAS Recidivism Risk Scores dataset. 

In the second notebook we present the same methods for the wine quality dataset and model from this course, found here [repository](https://github.com/lucasld/neural_network_analysis/tree/main/). Here, using fairness enhancement methods and comparing the outcomes makes less sense (even though both methods still perform rather well), but it serves more as a demonstration for how to integrate your own data and models with AIF360.

## Helper functions
Aditionally we provide a number of helper functions and handler classes for both methods in the `utils.py` and `eop.py` & `ceop.py` files. These might be useful should you want to use either method for your own use case.

# Contact
Henning Stegemann <henstegemann@uni-osnabrueck.de>  
Imogen Hüsing <ihuesing@uni-osnabrueck.de>

# References
https://github.com/Trusted-AI/AIF360

https://github.com/lucasld/neural_network_analysis/tree/main/

https://github.com/madammann/DNNA-Blackbox-Interpretability---LIME/