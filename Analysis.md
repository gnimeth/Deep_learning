
#  Overview

- The nonprofit organization Alphabet Soup requires a tool that can aid in selecting the most promising funding applicants who have the highest likelihood of succeeding in their ventures. Utilizing machine learning and neural networks, a binary classifier will be created using the features present in the given dataset to predict whether funding recipients will attain success if financed by Alphabet Soup.

![App Screenshot](https://www.stockvault.net/data/2009/05/08/108803/preview16.jpg)
# Results

### Data Preprocessing
- Load the crypto_market_data.csv into a DataFrame.

- Get the summary statistics and plot the data to see what the data looks like before proceeding.

![App Screenshot](https://raw.githubusercontent.com/gnimeth/Cryptoclustering/main/Outputs/bokeh_plot.png)

- Use the StandardScaler() module from scikit-learn to normalize the data from the CSV file.

### Compiling, Training, and Evaluating the Model

- Create a DataFrame with the scaled data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.

- Use the elbow method to find the best value for k

![App Screenshot](https://raw.githubusercontent.com/gnimeth/Cryptoclustering/main/Outputs/bokeh_plot%20(1).png)

- Cluster Cryptocurrencies with K-means Using the Original Scaled Data

![App Screenshot](https://raw.githubusercontent.com/gnimeth/Cryptoclustering/main/Outputs/bokeh_plot%20(2).png)

- Find the Best Value for k Using the PCA Data

![App Screenshot](https://raw.githubusercontent.com/gnimeth/Cryptoclustering/main/Outputs/bokeh_plot%20(3).png)

- Cluster Cryptocurrencies with K-means Using the PCA Data

![App Screenshot](https://raw.githubusercontent.com/gnimeth/Cryptoclustering/main/Outputs/bokeh_plot%20(4).png)


![App Screenshot](https://raw.githubusercontent.com/gnimeth/Cryptoclustering/main/Outputs/bokeh_plot%20(5).png)

# Summary




## Dependencies and Setup

```bash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
```


## Resources
- IRS. Tax Exempt Organization Search Bulk Data Downloads. - [IRS.gov](https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads)

