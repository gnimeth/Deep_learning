
#  Overview

- The nonprofit organization Alphabet Soup requires a tool that can aid in selecting the most promising funding applicants who have the highest likelihood of succeeding in their ventures. Utilizing machine learning and neural networks, a binary classifier will be created using the features present in the given dataset to predict whether funding recipients will attain success if financed by Alphabet Soup.

![App Screenshot](https://www.stockvault.net/data/2009/05/08/108803/preview16.jpg)
# Results

### Data Preprocessing

- The target variable for the model is "IS_SUCCESSFUL", which is a binary variable indicating whether or not the fundraising campaign was successful.

- The features for the model are:

  - EIN: Employer Identification Number, a unique identifier for organizations.
  - NAME: Name of the organization.
  - APPLICATION_TYPE: Type of application for funding.
  - AFFILIATION: Indicates if the organization is independent or associated with a larger organization.
  - CLASSIFICATION: IRS code that indicates the type of organization.
  - USE_CASE: Use case for funding.
  - ORGANIZATION: Type of organization.
  - STATUS: Current status of the organization.
  - INCOME_AMT: Income amount of the organization.
  - SPECIAL_CONSIDERATIONS: Indicates if the organization has any special considerations for funding.
  - ASK_AMT: Amount of funding requested.

- The variable that should be removed from the input data is the EIN (Employer Identification Number) because it is not relevant for predicting the success of a fundraising campaign. The organization's unique identifier does not have any impact on the success of a fundraising campaign.

### Compiling, Training, and Evaluating the Model

- In the first two neural network models, the architecture consists of three hidden layers with 20 units each and the activation function is leaky ReLU. The output layer is a single unit with a sigmoid activation function, which is appropriate for a binary classification problem like the one presented. The input dimension is 36.  The third neural network model has a different architecture, with two hidden layers and 22 units each, also using the leaky ReLU activation function. 

- The highest accuracy achieved among the three neural network models is 72.59%, which is below the target model performance of 75%. Therefore, the target model performance was not achieved with these models.

- An attempted optimization approach for the model was to adjust the bin sizes of the continuous variables. This approach involved changing the binning some of the continuous variables. However, this did not result in a substantial improvement in the model's performance.  Another approach was to drop more columns from the input data that were deemed to be less relevant or redundant. The "EIN" and "NAME" columns were already removed, but there were other columns that could be dropped as well. For example, the "SPECIAL_CONSIDERATIONS" column contained mostly "N" values, so it was dropped from the input data but did not help with improving the model's performance.

# Summary

- This report describes the development and optimization of a binary classifier using machine learning and neural networks to aid in selecting the most promising funding applicants for the nonprofit organization Alphabet Soup. The input data consists of 36 features, and three neural network models were built with different architectures using the leaky ReLU activation function. However, the highest accuracy achieved among the models was 72.59%, which is below the target model performance of 75%.

- To optimize the model, bin sizes of the continuous variables were adjusted, but this approach did not result in a significant improvement in model performance. Another approach was to drop more columns from the input data, such as the "SPECIAL_CONSIDERATIONS" column, which contained mostly "N" values. However, this did not help with improving the model's performance. Further optimization techniques may be necessary to achieve the desired model performance.



## Dependencies and Setup

```bash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
```


## Resources
- IRS. Tax Exempt Organization Search Bulk Data Downloads. - [IRS.gov](https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads)

