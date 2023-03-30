
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

