

## Attribute Types

| Attribute Type | Description                                                                                       | Example                                            |
| -------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Nominal**    | Categorical data without an intrinsic ordering or ranking.                                        | Colors: `Red`, `Blue`, `Green`                     |
|                |                                                                                                   | Fruits: `Apple`, `Banana`, `Cherry`                |
| **Ordinal**    | Categorical data with an intrinsic ordering or ranking but no consistent interval between values. | Education: `High School`, `Bachelor's`, `Master's` |
|                |                                                                                                   | Satisfaction: `Low`, `Medium`, `High`              |
| **Interval**   | Numerical data with meaningful intervals but no true zero.                                        | Temperature in Celsius: `10°C`, `20°C`, `30°C`     |
|                |                                                                                                   | Dates: `2000`, `2001`, `2002`                      |
| **Ratio**      | Numerical data with meaningful intervals and a true zero.                                         | Age: `10`, `20`, `30`                              |
|                |                                                                                                   | Income: `$1000`, `$2000`, `$3000`                  |

![[Pasted image 20240420185007.png]]


## Data Types

- **Discrete Data**: Consists of distinct or separate values. Often counts or categories.
  - **Example**: Number of students in a class, number of cars in a parking lot.
- **Continuous Data**: Can take any value within a range. Often measurements.
  - **Example**: Height of students, time taken to complete a task.

## Feature Transformations

### One-out-of-K Coding (One-Hot Encoding)

**Explanation:**
- This is used for nominal categorical data.
- Each category/value is transformed into a new binary feature (1 or 0).
- There will be as many new features as there are categories.

**Example:**
For the attribute `Color` with three categories `Red`, `Blue`, and `Green`:
- **Original Data**: `Red, Red, Blue, Green`
- **Transformed Data**:
  - `Color_Red: 1, 1, 0, 0`
  - `Color_Blue: 0, 0, 1, 0`
  - `Color_Green: 0, 0, 0, 1`

### Binarizing/Thresholding

**Explanation:**
- This process converts numerical values into binary values based on a threshold.
- It is useful for creating binary variables from continuous data.

**Example:**
For the attribute `Age` with a threshold at $30$:
- **Original Data**: `22, 35, 29, 31`
- **Transformed Data** $(\text{Age} > 30 = 1, \text{else} = 0)$:
  - `0, 1, 0, 1`

## Python Code for Feature Transformations

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, Binarizer

# Sample Data
data = {'Color': ['Red', 'Red', 'Blue', 'Green'],
        'Age': [22, 35, 29, 31]}
df = pd.DataFrame(data)

# One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[['Color']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Color']))
result_df = pd.concat([df.drop(['Color'], axis=1), encoded_df], axis=1)

# Binarizing
binarizer = Binarizer(threshold=30)
df['Age_binarized'] = binarizer.fit_transform(df[['Age']])

# Display results
print("One-Hot Encoded Data:")
print(result_df)
print("\nBinarized Data:")
print(df)
```

## Explanation for Beginners

- **One-out-of-K coding**: This makes it easier for algorithms to handle categorical data by removing the ordinal relationship between categories and instead treating each category as a separate, independent feature.
- **Binarizing**: Simplifies data by categorizing values into "above threshold" and "below threshold," which can be particularly useful when you're interested only in whether a value exceeds a certain critical level rather than its exact magnitude.

--- 


