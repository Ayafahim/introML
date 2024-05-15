
## S19, Q8) Suppose we use the classification error as impurity measure, which one of the following statements is true?
![[Pasted image 20240515180300.png]]
### Solution

1. **Data Setup**:
   - The attribute $x_4$ is used for binary splits at two different threshold values $z$: $x_4 \leq 0.43$ and $x_4 \leq 0.55$.
   - The data is split into intervals, and for each interval, the count of observations falling into each class ($y = 1$, $y = 2$, $y = 3$) is recorded.

2. **Objective**:
   - Calculate the impurity gain of each split to determine which split is better or to validate a given statement about the splits.

3. **Concepts Involved**:
   - **Classification Error as Impurity Measure**: This is the proportion of incorrectly classified instances if a node is split in a certain way.
   - **Impurity Gain Calculation**: The difference in impurity (classification error here) from before to after the split. Calculated as $\Delta = I(\text{parent}) - (p_{\text{left}} I(\text{left}) + p_{\text{right}} I(\text{right}))$ where $p$ is the proportion of instances in left and right child nodes respectively.

#### Steps to Calculate the Impurity Gain:

1. **Calculate Initial Impurity ($I(\text{parent})$)**:
   - Using the formula for classification error: $1 - \max(p(y=1), p(y=2), p(y=3))$, where $p(y=i)$ is the proportion of class $i$ instances at the node before splitting.

2. **Calculate Impurity of Each Child Node**:
   - Apply the same formula to the instances in each interval created by the split.

3. **Calculate Proportions of Instances ($p_{\text{left}}$ and $p_{\text{right}}$)**:
   - Determine how many instances go to each side of the split relative to the total number of instances at the node.

4. **Compute Impurity Gain**:
   - Use the weighted impurity of each child node to calculate the total impurity after the split and then determine the impurity gain.

#### Example Calculation for Split $x_4 \leq 0.43$:

- **Total instances**: $n_{y=1} + n_{y=2} + n_{y=3} = 263 + 359 + 358 = 980$
- **Before split**: 
  - Classification error $= 1 - \max(\frac{263}{980}, \frac{359}{980}, \frac{358}{980}) \approx 1 - 0.3663 = 0.6337$

- **After split (for $x_4 \leq 0.43$)**:
  - Left node: 143 (y=1), 137 (y=2), 54 (y=3)
    - $I(\text{left}) = 1 - \max(\frac{143}{334}, \frac{137}{334}, \frac{54}{334}) \approx 1 - 0.4281 = 0.5719$
  - Right node: 120 (y=1), 222 (y=2), 304 (y=3)
    - $I(\text{right}) = 1 - \max(\frac{120}{646}, \frac{222}{646}, \frac{304}{646}) \approx 1 - 0.4706 = 0.5294$
  - $p_{\text{left}} = \frac{334}{980} \approx 0.3408$
  - $p_{\text{right}} = \frac{646}{980} \approx 0.6592$

- **Calculate Impurity Gain**:
  - $\Delta = 0.6337 - (0.3408 \times 0.5719 + 0.6592 \times 0.5294) \approx 0.6337 - (0.1949 + 0.3489) = 0.6337 - 0.5438 = 0.0899$

Given this, we can evaluate the statements in the exam question to determine which one correctly describes the impurity gain from the splits. You can use the same process for the other split $x_4 \leq 0.55$. This thorough breakdown ensures accuracy and clarity in your answer.

```python
def calculate_classification_error(counts):
    total = sum(counts)
    max_class_count = max(counts)
    classification_error = 1 - (max_class_count / total)
    return classification_error

def impurity_gain(parent_counts, left_counts, right_counts):
    # Calculate the initial impurity
    parent_impurity = calculate_classification_error(parent_counts)
    
    # Calculate impurity for each child
    left_impurity = calculate_classification_error(left_counts)
    right_impurity = calculate_classification_error(right_counts)
    
    # Calculate proportions
    total_instances = sum(parent_counts)
    p_left = sum(left_counts) / total_instances
    p_right = sum(right_counts) / total_instances
    
    # Calculate impurity gain
    gain = parent_impurity - (p_left * left_impurity + p_right * right_impurity)
    return gain

# Example data from your problem
# Total counts for each class across all instances
parent_counts = [263, 359, 358]  # Total counts for y=1, y=2, y=3

# Counts for each class for the split at x4 <= 0.43
left_counts_043 = [143, 137, 54]  # Counts for y=1, y=2, y=3 in the left node
right_counts_043 = [120, 222, 304]  # Counts for y=1, y=2, y=3 in the right node

# Counts for each class for the split at x4 <= 0.55
left_counts_055 = [223, 251, 197]  # Counts for y=1, y=2, y=3 in the left node
right_counts_055 = [40, 108, 161]  # Counts for y=1, y=2, y=3 in the right node

# Calculate impurity gains for each split
gain_043 = impurity_gain(parent_counts, left_counts_043, right_counts_043)
gain_055 = impurity_gain(parent_counts, left_counts_055, right_counts_055)

print("Impurity Gain for x4 <= 0.43:", gain_043)
print("Impurity Gain for x4 <= 0.55:", gain_055)
```

## S19, Q9) Consider the splits in Table 3. Suppose we build a classification tree considering only the split x4 ≤ 0.55 and evaluate it on the same data it was trained upon. What is the accuracy?

1. **Count Correct Classifications**: Count how many observations were correctly classified by the split.
2. **Calculate Total Observations**: Determine the total number of observations classified by the split.
3. **Compute Accuracy**: Divide the number of correct classifications by the total number of observations.


```python
def calculate_right_node_counts(total_counts, left_counts):
    return {class_label: total_counts[class_label] - left_counts.get(class_label, 0) for class_label in total_counts}

def calculate_accuracy(total_counts, left_counts, right_counts):
    correct_predictions = max(left_counts.values()) + max(right_counts.values())
    total_predictions = sum(total_counts.values())
    return correct_predictions / total_predictions

# Total counts for each class
total_counts = {'y=1': 263, 'y=2': 359, 'y=3': 358}

# Counts for each class for the split at x4 <= 0.55 in the left node (given)
left_counts_055 = {'y=1': 223, 'y=2': 251, 'y=3': 197}

# Calculate right node counts based on total and left counts
right_counts_055 = calculate_right_node_counts(total_counts, left_counts_055)


## or count them yourself

# Split counts after applying x4 <= 0.55 (for example)  
split_counts = [  
    {'y=1': 223, 'y=2': 251, 'y=3': 197},  # left node  
    {'y=1': 40, 'y=2': 108, 'y=3': 161}  # right node (right_node_count=total_class_count−left_node_count)  
]

# Calculate accuracy
accuracy = calculate_accuracy(total_counts, left_counts_055, right_counts_055)
print("Accuracy of the split at x4 <= 0.55:", accuracy)
print("Right node counts:", right_counts_055)
```

