# 📘 Definitive Lesson: Introduction to Machine Learning with Python
### Based on notebook `s101.ipynb` — Explained for Complete Beginners

---

## 📋 Table of Contents

1. [What Is This Notebook About?](#overview)
2. [SECTION 1 — Installing & Importing Libraries](#section1)
3. [SECTION 2 — Loading the Dataset (Data Acquisition)](#section2)
4. [SECTION 3 — Exploratory Data Analysis (EDA)](#section3)
5. [SECTION 4 — Feature Engineering & Encoding](#section4)
6. [SECTION 5 — Building the Feature Matrix X](#section5)
7. [SECTION 6 — Standardization / Scaling](#section6)
8. [SECTION 7 — Visualizing Correlations](#section7)
9. [SECTION 8 — Linear Regression Theory](#section8)
10. [SECTION 9 — Loss Functions](#section9)
11. [SECTION 10 — The Custom Optimizer (Training Loop)](#section10)
12. [SECTION 11 — Evaluation & Denormalization](#section11)
13. [SECTION 12 — ML Models Overview (Instructor's Reference)](#section12)
14. [🔴 Critical Issue: No Train/Test Split](#critical)
15. [📚 Learning Resources](#resources)
16. [✅ Summary: Practice Ratings](#summary)

---

<a name="overview"></a>
## What Is This Notebook About?

This notebook teaches the **full ML pipeline** from scratch using a dataset about student habits and academic performance. The goal is to **predict a student's exam score** based on factors like study hours, sleep, diet, social media usage, etc.

**The ML Pipeline covered:**
```
Raw Data → EDA → Encoding → Scaling → Model → Training → Evaluation
```

Think of it like building a recipe predictor: you collect ingredients (data), understand what each ingredient is (EDA), convert everything to numbers a computer can use (encoding), normalize amounts (scaling), build a formula (model), and test if it works (evaluation).

---

<a name="section1"></a>
## SECTION 1 — Installing & Importing Libraries

### The Code
```python
%pip install numpy pandas matplotlib seaborn

import numpy
import pandas
from matplotlib import pyplot
import seaborn
```

### What Is Happening
- `%pip install` downloads external libraries (tools) that don't come built into Python.
- `import` makes those tools available to use in the notebook.

### The Four Libraries — What They Do

| Library | What It Is | Analogy |
|---|---|---|
| **numpy** | Math engine for arrays and calculations | A calculator for lists of numbers |
| **pandas** | Data table manager | Like Excel inside Python |
| **matplotlib** | Basic chart drawer | A drawing board for plots |
| **seaborn** | Advanced statistical chart drawer | A smarter, prettier drawing board |

### Why Is It Done?
Python alone can't handle large data tables or draw charts. These libraries are decades of work compressed into free tools.

### What If It's Not Done?
You'd get `ModuleNotFoundError` when trying to use `pandas`, etc.

### ⚠️ Quality Assessment: ACCEPTABLE — Could Be Improved

**Issue 1:** Best practice is to import with shorter aliases:
```python
# ✅ BETTER — standard convention everywhere in the data science world
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
Without aliases, you have to type `pandas.read_csv()` instead of the universally recognized `pd.read_csv()`. Every tutorial, book, and StackOverflow answer uses these aliases.

**Issue 2:** No version pinning. In production, you'd use a `requirements.txt` file.

---

<a name="section2"></a>
## SECTION 2 — Loading the Dataset

### The Code
```python
url = "https://raw.githubusercontent.com/.../student_habits_performance.csv"
students = pandas.read_csv(url)
students.head(10)
```

### What Is Happening
- A CSV (Comma-Separated Values) file is downloaded from GitHub.
- `read_csv()` parses it into a **DataFrame** — Python's version of an Excel table.
- `.head(10)` shows the first 10 rows so we can verify it loaded correctly.

### ML Concept: The Dataset
A dataset is a structured table of observations. Each **row** is one student (one observation). Each **column** is a characteristic (one **feature** or **variable**).

```
student_id | age | gender | study_hours | ... | exam_score
--------------------------------------------------------------
1          | 20  | Male   | 4.5         | ... | 78.3
2          | 19  | Female | 2.1         | ... | 61.0
```

The **exam_score** column is what we want to predict — it's called the **target** or **label** (y). Everything else is the **input** or **features** (X).

### ⚠️ Quality Assessment: GOOD — with one caveat
Loading from a URL is convenient for teaching. In real projects, you'd use a local file, a database, or a proper data versioning system. It's fine here.

---

<a name="section3"></a>
## SECTION 3 — Exploratory Data Analysis (EDA)

### The Code
```python
students.info()           # Column types, null counts
e1 = students["age"]      # Extract the "age" column
e1.plot.density()         # Draw a density (distribution) curve
e1.describe()             # Min, max, mean, std, quartiles
e1.plot.box(vert=False)   # Draw a box-and-whisker plot
e2 = students["gender"]
e2.describe()
e2.value_counts()         # Count how many Male/Female/Other
e2.value_counts().plot.pie()  # Pie chart
```

### What Is EDA?
**Exploratory Data Analysis** is the process of understanding your data before building a model. You would never build a house without surveying the land first — EDA is that survey.

**Key questions EDA answers:**
- Are there missing values (nulls)?
- What is the range of each column?
- Are there outliers?
- Is the data balanced (for categorical columns)?
- What kind of data is each column (numbers? text? dates)?

### The Plots Explained

**Density Plot (KDE - Kernel Density Estimate)**
Think of it as a smoothed histogram. If age were a pile of sand, the density plot shows the shape of that pile. A bell shape (Gaussian) is common and ideal for many algorithms.

**Box Plot (Box-and-Whisker)**
Shows 5 statistics at once:
```
|----[===|===]----o    (o = outlier)
 min  Q1 med Q3  max
```
- The **box** covers the middle 50% of the data (Q1 to Q3 = IQR)
- The **whiskers** extend to 1.5× IQR
- **Dots outside** the whiskers are outliers

**Pie Chart**
Shows proportions of categories. Here it shows the gender breakdown.

**`.describe()` output for numerical columns:**
```
count   1000.0    → how many values exist (no nulls = good)
mean      20.5    → average
std        1.8    → how spread out the values are
min       17.0    → smallest value
25%       19.0    → first quartile (25% of data is below this)
50%       20.0    → median (middle value)
75%       22.0    → third quartile (75% of data is below this)
max       24.0    → largest value
```

### The Instructor's Pattern: e1, e2, e3... (NAMING ISSUE)
```python
e1 = students["age"]
e2 = students["gender"]
# ...
e15 = students["exam_score"]
```

### ⚠️ Quality Assessment: 🔴 SHOULD BE IMPROVED

Using `e1`, `e2`, `e3` is a very bad naming practice called **magic numbers/variables**. After 10 minutes, you won't remember what `e7` is. After 1 week, nobody can read your code.

```python
# ✅ MUCH BETTER — self-documenting variable names
age = students["age"]
gender = students["gender"]
study_hours = students["study_hours_per_day"]
# etc.
```

**Why it matters:** Code is read far more often than it is written. Clear names are a form of documentation.

---

<a name="section4"></a>
## SECTION 4 — Feature Engineering & Encoding

### The Code
```python
# Binary encoding (Yes/No → 1/0)
x7 = (e6 == "Yes").astype(int)   # part_time_job

# One-hot-style encoding (categories → multiple binary columns)
x2 = (e2 == "Male").astype(int)   # is_male:  Male→1, other→0
x3 = (e2 == "Other").astype(int)  # is_other: Other→1, else→0

# Ordinal encoding (ordered categories → ordered numbers)
x10 = e9.map({
    "Poor": -1,
    "Fair": 0,
    "Good": 1,
})  # diet_quality

x12 = e11.map({
    "High School": 1,
    "Bachelor": 2,
    "Master": 4,
}).fillna(1.5)  # parental_education_level
```

### What Is Encoding?
**ML algorithms can only process numbers.** They can't understand the word "Male" or "Poor". Encoding is the process of translating human-readable categories into numbers.

Think of it like translating a recipe from Spanish to a universal language (numbers) so any chef in the world (algorithm) can follow it.

### Types of Encoding Used Here

#### 1. Binary Encoding
For Yes/No or True/False columns:
```
"Yes" → 1
"No"  → 0
```
Simple and correct for binary choices.

#### 2. Manual One-Hot Encoding for Gender
Gender has 3 values: Male, Female, Other.
The instructor creates 2 binary columns:
```
         is_male  is_other
Male  →    1        0
Female →   0        0       ← Female is the "reference" (both 0)
Other →    0        1
```
This is called "dummy encoding" or "one-hot with drop_first". It's mathematically correct and avoids **multicollinearity** (the "dummy variable trap").

**Why not just do `Male→1, Female→2, Other→3`?**
Because that implies Female is "twice Male" and Other is "three times Male" — which is nonsense. One-hot encoding avoids this false ordering.

#### 3. Ordinal Encoding
For categories that have a real order (Poor < Fair < Good):
```
Poor → -1, Fair → 0, Good → 1
```
This is valid when the ORDER is meaningful. The instructor's choice of -1/0/1 (centered) is fine.

**Issue with the instructor's scale:**
```python
# Diet:       Poor=-1, Fair=0, Good=1   (consistent spacing of 1)
# Internet:   Poor=-2, Average=1, Good=2  (INCONSISTENT: -2 to 1 is a jump of 3, 1 to 2 is a jump of 1)
# Education:  High School=1, Bachelor=2, Master=4  (INCONSISTENT spacing)
```
The internet and education scales have **inconsistent gaps** between values. This implies "Master is twice a Bachelor" which may not reflect reality. A better approach:

```python
# ✅ Consistent spacing
x13 = e12.map({"Poor": 0, "Average": 1, "Good": 2})
x12 = e11.map({"High School": 0, "Bachelor": 1, "Master": 2})
```

Or better yet, use `pd.get_dummies()` for categories without a clear order.

#### 4. `fillna(1.5)` — Handling Missing Values
```python
x12 = e11.map({...}).fillna(1.5)
```
If a student's parental education is unknown (NaN), the instructor fills it with 1.5. This is between "High School" (1) and "Bachelor" (2) — a guess.

**Better approaches:**
- Fill with the median of the known values
- Create an "Unknown" category

### ML Concept: Feature Engineering
Feature engineering is the art of creating the best input representation for your model. Good features = good model. This is often where experienced ML engineers spend most of their time.

### ⚠️ Quality Assessment: ACCEPTABLE with issues
Binary encoding: ✅ Good
One-hot for gender: ✅ Good concept
Ordinal mapping: ⚠️ Inconsistent scales for internet/education
fillna(1.5): ⚠️ Arbitrary — should be explained and justified

---

<a name="section5"></a>
## SECTION 5 — Building the Feature Matrix X and Target y

### The Code
```python
X = pandas.DataFrame([
    x1, x2, x3, x4, x5, x6,
    x7, x8, x9, x10, x11, x12,
    x13, x14, x15
]).T

y = e15  # exam_score (the target)
```

### What Is Happening
All the encoded feature series are stacked into a single 2D matrix X. Each column is a feature; each row is one student.

```
      age  is_male  is_other  study_hours  ... mental_health
0     20      1        0         4.5       ...     7
1     19      0        0         2.1       ...     5
...
```

The `.T` at the end **transposes** the DataFrame (flips rows and columns), because `pandas.DataFrame([series1, series2, ...])` stacks series as rows by default, but we want them as columns.

### ML Concept: X and y
In ML, the convention is:
- **X** (uppercase) = the feature matrix (2D, many rows and columns)
- **y** (lowercase) = the target vector (1D, one value per row)

This comes from linear algebra where a prediction is written as `ŷ = Xβ`.

### ⚠️ Quality Assessment: ACCEPTABLE
The `.T` trick works but is fragile. A cleaner approach:
```python
# ✅ BETTER — more explicit and readable
feature_columns = ["age", "is_male", "is_other", "study_hours_per_day", ...]
X = students[feature_columns]
y = students["exam_score"]
```

---

<a name="section6"></a>
## SECTION 6 — Standardization / Scaling

### The Code
```python
from sklearn.preprocessing import StandardScaler

Xs = pandas.DataFrame(
    StandardScaler().fit_transform(X),
    columns=X.columns
)

# Manual z-score for y
ys = (y - y.mean()) / y.std()
```

### What Is Happening
**StandardScaler** transforms each column so that:
- The **mean becomes 0**
- The **standard deviation becomes 1**

The formula is: `x_scaled = (x - mean) / std`

This is called **Z-score normalization** or **standardization**.

### Why Is This Done?
Imagine you're comparing the height of a person (170 cm) with their weight (70 kg). The height number is 2.4× bigger than the weight number, so an algorithm that uses raw numbers might wrongly "think" height is more important.

Standardization puts **all features on the same scale**, so the algorithm can compare them fairly.

**Concrete example:**
```
Before:  age=20,  study_hours=4.5,  attendance=87.3
After:   age=0.2, study_hours=0.8,  attendance=0.1
```
Now all values are in a similar range (roughly -3 to +3).

### When Is Scaling Critical?
- **Linear Regression**: Less critical (coefficients adjust), but helps with numerical stability
- **Gradient Descent**: VERY critical — unscaled data causes slow or failed training
- **SVM, KNN, Neural Networks**: CRITICAL — distances and gradients are affected by scale

### What Happens If You Don't Scale?
- Training is slow or fails to converge
- Features with large ranges dominate the model unfairly
- Coefficients (betas) have inconsistent magnitudes

### The Instructor's y Scaling
```python
ys = (y - y.mean()) / y.std()
```
This is the same formula as StandardScaler, done manually. It works, but it's inconsistent with the approach used for X (using sklearn for X but manual math for y). A consistent approach would use StandardScaler for both, or manual math for both.

### After Scaling: Before vs After Comparison
```python
X["age"].plot.density()    # Before: shape around 17-24
Xs["age"].plot.density()   # After: shape centered at 0, spread from ~-3 to +3
```
**The shape of the distribution does not change — only the scale changes.** This is an important insight.

### ⚠️ Quality Assessment: GOOD concept, INCONSISTENT execution
- Using StandardScaler for X: ✅
- Manually scaling y instead of using sklearn: ⚠️ Inconsistent
- Not saving the scaler to inverse-transform predictions later: ⚠️ Issue (handled manually later in the code)

---

<a name="section7"></a>
## SECTION 7 — Visualizing Correlations

### The Code
```python
seaborn.jointplot(x=Xs.iloc[:, 3], y=ys, kind="hex")   # study_hours vs exam_score
seaborn.jointplot(x=Xs.iloc[:, 4], y=ys, hue=y>=70)    # social_media vs exam
seaborn.jointplot(x=Xs.iloc[:, 5], y=ys, hue=y>=70)    # netflix vs exam
seaborn.jointplot(x=Xs.iloc[:, 3], y=ys, hue=y>=70)    # study_hours vs exam
```

### What Is a Joint Plot?
A **joint plot** shows the relationship between two variables at once:
- The **center** shows a scatter plot (or hex bin)
- The **top histogram** shows the distribution of X
- The **right histogram** shows the distribution of y

### What You're Looking For
- A **positive diagonal trend** → as X increases, y increases (positive correlation)
- A **negative diagonal trend** → as X increases, y decreases (negative correlation)
- A **cloud with no trend** → the variable doesn't help predict y

You'd expect study_hours to show a positive diagonal (more study → higher score), and netflix_hours to show a negative one.

### The `hue=y>=70` Trick
This colors points red/blue based on whether the student passed (score ≥ 70). It's a quick way to see if a feature separates "passing" from "failing" students.

### ⚠️ Quality Assessment: GOOD
Using joint plots for EDA is great practice. **Issue:** using `Xs.iloc[:, 3]` instead of `Xs["study_hours_per_day"]` makes it unclear which column you're looking at.

---

<a name="section8"></a>
## SECTION 8 — Linear Regression Theory

### Markdown Cells in the Notebook
```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₖ·xₖ
```

### What Is Linear Regression?
Linear regression tries to find a **straight-line relationship** between inputs (X) and an output (y).

**Analogy:** Imagine you want to predict house prices. You know that:
- Each extra bedroom adds $50,000
- Each year of age subtracts $3,000
- Being near a school adds $20,000

The "formula" is: `price = 200,000 + 50,000×bedrooms - 3,000×age + 20,000×near_school`

In ML notation:
- `β₀ = 200,000` is the **intercept** (base price)
- `β₁ = 50,000` is the **coefficient** for bedrooms
- The β values are what the model **learns**

### The Code Implementation
```python
beta0 = numpy.random.normal(0, 0.1, 1)   # Random starting intercept
betas = numpy.random.normal(0, 0.1, 15)  # Random starting coefficients (one per feature)

yp = beta0 + Xs.dot(betas)  # The prediction formula
```

`Xs.dot(betas)` is the **dot product** — it multiplies each feature by its corresponding coefficient and sums everything up. This is the "score" for each student.

### ⚠️ Quality Assessment: GOOD for learning
Implementing linear regression from scratch is an excellent educational exercise. In practice, you'd use `sklearn.linear_model.LinearRegression` which is faster, validated, and handles edge cases.

---

<a name="section9"></a>
## SECTION 9 — Loss Functions

### The Code
```python
def loss_mse(y, yp):
    e = y - yp          # error = actual - predicted
    mse = (e ** 2).mean()
    return mse

def loss_mae(y, yp):
    e = y - yp
    mae = e.abs().mean()
    return mae

# Metrics table
e = ys - yp
SSE  = (e ** 2).sum()       # Sum of Squared Errors
MSE  = (e ** 2).mean()      # Mean Squared Error
RMSE = (e ** 2).mean() ** 0.5  # Root Mean Squared Error
SAE  = e.abs().sum()        # Sum of Absolute Errors
MAE  = e.abs().mean()       # Mean Absolute Error
```

### What Is a Loss Function?
A **loss function** (also called cost function) measures how wrong the model's predictions are. It's a single number that summarizes the total error.

**Goal: minimize the loss function.**

### The Error Metrics Explained

| Metric | Formula | Meaning | Sensitive to Outliers? |
|--------|---------|---------|----------------------|
| **SSE** | Σ(y - ŷ)² | Total squared error (scale-dependent) | Yes |
| **MSE** | SSE / n | Average squared error | Yes |
| **RMSE** | √MSE | Average error in same units as y | Yes |
| **SAE** | Σ|y - ŷ| | Total absolute error | Less |
| **MAE** | SAE / n | Average absolute error in units of y | Less |

**RMSE vs MAE — Which to Use?**
- **RMSE** penalizes large errors more (because of the squaring). Use it when big mistakes are especially bad.
- **MAE** treats all errors equally. Use it when all error sizes matter equally.

**Analogy:** If you're predicting package delivery time:
- RMSE punishes being 2 hours late much more than being 1 hour late
- MAE treats each hour of lateness the same

### ⚠️ Quality Assessment: GOOD
The instructor correctly defines and explains all common metrics. The code is clean and readable.

---

<a name="section10"></a>
## SECTION 10 — The Custom Optimizer (Training Loop)

### The Code
```python
def optimizer(X, y, beta0, betas, loss):
    # Create slightly perturbed versions of the parameters
    beta0_n = beta0 + numpy.random.normal(0, 0.001, 1)
    betas_n = betas.copy() + numpy.random.normal(0, 0.001, len(betas))

    # Compute predictions and losses for old and new parameters
    yp1 = beta0 + X.dot(betas)
    yp2 = beta0_n + X.dot(betas_n)

    e1 = loss(y, yp1)
    e2 = loss(y, yp2)

    # Keep whichever version has lower loss
    if e1 > e2:
        return beta0_n, betas_n
    else:
        return beta0, betas

# Training loop
beta0 = numpy.random.normal(0, 0.1, 1)
betas = numpy.random.normal(0, 0.1, 15)

for i in range(100_000):
    beta0, betas = optimizer(Xs, ys, beta0, betas, loss_mse)
    if i % 1000 == 0:
        yp = beta0 + Xs.dot(betas)
        e = loss_mse(ys, yp)
        print(i, "MSE:", e)
```

### What Is Happening
The optimizer:
1. Takes the current coefficients (beta0, betas)
2. Creates slightly "nudged" versions by adding tiny random noise
3. Computes the loss for both old and new coefficients
4. Keeps whichever set of coefficients has **lower loss**
5. Repeats 100,000 times

**Analogy:** Imagine you're blindfolded on a hilly landscape trying to reach the valley (the minimum loss). In each step, you randomly shuffle your feet slightly and check if you moved downhill. If yes, you stay there. If no, you go back.

### The Instructor Calls This "Bayesian" — Is It?
The instructor's comment says `# Optimizador (usando bayesiana)`, but **this is NOT a Bayesian optimizer**.

This is actually a **Random Walk Hill Climbing** algorithm (also called random perturbation search). A true Bayesian optimizer would build a probabilistic model of the loss landscape and make informed decisions about where to sample next.

### ⚠️ Quality Assessment: 🔴 SHOULD BE IMPROVED — Multiple Issues

**Issue 1: Mislabeled as "Bayesian"**
The optimizer is actually a random walk hill climber. This is a conceptual error.

**Issue 2: Extremely Inefficient**
Running 100,000 iterations of random search is very slow and converges poorly. The standard approach for linear regression is either:
- **Closed-form solution (Normal Equation):** `β = (XᵀX)⁻¹Xᵀy` — exact answer in one computation
- **Gradient Descent:** Mathematically computes the direction of steepest descent — much faster

**Issue 3: No Train/Test Split** ← See the Critical Issue section below

**Issue 4: Tiny perturbation (0.001) means slow exploration**
The standard deviation of 0.001 means the optimizer barely moves each step. This is why 100,000 iterations are needed.

**The correct way — using sklearn (3 lines vs 100,000 iterations):**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)  # Instant, optimal solution
```

---

<a name="section11"></a>
## SECTION 11 — Evaluation & Denormalization

### The Code
```python
# Denormalize: convert scaled predictions back to original units
pandas.DataFrame([
    ys * y.std() + y.mean(),     # actual scores (denormalized)
    yp * y.std() + y.mean()      # predicted scores (denormalized)
]).T

seaborn.regplot(x=yp, y=ys, line_kws={"color": "red"})
```

### What Is Denormalization?
Since we scaled `y` to `ys = (y - mean) / std`, to get back to the original scale:
```
y_original = ys × std + mean
```

This is important because "MSE of 0.3 on scaled data" is meaningless to a teacher who wants to know "how many points off are you?".

### The `regplot` Visualization
A **regression plot** shows:
- **X-axis:** Predicted values (ŷ)
- **Y-axis:** Actual values (y)
- **Red line:** The "ideal" diagonal (if predictions were perfect, all points would be on this line)

If the model is good, the scatter of points should be clustered tightly around the diagonal.

### ⚠️ Quality Assessment: GOOD
Denormalizing predictions before showing them is correct practice. The regplot is the right visualization for regression evaluation.

---

<a name="critical"></a>
## 🔴 CRITICAL ISSUE: No Train/Test Split

### What Is Missing
**The entire notebook trains and evaluates on the same data.** This is one of the most serious mistakes in ML.

### Why Is This a Problem?
Imagine studying for an exam by memorizing the exact questions that will be on the exam. You'd score 100%, but you haven't actually learned anything — you just memorized.

In ML, if you train and test on the same data, your model might "memorize" the training data (called **overfitting**) and appear to perform well. But when it sees new, unseen data, it fails.

### The Solution: Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing, 80% for training
    random_state=42   # reproducibility
)

# Train ONLY on training data
model.fit(X_train, y_train)

# Evaluate ONLY on test data (data the model has never seen)
y_pred = model.predict(X_test)
```

This is **non-negotiable** in real ML work. Always split your data.

---

<a name="section12"></a>
## SECTION 12 — ML Models Overview (Instructor's Reference)

The notebook ends with a markdown cell summarizing the ML landscape. This is a good reference:

### Regression Models (predicting a number)
- **Linear Regression** — assumes a straight-line relationship
- **Ridge, Lasso, ElasticNet** — Linear regression with regularization (prevents overfitting)
- **Support Vector Regression** — finds the best "tube" around the data
- **Decision Tree Regression** — splits data into regions, predicts the average in each
- **Random Forest Regression** — averages many decision trees (more stable)

### Classification Models (predicting a category)
- **Logistic Regression** — predicts probability of belonging to a class
- **Decision Tree** — uses rules to classify
- **Random Forest** — many decision trees voting
- **XGBoost** — powerful gradient boosting algorithm

### Neural Networks
- Regression: Linear/ReLU activation
- Binary classification: Sigmoid activation
- Multi-class classification: Softmax activation

---

<a name="resources"></a>
## 📚 Learning Resources by Topic

### Python & Pandas
- **EN:** [Pandas Official Documentation](https://pandas.pydata.org/docs/)
- **EN:** [Kaggle Pandas Course (free)](https://www.kaggle.com/learn/pandas)
- **ES:** [Pandas en Español — Corey Schafer subtitulado](https://www.youtube.com/playlist?list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS)
- **Book:** "Python for Data Analysis" by Wes McKinney (creator of pandas)

### Exploratory Data Analysis (EDA)
- **EN:** [Kaggle EDA Guide](https://www.kaggle.com/learn/data-visualization)
- **ES:** [EDA con Python en Español](https://www.youtube.com/watch?v=Xi0h4e4usI0)
- **EN Video:** [StatQuest EDA playlist](https://www.youtube.com/watch?v=xTYPKqn_K5U)

### Encoding & Feature Engineering
- **EN:** [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- **ES:** [Feature Engineering en Español (Platzi)](https://platzi.com/cursos/feature-engineering/)
- **Book:** "Feature Engineering for Machine Learning" by Alice Zheng

### Scaling & Normalization
- **EN:** [StatQuest: Feature Scaling](https://www.youtube.com/watch?v=0MrgsYswT1c)
- **ES:** [Normalización vs Estandarización](https://www.youtube.com/watch?v=4-MNp5b-F4Q)

### Linear Regression
- **EN Video:** [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) ← HIGHLY RECOMMENDED
- **EN Video:** [3Blue1Brown: What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (builds intuition)
- **ES Video:** [Regresión Lineal desde Cero](https://www.youtube.com/watch?v=w2RJ1D6kz-o)
- **ES:** [Machine Learning con Python — Platzi](https://platzi.com/cursos/scikit/)

### Loss Functions & Optimization
- **EN:** [StatQuest: MSE and RMSE](https://www.youtube.com/watch?v=_-Cbe2bfxJs)
- **EN:** [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- **ES:** [Funciones de Pérdida en Español](https://www.youtube.com/watch?v=EuBBz3bI-aA)

### Train/Test Split & Model Evaluation
- **EN:** [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **EN Video:** [StatQuest: Train/Test Split](https://www.youtube.com/watch?v=fSytzGwwBVw)
- **ES Video:** [División de datos en Español](https://www.youtube.com/watch?v=Y4bGVFmFcFg)

### Full ML Courses (Free)
- **EN:** [fast.ai — Practical Deep Learning](https://www.fast.ai/) ← Hands-on, free
- **EN:** [Scikit-learn Tutorial](https://scikit-learn.org/stable/tutorial/index.html)
- **ES:** [Curso ML con Python — Curso en Video](https://www.youtube.com/watch?v=jMHLR8gJ9iI)
- **EN/ES:** [Coursera Machine Learning by Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction) ← The gold standard

### Books (Beginner-Friendly)
- "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron
- "Python Machine Learning" by Sebastian Raschka
- "Introduction to Machine Learning with Python" by Andreas Müller

---

<a name="summary"></a>
## ✅ Summary: Practice Ratings for Each Section

| Section | Instructor's Approach | Rating | Key Issue |
|---------|----------------------|--------|-----------|
| Library imports | No aliases | ⚠️ Acceptable | Should use `import pandas as pd` etc. |
| Variable naming (e1-e15) | Magic variable names | 🔴 Improve | Use descriptive names |
| EDA (info, describe, plots) | Thorough | ✅ Good | — |
| Binary encoding | Correct | ✅ Good | — |
| Ordinal encoding | Inconsistent scales | ⚠️ Acceptable | Internet/education scales are arbitrary |
| Feature matrix construction | .T trick works | ⚠️ Acceptable | Explicit column selection is cleaner |
| StandardScaler | Correct | ✅ Good | y scaled manually instead of with sklearn |
| Joint plots | Good visualization | ✅ Good | Uses .iloc[:, 3] instead of column names |
| Loss functions | Correct & complete | ✅ Good | — |
| Optimizer (labeled "Bayesian") | Random walk, mislabeled | 🔴 Improve | Not Bayesian; very inefficient |
| 100,000 iterations | Overkill for linear regression | 🔴 Improve | sklearn solves this in 1 line |
| **Train/Test Split** | **MISSING** | 🔴🔴 Critical | **This is non-negotiable in ML** |
| Denormalization | Correct | ✅ Good | — |
| Evaluation metrics | Complete | ✅ Good | — |

---

*This lesson was generated as a companion to notebook s101.ipynb from the course ia-cic-abril-2026 by Alan Badillo Salas. All explanations, quality assessments, and improvements are provided for educational purposes.*
