</head>
<body>
  <h1>Advanced Correlation Analysis Toolkit</h1>

  <p>
    This project provides a Python script to perform <strong>comprehensive correlation analyses</strong> 
    on cleaned datasets. It automatically computes <strong>Pearson</strong> and <strong>Spearman</strong> 
    correlation coefficients, their <strong>p-values</strong>, generates 
    <strong>high-resolution heatmaps</strong>, <strong>publication-ready tables</strong>, and 
    <strong>summary reports</strong> for predictor vs. target relationships.
  </p>

  <h2>Statistical Background</h2>
  <p>
    The script implements statistical measures useful for <strong>machine learning preprocessing</strong>. 
    All equations and explanations use proper subscripts, superscripts, and Greek letters for clarity.
  </p>

  <p><strong>Mean (average) of a variable X:</strong></p>
  <p>$$
  \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$X_i$$ = the ith observation of variable X</li>
    <li>$$\bar{X}$$ = mean value of X</li>
    <li>$$n$$ = total number of observations</li>
  </ul>

  <p><strong>Standard deviation:</strong></p>
  <p>$$
  \sigma_X = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$\sigma_X$$ = standard deviation of X</li>
    <li>$$X_i$$ = ith observation</li>
    <li>$$\bar{X}$$ = mean of X</li>
    <li>$$n$$ = total number of observations</li>
  </ul>

  <p><strong>Covariance between X and Y:</strong></p>
  <p>$$
  \text{Cov}(X,Y) = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$\text{Cov}(X,Y)$$ = covariance between X and Y</li>
    <li>$$X_i, Y_i$$ = ith observations of X and Y</li>
    <li>$$\bar{X}, \bar{Y}$$ = means of X and Y</li>
    <li>$$n$$ = total number of observations</li>
  </ul>

  <p><strong>Pearson Correlation Coefficient (r):</strong></p>
  <p>$$
  r_{XY} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}
  {\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$r_{XY}$$ = Pearson correlation coefficient between X and Y</li>
    <li>Numerator = covariance between X and Y</li>
    <li>Denominator = product of standard deviations of X and Y</li>
    <li>Value ranges from -1 to 1</li>
  </ul>

  <p><strong>Pearson p-value:</strong></p>
  <p>$$
  t = r \sqrt{\frac{n - 2}{1 - r^2}}, \quad df = n-2
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$t$$ = test statistic for correlation significance</li>
    <li>$$r$$ = Pearson correlation coefficient</li>
    <li>$$n$$ = number of paired observations</li>
    <li>$$df$$ = degrees of freedom = n - 2</li>
  </ul>

  <p><strong>Spearman Rank Correlation (ρ):</strong></p>
  <p>$$
  \rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$\rho$$ = Spearman rank correlation coefficient</li>
    <li>$$d_i$$ = difference in ranks of X_i and Y_i</li>
    <li>$$n$$ = number of paired observations</li>
  </ul>

  <p>Rank difference definition:</p>
  <p>$$
  d_i = \text{rank}(X_i) - \text{rank}(Y_i)
  $$</p>
  <p>Explanation: Each $$d_i$$ represents the difference in rank positions of X_i and Y_i in the dataset.</p>

  <p><strong>Spearman p-value:</strong></p>
  <p>$$
  z = \rho \sqrt{\frac{n - 1}{1 - \rho^2}}, \quad z \sim N(0,1)
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$z$$ = approximate test statistic</li>
    <li>$$\rho$$ = Spearman correlation coefficient</li>
    <li>$$n$$ = number of paired observations</li>
    <li>For large n, z follows standard normal distribution</li>
  </ul>

  <p><strong>Number of valid pairs (N):</strong></p>
  <p>$$
  N = \sum_{i=1}^n I(X_i \neq \text{NaN} \ \&\ Y_i \neq \text{NaN})
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$N$$ = number of valid paired observations</li>
    <li>$$I$$ = indicator function: 1 if both X_i and Y_i are valid, 0 otherwise</li>
  </ul>

  <p><strong>Coefficient of Determination (R²):</strong></p>
  <p>$$
  R^2 = r_{XY}^2
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$R^2$$ = proportion of variance in Y explained by X</li>
    <li>$$r_{XY}$$ = Pearson correlation coefficient between X and Y</li>
  </ul>

  <p><strong>Z-score standardization:</strong></p>
  <p>$$
  Z_i = \frac{X_i - \bar{X}}{\sigma_X}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$Z_i$$ = standardized value of X_i</li>
    <li>$$X_i$$ = original observation</li>
    <li>$$\bar{X}$$ = mean of X</li>
    <li>$$\sigma_X$$ = standard deviation of X</li>
  </ul>

  <h2>Features</h2>
  <ul>
    <li><strong>Automatic Preprocessing:</strong> Identifies numeric predictors and target variables, coercing non-numeric columns where possible.</li>
    <li><strong>Computation of Pearson & Spearman:</strong> Correlation coefficients and p-values with pairwise handling of missing data.</li>
    <li><strong>High-Resolution Heatmaps:</strong> HD (1200 dpi) heatmaps with customizable colormaps.</li>
    <li><strong>Publication-Ready Tables:</strong> Tables generated as CSV, Excel, LaTeX, and PNG with Times New Roman formatting.</li>
    <li><strong>Predictor vs Target Summary:</strong> A compact report pairing each predictor with the chosen target variable.</li>
  </ul>

  <h2>Usage Instructions</h2>
  <ol>
    <li>Place your cleaned dataset in the project directory.</li>
    <li>Run the script with Python:
      <pre><code>python Get_Corr_Results.py</code></pre>
    </li>
    <li>The script will:
      <ul>
        <li>Detect predictors and target variable (last non-empty column).</li>
        <li>Compute Pearson & Spearman correlation matrices and p-values.</li>
        <li>Export results to CSV, Excel, LaTeX, and PNG heatmaps/tables.</li>
        <li>Create a summary table for predictor–target correlations.</li>
      </ul>
    </li>
  </ol>

  <h2>Requirements</h2>
  <ul>
    <li>Python 3.8+</li>
    <li>Libraries: pandas, numpy, matplotlib, seaborn, scipy</li>
    <li>A cleaned dataset with at least two non-empty numeric columns</li>
  </ul>

  <h2>License</h2>
  <p>This project is licensed under the <strong>MIT License</strong>. Feel free to use, modify, and redistribute. Credit is appreciated.</p>

  <h2>Developer Info</h2>
  <ul>
    <li><strong>Developer:</strong> Engr. Tufail Mabood</li>
    <li><strong>Contact:</strong> <a href="https://wa.me/+923440907874">WhatsApp</a></li>
    <li><strong>Note:</strong> If you need help with statistical analysis or data preparation, feel free to reach out.</li>
  </ul>
</body>
