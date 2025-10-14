# Two approaches to add confidence intervals to your frontier trend plot

## OPTION 1: Using statsmodels (for confidence/prediction bands)
# Add this import at the top of your notebook
import statsmodels.api as sm

# ... your existing code for identifying frontier points ...
# Then replace the fitting section:

if len(df_frontier) >= 2:
    # Prepare data
    X = (df_frontier['date_obj'] - df_frontier['date_obj'].min()).dt.days.values.reshape(-1, 1)
    y = df_frontier['estimated_capability'].values
    
    # Fit with sklearn (for consistency)
    lr = LinearRegression()
    lr.fit(X, y)
    slope_per_year = lr.coef_[0] * 365.25
    r2 = lr.score(X, y)
    
    # Also fit with statsmodels for confidence intervals
    X_sm = sm.add_constant(X.flatten())
    model_sm = sm.OLS(y, X_sm).fit()
    
    # Get predictions with confidence intervals
    predictions = model_sm.get_prediction(X_sm)
    ci_lower = predictions.conf_int()[:, 0]  # 95% confidence interval
    ci_upper = predictions.conf_int()[:, 1]
    
    # Print confidence interval info
    print(f"Slope per year: {slope_per_year:.4f}")
    print(f"Slope 95% CI: [{model_sm.params[1] * 365.25 - 1.96 * model_sm.bse[1] * 365.25:.4f}, "
          f"{model_sm.params[1] * 365.25 + 1.96 * model_sm.bse[1] * 365.25:.4f}]")
    print(f"R^2: {r2:.4f}")
    
    # Plot with confidence bands
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df_all['date_obj'], df_all['estimated_capability'], 
               alpha=0.25, s=20, label='All models', color='lightblue')
    ax.scatter(df_frontier['date_obj'], df_frontier['estimated_capability'], 
               alpha=0.9, s=40, label=f'Top {TOP_N_AT_RELEASE} at release', color='blue')
    
    # Plot trend line
    ax.plot(df_frontier['date_obj'], lr.predict(X), 'r--', alpha=0.8, label='Frontier trend')
    
    # Plot confidence interval as shaded region
    ax.fill_between(df_frontier['date_obj'], ci_lower, ci_upper, 
                     alpha=0.2, color='red', label='95% Confidence Interval')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated Capability')
    ax.set_title(f'Frontier-at-Release Trend (Annual Growth: {slope_per_year:.4f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


## OPTION 2: Using Bootstrap (for slope confidence intervals)
# You can use the existing utility from analysis_utils

from analysis_utils import bootstrap_slope_analysis

# After identifying frontier points:
if len(df_frontier) >= 2:
    # Bootstrap analysis on frontier points
    bootstrap_results = bootstrap_slope_analysis(
        df_frontier, 
        x_col='date_obj', 
        y_col='estimated_capability',
        n_bootstrap=10000
    )
    
    print(f"Bootstrap Results:")
    print(f"  Mean slope (per year): {bootstrap_results['mean_slope']:.4f}")
    print(f"  Std slope: {bootstrap_results['std_slope']:.4f}")
    print(f"  95% CI: [{bootstrap_results['ci_2_5']:.4f}, {bootstrap_results['ci_97_5']:.4f}]")
    print(f"  90% CI: [{bootstrap_results['ci_5']:.4f}, {bootstrap_results['ci_95']:.4f}]")
    
    # You can also plot the distribution of bootstrapped slopes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: histogram of bootstrap slopes
    ax1.hist(bootstrap_results['slopes'], bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(bootstrap_results['mean_slope'], color='red', linestyle='--', 
                label=f"Mean: {bootstrap_results['mean_slope']:.4f}")
    ax1.axvline(bootstrap_results['ci_2_5'], color='orange', linestyle='--', 
                label=f"95% CI: [{bootstrap_results['ci_2_5']:.4f}, {bootstrap_results['ci_97_5']:.4f}]")
    ax1.axvline(bootstrap_results['ci_97_5'], color='orange', linestyle='--')
    ax1.set_xlabel('Annual Growth Rate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap Distribution of Annual Growth Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: main trend plot (same as before)
    ax2.scatter(df_all['date_obj'], df_all['estimated_capability'], 
                alpha=0.25, s=20, label='All models', color='lightblue')
    ax2.scatter(df_frontier['date_obj'], df_frontier['estimated_capability'], 
                alpha=0.9, s=40, label=f'Top {TOP_N_AT_RELEASE} at release', color='blue')
    
    # Fit and plot
    X = (df_frontier['date_obj'] - df_frontier['date_obj'].min()).dt.days.values.reshape(-1, 1)
    y = df_frontier['estimated_capability'].values
    lr = LinearRegression()
    lr.fit(X, y)
    ax2.plot(df_frontier['date_obj'], lr.predict(X), 'r--', alpha=0.8, label='Frontier trend')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Estimated Capability')
    ax2.set_title(f'Frontier Trend (Growth: {bootstrap_results["mean_slope"]:.4f} ± {bootstrap_results["std_slope"]:.4f}/year)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

