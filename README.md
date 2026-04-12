# Credit Risk Scorecard Development & Validation

End-to-end credit risk scorecard pipeline following industry-standard methodology used in bank risk modelling teams. Built on a 32,500+ borrower retail lending portfolio across USA, UK, and Canada.

## Key results

| Metric | LR Scorecard (production) | XGBoost + SHAP (benchmark) |
|--------|--------------------------|---------------------------|
| Gini (test) | 73.7% | 86.4% |
| KS (test) | 59.7% | 71.7% |
| Gini gap (train-test) | 2.0% ✓ Stable | 4.6% ✓ Acceptable |
| PSI | 0.014 ✓ No drift | 0.001 ✓ No drift |
| Regulatory accepted | Yes | Benchmark only |

**Recommendation:** Deploy LR scorecard for production. The 12.7-point Gini gap is the "cost of interpretability": acceptable given regulatory requirements. XGBoost validates the LR feature selection and serves as a monitoring benchmark.

## Methodology

The pipeline covers the full model lifecycle as practiced in bank CRM (Credit Risk Modelling) teams:

**Development**

1. Data quality assessment and cleaning (outlier removal, grade-conditional imputation)
2. WoE/IV feature selection: industry standard for credit scorecards
3. Logistic regression scorecard with points-based calibration (base 600, PDO 20)
4. XGBoost benchmark with SHAP global and local explanations

**Validation**

5. Gini, KS, CAP curve, PSI, calibration curve
6. 5-fold stratified cross-validation (CV AUC: 0.875 ± 0.006)
7. A/B comparison table: traditional vs ML approach

**Regulatory & Accounting**

8. Basel III IRB capital calculation using the ASRF framework
9. IFRS 9 three-stage ECL classification (12-month vs lifetime PD)
10. Macro-conditioned stress testing across four scenarios

**Monitoring**

11. Simulated 12-month production monitoring (monthly Gini/PSI tracking)
12. Feature-level PSI for early drift detection
13. Monitoring decision framework (monthly/quarterly/annual cadence)

**Other**

14. Early warning indicator framework (red-flag scoring)
15. Fairness assessment: gender, education, marital status have IV ≈ 0

## Approach comparison

This project explicitly builds and compares both traditional and ML approaches:

| | Traditional (Approach 1) | ML-based (Approach 2) |
|---|---|---|
| **Method** | WoE + Logistic Regression | XGBoost + SHAP |
| **Used by** | Banks (FICO, Experian, CBS) | Fintechs, research |
| **Strength** | Interpretable, regulatorily accepted, stable | Higher discrimination, captures non-linearity |
| **Weakness** | Misses non-linear patterns | Black-box, regulatory skepticism |
| **In this project** | Production model | Benchmark |

## Project structure

```
credit-risk-scorecard/
├── README.md
├── requirements.txt
├── credit_risk_scorecard.py          # Full script (runs end-to-end)
├── data/
│   └── Credit_Risk_Dataset.xlsx      # 32,581 borrowers, 29 features
├── notebooks/
│   └── credit_risk_scorecard.ipynb   # Jupyter notebook version
└── outputs/
    ├── fig1_default_rate_segments.png
    ├── fig2_correlation_heatmap.png
    ├── fig3_information_value.png
    ├── fig4_shap_global_importance.png
    ├── fig5_shap_beeswarm.png
    ├── fig6_shap_local_explanation.png
    ├── fig7_model_validation.png
    ├── fig8_basel_irb_capital.png
    ├── fig9_ifrs9_staging.png
    ├── fig10_stress_testing.png
    ├── fig11_early_warning.png
    ├── fig12_monitoring_dashboard.png
    ├── model_metrics.csv
    ├── information_value.csv
    ├── basel_capital_by_band.csv
    ├── ifrs9_staging.csv
    └── stress_test_results.csv
```

## Quick start

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy xgboost shap openpyxl
python credit_risk_scorecard.py
```

Or open `notebooks/credit_risk_scorecard.ipynb` in Jupyter.

## Sample outputs

### Information Value: feature selection
![IV](outputs/fig3_information_value.png)

### SHAP global feature importance
![SHAP](outputs/fig4_shap_global_importance.png)

### Model validation (ROC, KS, calibration)
![Validation](outputs/fig7_model_validation.png)

### Basel III IRB capital by score band
![Basel](outputs/fig8_basel_irb_capital.png)

### IFRS 9 stage classification
![IFRS9](outputs/fig9_ifrs9_staging.png)

### Stress testing
![Stress](outputs/fig10_stress_testing.png)

### Production monitoring dashboard
![Monitoring](outputs/fig12_monitoring_dashboard.png)

## References

1. Basel Committee on Banking Supervision (1999). *Credit Risk Modelling: Current Practices and Applications.*
2. Noguer i Alonso, M. & Sun, Y. (2025). *Credit Risk Modeling for Financial Institutions.* SSRN.
3. Golec, M. & AlabdulJalil, M. (2025). *Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy.* arXiv:2506.04290.
4. Hlongwane, R., Ramabao, K. & Mongwe, W. (2024). *A novel framework for enhancing transparency in credit scoring: Leveraging Shapley values for interpretable credit scorecards.* PLoS ONE 19(8).

## License

This project is for educational and portfolio purposes.
