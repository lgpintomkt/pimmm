# Physics-Informed Marketing Mix Modeling (`pimmm`)

![pimmm banner](banner.svg)
**Physics-Informed Marketing Mix Modeling (`pimmm`)** is a Python library that unifies machine learning response surfaces with continuous-time innovation-diffusion physics to forecast long-term product adoption and return on ad spend (ROAS).

By embedding the **Generalized Bass Model (GBM)** Ordinary Differential Equations directly into empirical response pipelines, `pimmm` resolves the early-stage non-identifiability problem. This allows robust trajectory forecasting even when training on sparse early-adoption data.

---

## 🌟 Key Features

* **Physics-Informed ODE Dynamics:** Enforces continuous-time innovation ($p$) and imitation ($q$) ODE constraints over marketing spend drivers.
* **Open Architecture & ML Integration:** Compatible with standard `scikit-learn` estimators (Gradient Boosting, Random Forests, Neural Networks).
* **Foundation Model & Enterprise Support:** Built-in adapter wrappers for **Tabular Foundation Models (TabFM)** and enterprise MMM platforms like **Google Meridian**.
* **Scikit-Learn API Compliance:** Adheres strictly to standard `.fit()` and `.predict()` paradigms for seamless integration into existing MLOps pipelines.
* **Early-Stage Robustness:** Designed specifically to bound long-term adoption trajectories when data is limited to early adoption stages (under 20% market penetration).

---

## 🏗️ Architecture Overview

Unlike traditional empirical response models that overfit or fail to capture saturation ceilings, `pimmm` operates as a hybrid architecture:

$$\frac{dF}{dt} = (p + q F(t))(1 - F(t)) \cdot g(x(t))$$

1. **Empirical Layer:** Machine learning estimators or MMM frameworks (such as Meridian or TabFM) compute the latent marketing effort $g(x(t))$ from high-dimensional spend signals.
2. **Physics Layer:** The ODE solver integrates the latent effort continuous-time trajectory to enforce natural structural growth, imitation effects, and market saturation caps ($F(t) \le 1$).

---

## 🚀 Quickstart

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from pimmm import PhysicsInformedMMM

# 1. Load your spend inputs (X) and adoption trajectories (y)
X_spend = ... # Marketing spend features
y_adoption = ... # Observed market penetration [0, 1]

# 2. Instantiate PI-MMM with any Scikit-Learn driver
model = PhysicsInformedMMM(
    response_model=GradientBoostingRegressor(),
    dt=0.5
)

# 3. Fit on early-stage adoption data
model.fit(X_spend, y_adoption)

# 4. Forecast continuous adoption S-curves
predictions = model.predict(X_spend_future)

# 5. Extract recovered physical constants
print(model.get_params_summary())
# Output: {'Innovation (p)': 0.012, 'Imitation (q)': 0.341, 'Scale Factor (gamma)': 0.82}
```
---

## 📦 Installation

```bash
pip install git+https://github.com/lgpintomkt/pimmm.git
```

## 📚 Academic Foundations & References

`pimmm` is developed as part of ongoing PhD research in Mathematics at the **University of Évora, Portugal** (IIFA / CIMA / DMAT). The core architecture, reduction hierarchy, and effort-time equivalences implemented in this package are formally established across the following works:

### Peer-Reviewed Publications
* Pinto, L. G., Cavique, L., Gomes, O., & Santos, J. M. A. (2024). Inhomogeneous Marketing Mix Diffusion. In F. Botta, M. Macedo, H. Barbosa, & R. Menezes (Eds.), *Complex Networks XV. CompleNet-Live 2024* (Springer Proceedings in Complexity, pp. 29–41). Springer, Cham. https://doi.org/10.1007/978-3-031-57515-0_3
* Pinto, L. G., Cavique, L., & Santos, J. M. A. (2022). Marketing Mix and New Product Diffusion Models. *Procedia Computer Science*, 204, 885–890. https://doi.org/10.1016/j.procs.2022.08.107
* Henriques, R., & Pinto, L. (2023). A novel evaluation framework for recommender systems in big data environments. Expert Systems with Applications, 231, 120659. https://doi.org/10.1016/j.eswa.2023.120659
