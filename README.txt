SMART GRID POSG + DATA4CYBER
============================

This package contains the corrected Streamlit POSG game and the Data4Cyber
replay dataset.

FILES
-----
app.py
posg_engine.py
posg_models.py
posg_types.py
data4cyber_replay.py
data4cyber_dataset(1).zip
requirements.txt

KEY PAYOFF CORRECTION
---------------------
No separate payoff_parameters.py file is required.
All payoff parameters are maintained in posg_models.py.

The correction separates:
  1. DATA4CYBER_DAMAGE_WEIGHTS
     Telemetry-component weights for SOC, PV, load, frequency, and price.

  2. ACTION_DAMAGE
     Stage/attacker-action damage values used by the strategic payoff matrix.

compute_payoff_matrix() now uses ACTION_DAMAGE directly, preventing the
telemetry-weight dictionary from being mistaken for an attacker-action
vector.

The payoff matrix remains:
  U_D = W1*Prevention
        - W2*P_success*ActionDamage
        - W3*DefenseCost
        - W4*OperationalDisruption

Data4Cyber telemetry-derived damage is still available through Damage().

RUN
---
1. Extract this ZIP.
2. Open a terminal in the extracted folder.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the game:

   streamlit run app.py

The app will automatically use data4cyber_dataset(1).zip when it is in the
same folder as app.py. You can also upload a Data4Cyber ZIP from the sidebar.

The game consumes actual Data4Cyber CSV rows and maps defender-visible
telemetry into POSG observations before running the Bayesian belief update.
The dataset attack_phase fields are retained as research ground truth and
are not used directly to construct the defender belief.
