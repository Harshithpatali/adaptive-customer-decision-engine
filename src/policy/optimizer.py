from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ACTIONS = ["No E-Mail", "Womens E-Mail", "Mens E-Mail"]

@dataclass(frozen=True)
class OptimizationConstraints:
    budget: float
    contact_capacity: Optional[int] = None
    action_capacity: Optional[Dict[str, int]] = None

def optimize_customer_policy(scores: pd.DataFrame, constraints: OptimizationConstraints) -> pd.DataFrame:
    required = {"customer_id", "action", "expected_value", "treatment_cost"}
    missing = required - set(scores.columns)
    if missing: raise ValueError(f'Missing optimizer columns: {sorted(missing)}')
    df = scores.copy().reset_index(drop=True)
    customers = df['customer_id'].unique()
    n = len(df)
    c = -(df['expected_value'].astype(float).to_numpy() - df['treatment_cost'].astype(float).to_numpy())
    customer_groups = list(df.groupby('customer_id', sort=False).groups.values())
    extra = int(constraints.contact_capacity is not None) + len(constraints.action_capacity or {}) + 1
    A = lil_matrix((len(customer_groups) + extra, n), dtype=float)
    lb = np.full(len(customer_groups) + extra, -np.inf); ub = np.full(len(customer_groups) + extra, np.inf)
    row = 0
    for idx in customer_groups:
        A[row, list(idx)] = 1.0; lb[row] = ub[row] = 1.0; row += 1
    if constraints.contact_capacity is not None:
        A[row, :] = (df['action'].to_numpy() != 'No E-Mail').astype(float); ub[row] = float(constraints.contact_capacity); row += 1
    for action, cap in (constraints.action_capacity or {}).items():
        A[row, :] = (df['action'].to_numpy() == action).astype(float); ub[row] = float(cap); row += 1
    A[row, :] = df['treatment_cost'].astype(float).to_numpy(); ub[row] = float(constraints.budget)
    result = milp(c=c, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)), constraints=LinearConstraint(A.tocsr(), lb, ub), options={'time_limit':120})
    if not result.success: raise RuntimeError(f'Policy optimization failed: {result.message}')
    selected = df.loc[np.asarray(result.x) > 0.5].copy()
    selected['net_expected_value'] = selected['expected_value'] - selected['treatment_cost']
    return selected[['customer_id','action','expected_value','treatment_cost','net_expected_value']].sort_values('net_expected_value', ascending=False).reset_index(drop=True)

def summarize_policy(selected: pd.DataFrame) -> dict:
    counts = selected['action'].value_counts().reindex(ACTIONS, fill_value=0)
    return {'customers': int(len(selected)), 'contacts': int((selected['action'] != 'No E-Mail').sum()), 'spend': float(selected['treatment_cost'].sum()), 'expected_value': float(selected['expected_value'].sum()), 'net_expected_value': float(selected['net_expected_value'].sum()), 'action_counts': {k:int(v) for k,v in counts.items()}}

def optimize_causal_incremental_policy(scores: pd.DataFrame, constraints: OptimizationConstraints) -> pd.DataFrame:
    """Optimize modeled causal incremental net value under budget/capacity constraints.

    Expected value must be on the same monetary scale as treatment_cost. The
    optimizer therefore maximizes incremental potential value relative to the
    no-email counterfactual, less treatment cost, rather than absolute revenue.
    """
    required = {"customer_id", "action", "net_incremental_value", "treatment_cost"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing causal optimizer columns: {sorted(missing)}")
    df = scores.copy().reset_index(drop=True)
    n = len(df)
    groups = list(df.groupby("customer_id", sort=False).groups.values())
    extra = int(constraints.contact_capacity is not None) + len(constraints.action_capacity or {}) + 1
    A = lil_matrix((len(groups) + extra, n), dtype=float)
    lb = np.full(len(groups) + extra, -np.inf)
    ub = np.full(len(groups) + extra, np.inf)
    row = 0
    for idx in groups:
        A[row, list(idx)] = 1.0
        lb[row] = ub[row] = 1.0
        row += 1
    if constraints.contact_capacity is not None:
        A[row, :] = (df.action.to_numpy() != "No E-Mail").astype(float)
        ub[row] = float(constraints.contact_capacity)
        row += 1
    for action, cap in (constraints.action_capacity or {}).items():
        A[row, :] = (df.action.to_numpy() == action).astype(float)
        ub[row] = float(cap)
        row += 1
    A[row, :] = df.treatment_cost.astype(float).to_numpy()
    ub[row] = float(constraints.budget)
    result = milp(
        c=-df.net_incremental_value.astype(float).to_numpy(),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options={"time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(f"Causal policy optimization failed: {result.message}")
    selected = df.loc[np.asarray(result.x) > 0.5].copy()
    return selected[["customer_id", "action", "incremental_value", "treatment_cost", "net_incremental_value"]].sort_values("net_incremental_value", ascending=False).reset_index(drop=True)
