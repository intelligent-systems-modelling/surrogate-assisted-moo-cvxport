from pymoo.core.callback import Callback
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.core.problem import Problem
from pymoo.factory import get_termination
from copy import deepcopy
import time
import numpy as np
import pandas as pd
import cvxportfolio as cp
import warnings
from os.path import exists
from typing import Union

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.evaluator import set_cv

warnings.filterwarnings('ignore')

EVALUATIONS = 510


def get_pareto_frontier(table, flatten=True, **kwargs):
    pf = table.loc[table.is_pareto, ['Return', 'Risk']]
    pf.Return *= -1
    return pf.values


def run_experiment(experiment_name: str,
                   algorithm,
                   repeats: Union[int, float],
                   start_t: str,
                   end_t: str,
                   data_dir: str,
                   tmp_folder: str,
                   ground_truth_file_path: str,
                   multi_period: bool = False,
                   overwrite: bool = False):
    ground_results_table = pd.read_csv(ground_truth_file_path)
    pf = get_pareto_frontier(ground_results_table)
    proto = algorithm
    if repeats == 0:
        cvxportfolio_problem = CVXPortfolioProblem(start_t,
                                                   end_t,
                                                   multi_period=multi_period,
                                                   data_dir=data_dir,
                                                   ground_truth_file_path=ground_truth_file_path)
        algorithm = deepcopy(proto)
        termination = get_termination("n_eval", EVALUATIONS)

        algorithm.setup(
            cvxportfolio_problem,
            termination=termination,
            callback=AttachResult(),
            verbose=True,
            save_history=True,
        )

    for i in range(repeats):
        cvxportfolio_problem = CVXPortfolioProblem(start_t,
                                                   end_t,
                                                   multi_period=multi_period,
                                                   data_dir=data_dir,
                                                   ground_truth_file_path=ground_truth_file_path)
        algorithm = deepcopy(proto)
        termination = get_termination("n_eval", EVALUATIONS)

        algorithm.setup(
            cvxportfolio_problem,
            termination=termination,
            callback=AttachResult(),
            verbose=True,
            save_history=True,
        )

        checkpoint_path = f"{tmp_folder}/checkpoint_{experiment_name}_{i}.npy"
        checkpoint_exists = exists(checkpoint_path)

        if overwrite or not checkpoint_exists:
            while algorithm.has_next():
                algorithm.next()
                pf_opt = algorithm.opt.get("F")
                hv, gdp, igdp = (Hypervolume(normalize=False, ref_point=[0.0, 40.0]).do(pf_opt),
                                 GDPlus(normalize=False,
                                        pf=pf).do(pf_opt), IGDPlus(normalize=False,
                                                                   pf=pf).do(pf_opt))
                atSR = Hypervolume(normalize=False, ref_point=[0.0, 100.0]).do(pf_opt)
                atSR /= Hypervolume(normalize=False, ref_point=[0.0, 100.0]).do(pf)

                print(hv, gdp, igdp, atSR)

            np.save(checkpoint_path, algorithm)

        (checkpoint,) = np.load(checkpoint_path, allow_pickle=True).flatten()

        print(f"Loaded Checkpoint {i}:", checkpoint)

        pf_opt = checkpoint.opt.get("F")
        hv, gdp, igdp = (Hypervolume(normalize=False, ref_point=[0.0, 40.0]).do(pf_opt),
                         GDPlus(normalize=False, pf=pf).do(pf_opt), IGDPlus(normalize=False,
                                                                            pf=pf).do(pf_opt))
        print(hv, gdp, igdp)


def print_results(checkpoints, ground_truth_file_path, ref_point=None):
    ground_results_table = pd.read_csv(ground_truth_file_path)
    pf = get_pareto_frontier(ground_results_table)
    if ref_point is None:
        ref_point = [0.0, 40.0]
    hv_sum, gdp_sum, igdp_sum = 0, 0, 0
    hv_max, gdp_min, igdp_min = None, None, None
    df_max_ret, df_max_ris = None, None
    for checkpoint in checkpoints:
        pf_opt = checkpoint.opt.get("F")
        df_pf_opt = pd.DataFrame(pf_opt.copy(), columns=['Return', 'Risk'])
        df_pf_opt.Return *= -1
        if df_max_ret is None:
            df_max_ret = df_pf_opt.max().Return
            df_max_ris = df_pf_opt.max().Risk
        else:
            df_max_ret = np.max([df_pf_opt.max().Return, df_max_ret])
            df_max_ris = np.max([df_pf_opt.max().Risk, df_max_ris])
        hv, gdp, igdp = (Hypervolume(normalize=False, ref_point=ref_point).do(pf_opt),
                         GDPlus(normalize=False, pf=pf).do(pf_opt), IGDPlus(normalize=False,
                                                                            pf=pf).do(pf_opt))
        hv_sum += hv
        if hv_max is None or hv > hv_max:
            df_hv_max = pd.DataFrame(pf_opt, columns=['Return', 'Risk'])
            df_hv_max.Return *= -1
        hv_max = hv if hv_max is None else max(hv_max, hv)
        gdp_sum += gdp
        gdp_min = gdp if gdp_min is None else min(gdp_min, gdp)
        igdp_sum += igdp
        igdp_min = igdp if igdp_min is None else min(igdp_min, igdp)


def get_results(checkpoints, ground_truth_file_path, ref_point=None):
    ground_results_table = pd.read_csv(ground_truth_file_path)
    pf = get_pareto_frontier(ground_results_table)
    if ref_point is None:
        ref_point = [0.0, 40.0]
    hv_sum, gdp_sum, igdp_sum = 0, 0, 0
    hv_sum_2, gdp_sum_2, igdp_sum_2 = 0, 0, 0
    hv_max, gdp_min, igdp_min = None, None, None
    df_max_ret, df_max_ris = None, None
    for checkpoint in checkpoints:
        pf_opt = checkpoint.opt.get("F")
        df_pf_opt = pd.DataFrame(pf_opt.copy(), columns=['Return', 'Risk'])
        df_pf_opt.Return *= -1
        if df_max_ret is None:
            df_max_ret = df_pf_opt.max().Return
            df_max_ris = df_pf_opt.max().Risk
        else:
            df_max_ret = np.max([df_pf_opt.max().Return, df_max_ret])
            df_max_ris = np.max([df_pf_opt.max().Risk, df_max_ris])
        hv, gdp, igdp = (Hypervolume(normalize=False, ref_point=ref_point).do(pf_opt),
                         GDPlus(normalize=False, pf=pf).do(pf_opt), IGDPlus(normalize=False,
                                                                            pf=pf).do(pf_opt))
        hv_sum += hv
        hv_sum_2 += hv**2
        if hv_max is None or hv > hv_max:
            df_hv_max = pd.DataFrame(pf_opt, columns=['Return', 'Risk'])
            df_hv_max.Return *= -1
    #         df_hv_max.to_csv('csv_results/%s.csv'%(experiment_name))
        hv_max = hv if hv_max is None else max(hv_max, hv)
        gdp_sum += gdp
        gdp_sum_2 += gdp**2
        gdp_min = gdp if gdp_min is None else min(gdp_min, gdp)
        igdp_sum += igdp
        igdp_sum_2 += igdp**2
        igdp_min = igdp if igdp_min is None else min(igdp_min, igdp)
    hv_mu, gdp_mu, igdp_mu = (hv_sum / len(checkpoints), gdp_sum / len(checkpoints),
                              igdp_sum / len(checkpoints))
    hv_std = ((hv_sum_2 - 2 * hv_sum * hv_mu) / len(checkpoints) + hv_mu**2)**0.5
    gdp_std = ((gdp_sum_2 - 2 * gdp_sum * gdp_mu) / len(checkpoints) + gdp_mu**2)**0.5
    igdp_std = ((igdp_sum_2 - 2 * igdp_sum * igdp_mu) / len(checkpoints) + igdp_mu**2)**0.5
    return df_max_ret, df_max_ris, hv_max, hv_mu, hv_std, gdp_min, gdp_mu, gdp_std, igdp_min, igdp_mu, igdp_std


class CVXPortfolioProblem(Problem):

    def __init__(self, start_t, end_t, data_dir, ground_truth_file_path, multi_period=False):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([-1.0, 5.5, -1.0]),
                         xu=np.array([3.0, 8.0, 3.0]))
        self.start_t = start_t
        self.end_t = end_t
        self.MPO = multi_period
        self.table = pd.read_csv(ground_truth_file_path)
        self._load_data(data_dir, start_t, end_t)

    def _load_data(self, datadir, start_t, end_t):
        sigmas = pd.read_csv(datadir + 'sigmas.csv.gz', index_col=0, parse_dates=[0]).iloc[:, :-1]
        returns = pd.read_csv(datadir + 'returns.csv.gz', index_col=0, parse_dates=[0])
        volumes = pd.read_csv(datadir + 'volumes.csv.gz', index_col=0, parse_dates=[0]).iloc[:, :-1]

        self.w_b = pd.Series(index=returns.columns, dtype=np.float32, data=1.0)
        self.w_b.USDOLLAR = 0.
        self.w_b /= sum(self.w_b)

        sigmas = sigmas[start_t:end_t]
        self.returns = returns[start_t:end_t]
        volumes = volumes[start_t:end_t]

        simulated_tcost = cp.TcostModel(half_spread=0.0005 / 2.,
                                        nonlin_coeff=1.,
                                        sigma=sigmas,
                                        volume=volumes)
        simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
        self.simulator = cp.MarketSimulator(returns,
                                            costs=[simulated_tcost, simulated_hcost],
                                            market_volumes=volumes,
                                            cash_key='USDOLLAR')

        return_estimate = pd.read_csv(datadir + 'return_estimate.csv.gz',
                                      index_col=0,
                                      parse_dates=[0]).dropna()
        volume_estimate = pd.read_csv(datadir + 'volume_estimate.csv.gz',
                                      index_col=0,
                                      parse_dates=[0]).dropna()
        sigma_estimate = pd.read_csv(datadir + 'sigma_estimate.csv.gz',
                                     index_col=0,
                                     parse_dates=[0]).dropna()

        self.return_estimate = return_estimate[start_t:end_t]
        volume_estimate = volume_estimate[start_t:end_t]
        sigma_estimate = sigma_estimate[start_t:end_t]

        all_return_estimates = {}
        for i, t in enumerate(returns.index[:-1]):
            all_return_estimates[(t, t)] = return_estimate.loc[t]
            tp1 = returns.index[i + 1]
            all_return_estimates[(t, tp1)] = return_estimate.loc[tp1]
        self.returns_forecast = cp.MPOReturnsForecast(all_return_estimates)

        self.optimization_tcost = cp.TcostModel(half_spread=0.0005 / 2.,
                                                nonlin_coeff=1.,
                                                sigma=sigma_estimate,
                                                volume=volume_estimate)
        self.optimization_hcost = cp.HcostModel(borrow_costs=0.0001)

        risk_data = pd.HDFStore(datadir + 'risk_model.h5', mode='a')
        self.risk_model = cp.FactorModelSigma(risk_data.exposures, risk_data.factor_sigma,
                                              risk_data.idyos)
        risk_data.close()

    def _calc_pareto_front(self, *args, **kwargs):
        pf = self.table.loc[self.table.is_pareto, ['Return', 'Risk']]
        pf.Return *= -1
        return pf.values

    def _calc_pareto_set(self, *args, **kwargs):
        return self.table.loc[
            self.table.is_pareto,
            [r'$\gamma^\mathrm{risk}$', r'$\gamma^\mathrm{trade}$', r'$\gamma^\mathrm{hold}$'
            ]].values

    def _evaluate(self, P, out, *args, **kwargs):
        """ P with n rows and m columns as an input.
            Each row represents an individual, and each column an optimization variable.
        """
        policies = {}
        P_rescale = []
        for gamma_risk, gamma_tcost, gamma_holding in P:
            gamma_risk = np.power(10.0, gamma_risk)
            gamma_holding = np.power(10.0, gamma_holding)
            P_rescale.append((gamma_risk, gamma_tcost, gamma_holding))
            if self.MPO:
                policies[(gamma_risk, gamma_tcost, gamma_holding)] = \
                    cp.MultiPeriodOpt(return_forecast=self.returns_forecast,
                                      costs=[
                                          gamma_risk * self.risk_model,
                                          gamma_tcost * self.optimization_tcost,
                                          gamma_holding * self.optimization_hcost],
                                      constraints=[cp.LeverageLimit(3)],
                                      trading_times=list(self.returns.index),
                                      lookahead_periods=2,
                                      terminal_weights=None)
            else:
                policies[(gamma_risk, gamma_tcost, gamma_holding)] = cp.SinglePeriodOpt(
                    self.return_estimate,
                    costs=[
                        gamma_risk * self.risk_model, gamma_tcost * self.optimization_tcost,
                        gamma_holding * self.optimization_hcost
                    ],
                    constraints=[cp.LeverageLimit(3)])

        policy_keys = policies.keys()
        policy_values = [policies[k] for k in policy_keys]

        results = self.simulator.run_multiple_backtest(1E8 * self.w_b,
                                                       start_time=self.start_t,
                                                       end_time=self.end_t,
                                                       policies=policy_values,
                                                       parallel=True)

        policy_results = dict(zip(policy_keys, results))

        Return = [-policy_results[k].excess_returns.mean() * 100 * 250 for k in P_rescale]
        Risk = [policy_results[k].excess_returns.std() * 100 * np.sqrt(250) for k in P_rescale]

        out["F"] = np.column_stack([Return, Risk])


class AttachResult(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data = {'end_time': []}

    def notify(self, algorithm, **kwargs):
        self.data['end_time'].append(time.time())


class CSVAlgorithm(Algorithm):

    def __init__(self, csv_file, **kwargs):
        super().__init__(**kwargs)
        self.csv_file = csv_file

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.results_table = pd.read_csv(self.csv_file)
        self.results_table.Return *= -1

    def _initialize_advance(self, infills=None, **kwargs):
        pop_F = self.results_table.loc[:, [r'Return', r'Risk']].values

        pop_X = self.results_table.loc[:, [
            r'$\gamma^\mathrm{risk}$', r'$\gamma^\mathrm{trade}$', r'$\gamma^\mathrm{hold}$'
        ]].values

        self.pop = Population.new("X", pop_X, "F", pop_F, "evaluated", True)
        set_cv(self.pop)
