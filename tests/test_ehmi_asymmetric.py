import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'SCPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)

import commotions  # noqa: E402
import sc_scenario  # noqa: E402
from sc_scenario import (CtrlType, DerivedAssumption, OptionalAssumption,  # noqa: E402
                         i_CONSTANT, i_PASS1ST, i_PASS2ND)
import sc_fitting  # noqa: E402
import sc_plot  # noqa: E402
import sc_run  # noqa: E402
import do_1_deterministic_fitting  # noqa: E402
import do_5_probabilistic_fitting  # noqa: E402
import do_10_get_empirical_HIKER_CITs  # noqa: E402


def test_oec_does_not_collide_with_existing_model_names():
    names = set()
    for b in do_1_deterministic_fitting.BASE_MODELS:
        for v in do_1_deterministic_fitting.MODEL_VARIANTS:
            names.add(b + v)
    for b in do_5_probabilistic_fitting.BASE_MODELS:
        for v in do_5_probabilistic_fitting.MODEL_VARIANTS:
            names.add(b + v)
    for b in sc_plot.BASE_MODELS:
        for v in sc_plot.MODEL_VARIANTS:
            names.add(b + v)
    for fname in (sc_fitting.RETAINED_DET_FNAME,
                  sc_fitting.RETAINED_PROB_FNAME,
                  sc_fitting.RETAINED_COMB_FNAME):
        for m in sc_fitting.load_results(fname, verbose=False):
            names.add(m.model)
    assert len(names) > 50
    for name in names:
        d = sc_scenario.get_assumptions_dict_from_string(name)
        assert d[OptionalAssumption.oEC] is False, name


def test_parse_and_displayed_ehmi_signals():
    assert sc_scenario.parse_ehmi_signal(None) == []
    assert sc_scenario.parse_ehmi_signal((1.5, 'yield')) == [(1.5, math.inf, 'yield')]
    assert sc_scenario.parse_ehmi_signal([(1.0, 3.0, 'proceed'), (2.5, 'yield')]) == [
        (1.0, 3.0, 'proceed'), (2.5, math.inf, 'yield')
    ]
    with pytest.raises(ValueError):
        sc_scenario.parse_ehmi_signal((1.0, 'blink'))
    with pytest.raises(ValueError):
        sc_scenario.parse_ehmi_signal((3.0, 2.0, 'yield'))
    with pytest.raises(TypeError):
        sc_scenario.parse_ehmi_signal('yield')

    sim = sc_scenario.SCSimulation(
        ctrl_types=(CtrlType.SPEED, CtrlType.ACCELERATION),
        widths=(0.8, 1.8), lengths=(0.8, 4.2),
        goal_positions=np.array([[0.0, 5.0], [-50.0, 0.0]]),
        initial_positions=np.array([[0.0, -4.0], [30.0, 0.0]]),
        initial_speeds=np.array([0.0, 10.0]),
        const_accs=(0.0, 0.0),
        end_time=4.0, time_step=0.1,
        ehmi_signals=(None, [(0.5, 2.0, 'proceed'), (1.5, 3.0, 'yield')]))
    veh = sim.agents[1]
    assert veh.get_displayed_ehmi(0.2) is None
    assert veh.get_displayed_ehmi(1.0) == 'proceed'
    assert veh.get_displayed_ehmi(1.7) == 'yield'  # latest overlapping signal wins
    assert veh.get_displayed_ehmi(2.5) == 'yield'
    assert veh.get_displayed_ehmi(3.0) is None


def test_oec_disabled_or_no_cue_is_bit_identical():
    scen = sc_fitting.HIKER_SCENARIOS['30_3_y']
    scen_ehmi = sc_fitting.HIKER_EHMI_SCENARIOS['30_3_y_e']
    params = {'T_delta': 40.0, 'beta_V': 50.0}

    sim_base = sc_fitting.construct_model_and_simulate_scenario(
        'oVAoBEvoAI', dict(params), scen, apply_stop_criteria=False)
    # Cue present on vehicle, but pedestrian does NOT have oEC -> bit-identical!
    sim_cue_ignored = sc_fitting.construct_model_and_simulate_scenario(
        'oVAoBEvoAI', dict(params), scen_ehmi, apply_stop_criteria=False)
    # oEC enabled, but NO cue displayed -> bit-identical activations/trajectories!
    sim_oec_no_cue = sc_fitting.construct_model_and_simulate_scenario(
        'oVAoBEvoAIoEC', dict(params), scen, apply_stop_criteria=False)

    for other in (sim_cue_ignored, sim_oec_no_cue):
        for i_ag in range(2):
            assert np.array_equal(
                sim_base.agents[i_ag].trajectory.pos,
                other.agents[i_ag].trajectory.pos, equal_nan=True)
            assert np.array_equal(
                sim_base.agents[i_ag].states.beh_probs_given_actions,
                other.agents[i_ag].states.beh_probs_given_actions, equal_nan=True)


def test_ehmi_evidence_low_pass_dynamics_and_standalone_oec():
    assumptions = sc_scenario.get_assumptions_dict(False, oVA=True, oEC=True)
    params, params_k = sc_scenario.get_default_params(oVA=True)
    params.T_E = 0.5
    params.beta_E = 4.0
    dt = 0.1
    sim = sc_scenario.SCSimulation(
        ctrl_types=(CtrlType.SPEED, CtrlType.ACCELERATION),
        widths=(0.8, 1.8), lengths=(0.8, 4.2),
        goal_positions=np.array([[0.0, 5.0], [-50.0, 0.0]]),
        initial_positions=np.array([[0.0, -5.0], [60.0, 0.0]]),
        initial_speeds=np.array([0.0, 10.0]),
        const_accs=(None, 0.0),
        end_time=3.0, time_step=dt,
        optional_assumptions=assumptions,
        params=params, params_k=params_k,
        ehmi_signals=(None, (0.5, 1.5, 'yield')))
    ped = sim.agents[0]
    assert ped.assumptions[DerivedAssumption.dBE] is True
    assert ped.params.beta_O == 0
    assert ped.params.beta_V == 0
    sim.run()

    ev_yield = ped.states.ehmi_evidence[i_PASS2ND, :]
    ev_proc = ped.states.ehmi_evidence[i_PASS1ST, :]
    assert np.all(ev_proc == 0.0)
    assert np.all(ev_yield[:5] == 0.0)  # t = 0.0 .. 0.4
    # At t = 0.5 (index 5), first step with cue -> f = 0.1 / 0.5 = 0.2
    assert ev_yield[5] == pytest.approx(0.2)
    # Rises monotonically during [0.5, 1.4]
    assert np.all(np.diff(ev_yield[5:15]) > 0)
    # Decays monotonically once cue turns off at t = 1.5 (index 15)
    assert ev_yield[15] == pytest.approx(0.8 * ev_yield[14])
    assert np.all(np.diff(ev_yield[14:25]) < 0)
    # Softmax belief for Pass 2nd > 0.5 while cue evidence > 0
    assert ped.states.beh_probs_given_actions[i_PASS2ND, ped.i_no_action, 10] > 0.8

    # Verify plotting with ehmi_evidence=True executes cleanly
    sim.do_plots(ehmi_evidence=True)
    plt.close('all')


def test_oec_increases_yield_belief_and_accelerates_hiker_crossing():
    scen_no_ehmi = sc_fitting.HIKER_SCENARIOS['30_2_y']
    scen_ehmi = sc_fitting.HIKER_EHMI_SCENARIOS['30_2_y_e']
    params = {'T_delta': 40.0, 'beta_V': 25.0, 'T_Of': 1.0, 'sigma_O': 0.5,
              'beta_E': 5.0, 'T_E': 0.3}

    sim_no = sc_fitting.construct_model_and_simulate_scenario(
        'oVAoBEooBEvoAIoEC', dict(params), scen_no_ehmi)
    sim_ehmi = sc_fitting.construct_model_and_simulate_scenario(
        'oVAoBEooBEvoAIoEC', dict(params), scen_ehmi)

    cit_no = sc_fitting.metric_hiker_cit(sim_no)
    cit_ehmi = sc_fitting.metric_hiker_cit(sim_ehmi)
    assert math.isfinite(cit_no) and math.isfinite(cit_ehmi)
    assert cit_ehmi < cit_no

    # Compare belief P(veh passes 2nd) right after yielding + eHMI onset
    ped_no = sim_no.agents[0]
    ped_ehmi = sim_ehmi.agents[0]
    t_check = scen_ehmi.veh_ehmi_time + 0.5
    idx_check = int(np.nonzero(sim_ehmi.time_stamps >= t_check)[0][0])
    p2_no = ped_no.states.beh_probs_given_actions[i_PASS2ND, ped_no.i_no_action, idx_check]
    p2_ehmi = ped_ehmi.states.beh_probs_given_actions[i_PASS2ND, ped_ehmi.i_no_action, idx_check]
    assert p2_ehmi > p2_no


def test_per_agent_asymmetric_models_and_parameters():
    sim_asym = sc_run.run_simulation(
        model=('oVAoBEvoAIoEC', 'oVA'),
        initial_cp_distances=(3.5, 35.0),
        initial_speeds=(1.3, 12.0),
        end_time=4.0,
        params_dict=({'T_delta': 50.0, 'beta_V': 60.0, 'beta_E': 4.0, 'T_E': 0.4},
                     {'T_delta': 15.0}),
        veh_ehmi_time=0.5)
    ped, veh = sim_asym.agents
    assert ped.assumptions[OptionalAssumption.oEC] is True
    assert ped.assumptions[OptionalAssumption.oBEv] is True
    assert veh.assumptions[OptionalAssumption.oEC] is False
    assert veh.assumptions[OptionalAssumption.oBEv] is False
    assert veh.assumptions[DerivedAssumption.dBE] is False
    assert ped.params.T_delta == 50.0
    assert veh.params.T_delta == 15.0

    # Symmetrical run with both agents using pedestrian spec differs from asymmetric run
    sim_sym = sc_run.run_simulation(
        model='oVAoBEvoAIoEC',
        initial_cp_distances=(3.5, 35.0),
        initial_speeds=(1.3, 12.0),
        end_time=4.0,
        params_dict={'T_delta': 50.0, 'beta_V': 60.0, 'beta_E': 4.0, 'T_E': 0.4},
        veh_ehmi_time=0.5)
    assert not np.array_equal(
        sim_asym.agents[1].trajectory.long_speed,
        sim_sym.agents[1].trajectory.long_speed)

    # Plotting works when only one agent estimates behaviours (dBE differs)
    sim_asym.do_plots(beh_probs=True, beh_activs=True, ehmi_evidence=True)
    plt.close('all')

    # Validation errors on bad tuple lengths / types
    with pytest.raises(ValueError):
        sc_run.run_simulation(
            model=('oVA', 'oVA', 'oVA'),
            initial_cp_distances=(3.5, 35.0),
            initial_speeds=(1.3, 12.0),
            end_time=1.0,
            params_dict={'T_delta': 30.0})
    with pytest.raises(ValueError):
        sc_scenario.get_per_agent_values([{}, {}, {}], 'optional_assumptions', dict)
    with pytest.raises(TypeError):
        sc_scenario.get_per_agent_values(123, 'optional_assumptions', dict)


def test_hiker_ehmi_scenarios_and_empirical_extraction(tmp_path, monkeypatch):
    assert len(sc_fitting.HIKER_SCENARIOS) == 24
    assert len(sc_fitting.HIKER_EHMI_SCENARIOS) == 12
    assert '25_3_y_e' in sc_fitting.HIKER_EHMI_SCENARIOS
    assert sc_fitting.HIKER_EHMI_SCENARIOS['25_3_y_e'].veh_ehmi_time is not None
    with pytest.raises(ValueError):
        sc_fitting.HIKERScenario('bad', 25, 3, veh_yielding=False, veh_ehmi=True)

    # Empirical CSV extraction with and without eHMI
    hiker_df = pd.read_csv(os.path.join(REPO, 'SCPaper', 'data', 'hiker_cts.csv'))
    data_no_ehmi, n_no = do_10_get_empirical_HIKER_CITs.extract_hiker_cit_data(
        hiker_df, include_ehmi=False)
    data_with_ehmi, n_with = do_10_get_empirical_HIKER_CITs.extract_hiker_cit_data(
        hiker_df, include_ehmi=True)
    assert len(data_no_ehmi) == 24
    assert len(data_with_ehmi) == 36
    assert n_with > n_no > 0
    for k in data_no_ehmi:
        assert data_no_ehmi[k].equals(data_with_ehmi[k])
    assert len(data_with_ehmi['25_3_y_e']) > 0

    # ParameterSearch registers beta_E and T_E when oEC is enabled
    monkeypatch.setattr(sc_fitting, 'FIT_RESULTS_FOLDER', str(tmp_path) + os.sep)
    scen = {'s': sc_fitting.SCPaperScenario('s', initial_ttcas=(2.0, 5.0),
                                            end_time=0.4,
                                            metric_names=('ped_av_speed',),
                                            veh_ehmi_time=0.0)}
    assumptions = sc_scenario.get_assumptions_dict_from_string('oVAoEC')
    ps = sc_fitting.SCPaperParameterSearch(
        'oVAoEC', scen, assumptions,
        sc_fitting.DEFAULT_PARAMS, sc_fitting.get_default_params_k('oVAoEC'),
        {'T_delta': (30.0,), 'beta_E': (2.0, 5.0), 'T_E': (0.4,)},
        parallel=False, file_name_format='test_fit_%s.pkl')
    assert ps.param_names == ('T_delta', 'beta_E', 'T_E')
    assert ps.results.params_matrix.shape == (2, 3)
