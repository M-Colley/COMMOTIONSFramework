import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'SCPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)

import commotions  # noqa: E402
import sc_scenario  # noqa: E402
import sc_scenario_helper  # noqa: E402
import sc_scenario_perception  # noqa: E402
import sc_fitting  # noqa: E402
import sc_run  # noqa: E402


# =============================================================================
# Part A: Cooperative MRO & AgentWithIntermittentActions tests
# =============================================================================

class _ExtraMixin(commotions.BaseAgent):
    """Cooperative mixin used to verify diamond multiple inheritance."""
    def __init__(self, *args, extra_tag=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_tag = extra_tag


class _GoalThenIntermittent(commotions.AgentWithGoal,
                            commotions.AgentWithIntermittentActions):
    pass


class _IntermittentThenGoal(commotions.AgentWithIntermittentActions,
                            commotions.AgentWithGoal):
    pass


class _DiamondAgent(_ExtraMixin,
                    commotions.AgentWithGoal,
                    commotions.AgentWithIntermittentActions):
    pass


def test_base_and_goal_agent_positional_compatibility():
    """commotions.__main__ style positional and keyword construction."""
    sim = commotions.Simulation(0, 2.0, 0.1)
    ag_a = commotions.BaseAgent(
        'A', sim,
        commotions.KinematicState(pos=np.array((-5.0, -5.0)), long_speed=1.0),
        commotions.ActionState(yaw_rate=0.1),
        True, 'b')
    ag_b = commotions.AgentWithGoal(
        'B', sim, np.array((0.0, 5.0)),
        commotions.KinematicState(pos=np.array((0.0, -5.0)), long_speed=1.2,
                                  yaw_angle=None),
        commotions.ActionState(long_acc=0.5),
        False, 'r')
    assert ag_a.i_agent == 0
    assert ag_b.i_agent == 1
    assert ag_b.can_reverse is False
    assert math.isclose(ag_b.trajectory.yaw_angle[0], math.pi / 2)

    sim.run()
    assert np.all(np.isfinite(ag_a.trajectory.pos))
    assert np.all(np.isfinite(ag_b.trajectory.pos))
    assert ag_b.trajectory.pos[1, -1] > ag_b.trajectory.pos[1, 0]

    # Line intersection helper from commotions.__main__
    pt = commotions.get_intersection_of_lines(
        np.array((0.0, 0.0)), np.array((1.0, 0.0)),
        np.array((0.5, -1.0)), np.array((0.5, 1.0)))
    assert np.allclose(pt, np.array((0.5, 0.0)))
    assert commotions.get_intersection_of_lines(
        np.array((-1.0, 0.0)), np.array((-1.0, 1.0)),
        np.array((-2.0, 0.0)), np.array((-2.0, 1.0))) is None


def test_cooperative_mro_and_unexpected_kwargs():
    sim = commotions.Simulation(0, 2.0, 0.1)  # 20 steps

    # Order 1: (AgentWithGoal, AgentWithIntermittentActions)
    ag1 = _GoalThenIntermittent(
        'G1', sim, np.array((10.0, 0.0)),
        commotions.KinematicState(pos=np.array((0.0, 0.0)), long_speed=1.0,
                                  yaw_angle=None),
        can_reverse=False, action_duration=0.5, prediction_time=0.8)
    assert ag1.n_action_time_steps == 5
    assert ag1.n_prediction_time_steps == 8
    assert ag1.n_actions_vector_length == 20 + 8
    assert ag1.action_long_accs.shape == (28,)
    assert ag1.action_yaw_rates.shape == (28,)
    assert math.isclose(ag1.trajectory.yaw_angle[0], 0.0)

    # Order 2: (AgentWithIntermittentActions, AgentWithGoal) with positional args
    ag2 = _IntermittentThenGoal(
        'G2', sim, np.array((0.0, 10.0)),
        commotions.KinematicState(pos=np.array((0.0, 0.0)), long_speed=2.0,
                                  yaw_angle=None),
        action_duration=0.4)
    assert ag2.n_action_time_steps == 4
    assert ag2.n_prediction_time_steps == 4
    assert ag2.n_actions_vector_length == 24
    assert math.isclose(ag2.trajectory.yaw_angle[0], math.pi / 2)

    # Diamond inheritance with an extra cooperative mixin
    ag3 = _DiamondAgent(
        'D', sim, np.array((-5.0, 0.0)),
        commotions.KinematicState(pos=np.array((5.0, 0.0)), long_speed=1.0,
                                  yaw_angle=None),
        action_duration=0.3, extra_tag='custom')
    assert ag3.extra_tag == 'custom'
    assert ag3.n_action_time_steps == 3

    # Deferred setup_intermittent_actions (as used by SCAgent)
    ag4 = commotions.AgentWithIntermittentActions('I', sim, None)
    assert not hasattr(ag4, 'action_long_accs')
    ag4.setup_intermittent_actions(0.5)
    assert ag4.n_action_time_steps == 5
    assert ag4.n_prediction_time_steps == 5

    # Specifying prediction_time without action_duration raises ValueError
    with pytest.raises(ValueError):
        commotions.AgentWithIntermittentActions(
            'bad', sim, None, prediction_time=0.5)

    # Unexpected keyword arguments raise TypeError via BaseAgent -> object
    with pytest.raises(TypeError):
        commotions.BaseAgent('bad_kw', sim, None, unknown_kwarg=123)
    with pytest.raises(TypeError):
        _GoalThenIntermittent('bad_kw2', sim, np.zeros(2), None,
                              action_duration=0.5, bogus=True)

    # SCAgent inherits cooperatively from both classes
    assert issubclass(sc_scenario.SCAgent, commotions.AgentWithGoal)
    assert issubclass(sc_scenario.SCAgent,
                      commotions.AgentWithIntermittentActions)


def test_intermittent_action_methods():
    sim = commotions.Simulation(0, 2.0, 0.1)  # 20 steps
    ag = _GoalThenIntermittent(
        'A', sim, np.array((10.0, 0.0)),
        commotions.KinematicState(pos=np.array((0.0, 0.0)), long_speed=1.0,
                                  yaw_angle=0.0),
        can_reverse=False, action_duration=0.5, prediction_time=0.5)

    sim.state.set_time_step(2)
    ag.add_uniform_action(-2.0)  # steps 2..6
    expected = np.zeros(ag.n_actions_vector_length)
    expected[2:7] = -2.0
    assert np.array_equal(ag.action_long_accs, expected)

    # Moving: clear_future_neg_accs_if_stopped leaves negative accelerations
    ag.clear_future_neg_accs_if_stopped(long_speed=0.5)
    assert np.array_equal(ag.action_long_accs, expected)

    # Stopped and can_reverse=False: clears from current step onward only
    sim.state.set_time_step(4)
    ag.clear_future_neg_accs_if_stopped(long_speed=0.0)
    assert np.array_equal(ag.action_long_accs[:4], np.array([0.0, 0.0, -2.0, -2.0]))
    assert np.all(ag.action_long_accs[4:] == 0.0)

    # Ramp action on custom array vs default array
    sim.state.set_time_step(0)
    ag.action_long_accs[:] = 0.0
    ag.add_ramp_action(1.0)
    ref = np.zeros(ag.n_actions_vector_length)
    commotions.add_linear_ramp_action_to_array(ref, 1.0, 5, 0)
    assert np.array_equal(ag.action_long_accs, ref)

    # Predicted state with array and scalar inputs
    pred_arr = ag.get_predicted_state()
    direct_arr = ag.get_future_kinematic_state(
        ag.action_long_accs, yaw_rate=ag.action_yaw_rates,
        n_time_steps_to_advance=ag.n_prediction_time_steps)
    assert np.allclose(pred_arr.pos, direct_arr.pos)
    assert math.isclose(pred_arr.long_speed, direct_arr.long_speed)
    assert pred_arr.long_acc == ag.action_long_accs[5]

    pred_scalar = ag.get_predicted_state(long_accs=0.75, yaw_rates=0.0)
    assert pred_scalar.long_acc == 0.75


# =============================================================================
# Part B: Geometry, OBB collision distances, looming, and perception tests
# =============================================================================

def _obb_corners(center, heading_angle, length, width):
    h = np.array((math.cos(heading_angle), math.sin(heading_angle)))
    n = np.array((-h[1], h[0]))
    hl = length / 2.0
    hw = width / 2.0
    return np.array([
        center + hl * h + hw * n,
        center + hl * h - hw * n,
        center - hl * h - hw * n,
        center - hl * h + hw * n,
    ])


def _obb_overlap_sat(c1, yaw1, l1, w1, c2, yaw2, l2, w2):
    """Exact 2D oriented bounding box overlap via Separating Axis Theorem."""
    corners1 = _obb_corners(c1, yaw1, l1, w1)
    corners2 = _obb_corners(c2, yaw2, l2, w2)
    axes = [
        np.array((math.cos(yaw1), math.sin(yaw1))),
        np.array((-math.sin(yaw1), math.cos(yaw1))),
        np.array((math.cos(yaw2), math.sin(yaw2))),
        np.array((-math.sin(yaw2), math.cos(yaw2))),
    ]
    for axis in axes:
        p1 = corners1 @ axis
        p2 = corners2 @ axis
        if p1.max() < p2.min() - 1e-12 or p2.max() < p1.min() - 1e-12:
            return False
    return True


def _brute_force_coll_dist(ego_length, ego_width, oth_length, oth_width,
                           crossing_angle_deg):
    """Find max |s1| along agent 1's path for which there exists s2 along
    agent 2's path such that their 2D oriented bounding boxes overlap.
    """
    theta = math.radians(crossing_angle_deg)
    yaw1 = math.pi / 2.0
    yaw2 = math.pi / 2.0 + theta
    h1 = np.array((math.cos(yaw1), math.sin(yaw1)))
    h2 = np.array((math.cos(yaw2), math.sin(yaw2)))

    def overlaps_any_s2(s1):
        c1 = -s1 * h1
        # Project c1 onto h2 to centre the search window for s2
        s2_center = -float(np.dot(c1, h2))
        half_span = ego_length + ego_width + oth_length + oth_width + 1.0
        s2_grid = np.linspace(s2_center - half_span, s2_center + half_span, 801)
        for s2 in s2_grid:
            if _obb_overlap_sat(c1, yaw1, ego_length, ego_width,
                                -s2 * h2, yaw2, oth_length, oth_width):
                return True
        return False

    # Coarse bracket then binary search for the exact threshold s1 >= 0
    lo = 0.0
    hi = (ego_length + ego_width + oth_width) / math.sin(theta) + 2.0
    assert overlaps_any_s2(lo)
    assert not overlaps_any_s2(hi)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if overlaps_any_s2(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@pytest.mark.parametrize('angle_deg', [45, 60, 90, 120, 135])
@pytest.mark.parametrize('dims', [
    # (ego_len, ego_wid, oth_len, oth_wid)
    (0.8, 0.8, 4.2, 1.8),  # pedestrian ego, vehicle other
    (4.2, 1.8, 0.8, 0.8),  # vehicle ego, pedestrian other
    (3.4, 1.2, 2.2, 1.5),  # general asymmetric rectangles
])
def test_coll_dist_matches_brute_force_2d_obb(angle_deg, dims):
    ego_len, ego_wid, oth_len, oth_wid = dims
    analytic = sc_scenario_helper.get_agent_coll_dist(
        ego_len, oth_wid, ego_width=ego_wid,
        crossing_angle=math.radians(angle_deg))
    brute = _brute_force_coll_dist(ego_len, ego_wid, oth_len, oth_wid,
                                   angle_deg)
    assert math.isclose(analytic, brute, abs_tol=1e-4)

    # Symmetry around 90 deg: coll_dist(theta) == coll_dist(180 - theta)
    supp_analytic = sc_scenario_helper.get_agent_coll_dist(
        ego_len, oth_wid, ego_width=ego_wid,
        crossing_angle=math.radians(180 - angle_deg))
    assert math.isclose(analytic, supp_analytic, rel_tol=1e-12, abs_tol=1e-12)


def test_coll_dist_and_crossing_cos_sin_edge_cases():
    # Legacy 2-argument call works for perpendicular paths and snaps near pi/2
    d_legacy = sc_scenario_helper.get_agent_coll_dist(4.2, 0.8)
    assert d_legacy == 0.8 / 2 + 4.2 / 2
    assert sc_scenario_helper.get_agent_coll_dist(
        4.2, 0.8, crossing_angle=-math.pi / 2) == d_legacy
    assert sc_scenario_helper.get_agent_coll_dist(
        4.2, 0.8, crossing_angle=math.pi / 2 + 1e-12) == d_legacy

    # Non-perpendicular without ego_width raises ValueError
    with pytest.raises(ValueError):
        sc_scenario_helper.get_agent_coll_dist(
            4.2, 0.8, crossing_angle=math.radians(60))

    # Parallel and antiparallel paths raise ValueError
    for bad_angle in (0.0, math.pi, -math.pi, 1e-8, math.pi - 1e-8):
        with pytest.raises(ValueError):
            sc_scenario_helper.get_crossing_cos_sin(bad_angle, 0.0)
        with pytest.raises(ValueError):
            sc_scenario_helper.get_agent_coll_dist(
                4.2, 0.8, ego_width=1.8, crossing_angle=bad_angle)

    for bad_deg in (0, 180, -45, 200, 1e-8):
        with pytest.raises(ValueError):
            sc_scenario.get_crossing_geometry(bad_deg, (5.0, 40.0))


@pytest.mark.parametrize('angle_deg', [45, 60, 90, 120, 135])
def test_inter_agent_dists_matches_2d_positions(angle_deg):
    init_pos, goal_pos = sc_scenario.get_crossing_geometry(
        angle_deg, (10.0, 25.0))
    headings = goal_pos - init_pos
    headings = headings / np.linalg.norm(headings, axis=1, keepdims=True)
    yaws = np.arctan2(headings[:, 1], headings[:, 0])
    c, _ = sc_scenario_helper.get_crossing_cos_sin(yaws[0], yaws[1])

    d1s = np.array([15.0, 5.0, 0.0, -3.5, -10.0])
    d2s = np.array([30.0, -4.0, 8.0, 12.0, -6.0])
    dists = sc_scenario_helper.get_inter_agent_dists(d1s, d2s, c)

    cp = np.zeros(2)
    for k in range(len(d1s)):
        s1 = commotions.KinematicState(yaw_angle=yaws[0])
        s1.signed_CP_dist = d1s[k]
        p1 = sc_scenario_helper.get_pos_from_signed_dist_to_conflict_pt(cp, s1)
        s2 = commotions.KinematicState(yaw_angle=yaws[1])
        s2.signed_CP_dist = d2s[k]
        p2 = sc_scenario_helper.get_pos_from_signed_dist_to_conflict_pt(cp, s2)
        assert math.isclose(dists[k], float(np.linalg.norm(p1 - p2)),
                            rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize('angle_deg', [60, 90, 120])
@pytest.mark.parametrize('state_tuple', [
    # (d_ego, v_ego, d_oth, v_oth, eff_width)
    (6.0, 1.3, 25.0, 13.8, 1.8),
    (25.0, 12.0, 4.5, 1.4, 0.8),
    (3.0, 0.0, 15.0, 10.0, 1.8),   # stationary ego
    (-2.5, 1.2, 8.0, 9.5, 1.8),    # ego already past conflict point
    (10.0, 8.0, -4.0, 1.5, 0.8),   # other already past conflict point
])
def test_optical_expansion_matches_finite_difference(angle_deg, state_tuple):
    d1, v1, d2, v2, eff_w = state_tuple
    c, _ = sc_scenario_helper.get_crossing_cos_sin(
        math.radians(angle_deg), 0.0)
    oth_img = sc_scenario_helper.SCAgentImage(
        ctrl_type=sc_scenario.CtrlType.ACCELERATION, params=None,
        v_free=None, g_free=None, V_free=None, coll_dist=None,
        eff_width=eff_w, cos_crossing_angle=c)

    def make_states(dt):
        ego = commotions.KinematicState(long_speed=v1)
        ego.signed_CP_dist = d1 - v1 * dt
        oth = commotions.KinematicState(long_speed=v2)
        oth.signed_CP_dist = d2 - v2 * dt
        return ego, oth

    dt = 1e-7
    ego_p, oth_p = make_states(+dt)
    ego_m, oth_m = make_states(-dt)
    ego_0, oth_0 = make_states(0.0)

    # Test both 3-arg (reads oth_img.cos_crossing_angle) and explicit 4-arg calls
    theta_p = sc_scenario_helper.get_agent_optical_size(ego_p, oth_p, oth_img)
    theta_m = sc_scenario_helper.get_agent_optical_size(
        ego_m, oth_m, oth_img, cos_crossing_angle=c)
    fd_theta_dot = (theta_p - theta_m) / (2.0 * dt)

    analytic_scalar = sc_scenario_helper.get_agent_optical_exp(
        ego_0, oth_0, oth_img)
    analytic_explicit = sc_scenario_helper.get_agent_optical_exp(
        ego_0, oth_0, oth_img, cos_crossing_angle=c)
    analytic_vec = sc_scenario_helper.get_agent_optical_exps(
        np.array([d1]), np.array([v1]),
        np.array([d2]), np.array([v2]), eff_w, c)[0]

    assert math.isclose(analytic_scalar, analytic_explicit,
                        rel_tol=1e-15, abs_tol=1e-15)
    assert math.isclose(analytic_scalar, analytic_vec,
                        rel_tol=1e-15, abs_tol=1e-15)
    assert math.isclose(analytic_scalar, fd_theta_dot,
                        rel_tol=1e-7, abs_tol=1e-9)


def test_perception_modes_and_crossing_angle():
    """sc_scenario_perception.__main__ style test across angles."""
    for angle_deg, expected_c in ((60, 0.5), (90, 0.0), (120, -0.5)):
        init_pos, goal_pos = sc_scenario.get_crossing_geometry(
            angle_deg, (5.0, 40.0))
        sim = commotions.Simulation(start_time=0, end_time=1.0, time_step=0.05)
        sim.conflict_point = np.zeros(2)
        speeds = (1.5, 10.0)
        for i_ag, name in enumerate(('P', 'V')):
            commotions.AgentWithGoal(
                name, sim, goal_pos[i_ag],
                commotions.KinematicState(pos=init_pos[i_ag],
                                          long_speed=speeds[i_ag],
                                          yaw_angle=None))
        sim.run()

        c, _ = sc_scenario_helper.get_crossing_cos_sin(
            sim.agents[0].trajectory.yaw_angle[0],
            sim.agents[1].trajectory.yaw_angle[0])
        assert math.isclose(c, expected_c, abs_tol=1e-12)

        prior = sc_scenario_perception.KalmanPrior(40.0, 40.0, 10.0, 10.0)
        percs = {
            'base': sc_scenario_perception.Perception(
                sim, cos_crossing_angle=c),
            'noisy': sc_scenario_perception.Perception(
                sim, pos_obs_noise_stddev=2.0, noise_seed=7,
                cos_crossing_angle=c),
            'noisy_ang': sc_scenario_perception.Perception(
                sim, angular_perception=True, ego_eye_height=1.6,
                pos_obs_noise_stddev=0.01, noise_seed=7,
                cos_crossing_angle=c),
            'kalman_ang_draw': sc_scenario_perception.Perception(
                sim, angular_perception=True, ego_eye_height=1.6,
                pos_obs_noise_stddev=0.01, kalman_filter=True, prior=prior,
                spd_proc_noise_stddev=0.1, draw_from_estimate=True,
                noise_seed=7, cos_crossing_angle=c),
        }
        for k in range(sim.settings.n_time_steps):
            states = []
            for ag in sim.agents:
                st = ag.get_kinematic_state(k)
                st.signed_CP_dist = (
                    sc_scenario_helper.get_signed_dist_to_conflict_pt(
                        sim.conflict_point, st))
                states.append(st)
            for perc in percs.values():
                perc.update(k, states[0], states[1])
                assert np.all(np.isfinite(perc.perc_oth_state.pos))

        assert np.allclose(percs['base'].states.x_perceived[0, :],
                           percs['base'].states.x_true[0, :])
        assert np.all(np.isfinite(percs['kalman_ang_draw'].states.x_perceived))

    # Acute (60 deg) has smaller inter-agent distance D than obtuse (120 deg)
    # when both agents are approaching, so angular noise stddev must be smaller
    sim_dummy = commotions.Simulation(0, 0.2, 0.1)
    sim_dummy.conflict_point = np.zeros(2)
    ego = commotions.KinematicState(pos=np.zeros(2), long_speed=1.3,
                                    yaw_angle=math.pi / 2)
    ego.signed_CP_dist = 20.0
    oth = commotions.KinematicState(pos=np.zeros(2), long_speed=10.0,
                                    yaw_angle=0.0)
    oth.signed_CP_dist = 20.0
    errs = {}
    for label, c_val in (('acute60', 0.5), ('perp90', 0.0), ('obtuse120', -0.5)):
        p = sc_scenario_perception.Perception(
            sim_dummy, angular_perception=True, ego_eye_height=1.5,
            pos_obs_noise_stddev=0.02, noise_seed=42, cos_crossing_angle=c_val)
        p.update(0, ego, oth)
        errs[label] = abs(p.states.x_noisy[0, 0] - 20.0)
    assert errs['acute60'] < errs['perp90'] < errs['obtuse120']


# =============================================================================
# Full SCSimulation & SCPaper (sc_fitting / sc_run) tests
# =============================================================================

def test_exact_90_deg_equivalence_and_rigid_rotation():
    init_90, goal_90 = sc_scenario.get_crossing_geometry(90, (5.0, 35.0))
    assert np.array_equal(init_90, np.array([[0.0, -5.0], [35.0, 0.0]]))
    assert np.array_equal(goal_90, np.array([[0.0, 5.0], [-50.0, 0.0]]))

    assumptions = sc_scenario.get_assumptions_dict(
        default_value=False, oVA=True, oVAl=True, oBEo=True, oBEv=True,
        oAI=True, oSNv=True, oPF=True)
    params, params_k = sc_scenario.get_default_params(oVA=True)
    priors = (
        sc_scenario_perception.KalmanPrior(35.0, 35.0, 10.0, 10.0),
        sc_scenario_perception.KalmanPrior(5.0, 5.0, 1.5, 1.5),
    )

    sim_std = sc_scenario.SCSimulation(
        (sc_scenario.CtrlType.SPEED, sc_scenario.CtrlType.ACCELERATION),
        (0.8, 1.8), (0.8, 4.2), goal_90, init_90,
        initial_speeds=np.array((1.3, 10.0)), end_time=4.0, time_step=0.1,
        optional_assumptions=assumptions, params=params, params_k=params_k,
        kalman_priors=priors, noise_seeds=(11, 22))
    sim_std.run()

    # Rigidly rotate the entire 90-degree crossing by 37 degrees
    rot_rad = math.radians(37.0)
    rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                        [math.sin(rot_rad), math.cos(rot_rad)]])
    init_rot = init_90 @ rot_mat.T
    goal_rot = goal_90 @ rot_mat.T

    sim_rot = sc_scenario.SCSimulation(
        (sc_scenario.CtrlType.SPEED, sc_scenario.CtrlType.ACCELERATION),
        (0.8, 1.8), (0.8, 4.2), goal_rot, init_rot,
        initial_speeds=np.array((1.3, 10.0)), end_time=4.0, time_step=0.1,
        optional_assumptions=assumptions, params=params, params_k=params_k,
        kalman_priors=priors, noise_seeds=(11, 22))
    sim_rot.run()

    for ag_s, ag_r in zip(sim_std.agents, sim_rot.agents):
        assert ag_r.cos_crossing_angle == 0.0
        assert ag_r.coll_dist == ag_s.coll_dist
        assert np.allclose(ag_r.signed_CP_dists, ag_s.signed_CP_dists,
                           rtol=1e-12, atol=1e-12)
        assert np.allclose(ag_r.trajectory.long_speed,
                           ag_s.trajectory.long_speed, rtol=1e-12, atol=1e-12)
        assert np.allclose(ag_r.states.thetaDot, ag_s.states.thetaDot,
                           rtol=1e-12, atol=1e-12)
        pos_unrot = rot_mat.T @ ag_r.trajectory.pos
        assert np.allclose(pos_unrot, ag_s.trajectory.pos,
                           rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('angle_deg', [60, 120])
def test_non_perpendicular_sc_simulation_and_after_simulation(angle_deg):
    init_pos, goal_pos = sc_scenario.get_crossing_geometry(
        angle_deg, (4.0, 30.0))
    assumptions = sc_scenario.get_assumptions_dict(
        default_value=False, oVA=True, oVAl=True, oBEv=True, oAI=True,
        oSNv=True, oPF=True)
    params, params_k = sc_scenario.get_default_params(oVA=True)
    priors = (
        sc_scenario_perception.KalmanPrior(30.0, 30.0, 10.0, 10.0),
        sc_scenario_perception.KalmanPrior(4.0, 4.0, 1.5, 1.5),
    )
    widths = (0.8, 1.8)
    lengths = (0.8, 4.2)

    sim = sc_scenario.SCSimulation(
        (sc_scenario.CtrlType.SPEED, sc_scenario.CtrlType.ACCELERATION),
        widths, lengths, goal_pos, init_pos,
        initial_speeds=np.array((1.3, 10.0)), end_time=6.0, time_step=0.1,
        optional_assumptions=assumptions, params=params, params_k=params_k,
        kalman_priors=priors, noise_seeds=(5, 6))
    sim.run()

    expected_cos = math.cos(math.radians(angle_deg))
    for i_ag, ag in enumerate(sim.agents):
        assert math.isclose(ag.cos_crossing_angle, expected_cos, abs_tol=1e-12)
        assert math.isclose(ag.perception.cos_crossing_angle, expected_cos,
                            abs_tol=1e-12)
        expected_cd = sc_scenario_helper.get_agent_coll_dist(
            lengths[i_ag], widths[1 - i_ag], ego_width=widths[i_ag],
            crossing_angle=math.radians(angle_deg))
        assert math.isclose(ag.coll_dist, expected_cd, rel_tol=1e-12)
        # Non-perpendicular collision distance is strictly larger than 90 deg
        cd_90 = sc_scenario_helper.get_agent_coll_dist(
            lengths[i_ag], widths[1 - i_ag])
        assert ag.coll_dist > cd_90

        # Verify ca_entry_sample / ca_entry_time matches signed_CP_dists
        entered_by_cp = np.nonzero(
            np.abs(ag.signed_CP_dists) <= ag.coll_dist + 1e-12)[0]
        if len(entered_by_cp) == 0:
            assert math.isinf(ag.ca_entry_time)
        else:
            assert ag.ca_entry_sample == entered_by_cp[0]
            assert ag.ca_entry_time == sim.time_stamps[entered_by_cp[0]]

    # Smoke-test plotting with Agg backend
    sim.do_plots(trajs=True, kinem_states=True, looming=True)
    plt.close('all')

    # Parallel paths inside SCSimulation must fail in prepare_for_simulation
    bad_sim = sc_scenario.SCSimulation(
        (sc_scenario.CtrlType.SPEED, sc_scenario.CtrlType.ACCELERATION),
        widths, lengths,
        goal_positions=np.array([[0.0, 10.0], [0.0, 20.0]]),
        initial_positions=np.array([[0.0, -5.0], [0.0, -15.0]]),
        initial_speeds=np.array((1.0, 5.0)), end_time=1.0, time_step=0.1)
    with pytest.raises(ValueError):
        bad_sim.run()


def test_sc_fitting_and_sc_run_crossing_angles():
    model_w_params = sc_run.get_model_with_params('oVAoBEvoAI')
    init_dists = (4.5, 38.0)
    init_speeds = sc_fitting.AGENT_FREE_SPEEDS

    sim_default = sc_run.run_simulation(
        model_w_params, initial_cp_distances=init_dists,
        initial_speeds=init_speeds, end_time=5.0, idx_parameterisation=0)
    sim_90 = sc_run.run_simulation(
        model_w_params, initial_cp_distances=init_dists,
        initial_speeds=init_speeds, end_time=5.0, idx_parameterisation=0,
        crossing_angle=90)

    for a_def, a_90 in zip(sim_default.agents, sim_90.agents):
        assert np.array_equal(a_def.trajectory.pos, a_90.trajectory.pos)
        assert np.array_equal(a_def.trajectory.long_speed,
                              a_90.trajectory.long_speed)
        assert np.array_equal(a_def.signed_CP_dists, a_90.signed_CP_dists)

    sims_by_angle = {}
    for angle_deg in (60, 120):
        sim_ang = sc_run.run_simulation(
            model_w_params, initial_cp_distances=init_dists,
            initial_speeds=init_speeds, end_time=6.0, idx_parameterisation=0,
            crossing_angle=angle_deg)
        sims_by_angle[angle_deg] = sim_ang
        # Initial signed distances to conflict point match caller's init_dists
        for i_ag in range(2):
            assert math.isclose(sim_ang.agents[i_ag].signed_CP_dists[0],
                                init_dists[i_ag], rel_tol=1e-12, abs_tol=1e-12)
        # Vehicle heading is rotated angle_deg CCW from pedestrian (+y)
        ped_yaw = sim_ang.agents[0].trajectory.yaw_angle[0]
        veh_yaw = sim_ang.agents[1].trajectory.yaw_angle[0]
        c, _ = sc_scenario_helper.get_crossing_cos_sin(ped_yaw, veh_yaw)
        assert math.isclose(c, math.cos(math.radians(angle_deg)), abs_tol=1e-12)

    # 60 deg and 120 deg have equal collision distances (sin & |cot| symmetric)
    # when looming is off; verify SCPaperScenario with looming (oVAl) differs
    # between 60 and 120 deg because inter-agent distance D differs!
    assert math.isclose(sims_by_angle[60].agents[0].coll_dist,
                        sims_by_angle[120].agents[0].coll_dist, rel_tol=1e-12)
    scen_60 = sc_fitting.SCPaperScenario(
        'Enc60', initial_ttcas=(3.0, 3.0),
        stop_criteria=sc_fitting.EXITED_STOP,
        metric_names=sc_fitting.TWO_AG_METRIC_NAMES,
        crossing_angle=60)
    scen_120 = sc_fitting.SCPaperScenario(
        'Enc120', initial_ttcas=(3.0, 3.0),
        stop_criteria=sc_fitting.EXITED_STOP,
        metric_names=sc_fitting.TWO_AG_METRIC_NAMES,
        crossing_angle=120)
    looming_params = {'T_delta': 30, 'thetaDot_0': 0.001, 'thetaDot_1': 0.05,
                      'beta_V': 80}
    sim_l60 = sc_fitting.construct_model_and_simulate_scenario(
        'oVAoVAloBEvoAI', looming_params, scen_60)
    sim_l120 = sc_fitting.construct_model_and_simulate_scenario(
        'oVAoVAloBEvoAI', looming_params, scen_120)
    # Initial thetaDot at t=0 differs between acute (60) and obtuse (120)
    assert not math.isclose(sim_l60.agents[0].states.thetaDot[0],
                            sim_l120.agents[0].states.thetaDot[0],
                            rel_tol=1e-3)
    m60 = sc_fitting.get_metrics_for_scenario(scen_60, sim_l60)
    m120 = sc_fitting.get_metrics_for_scenario(scen_120, sim_l120)
    assert 'Enc60_collision' in m60 and 'Enc120_collision' in m120

    # Invalid crossing angles rejected by SCPaperScenario
    for bad_deg in (0, 180, -10, 190):
        with pytest.raises(ValueError):
            sc_fitting.SCPaperScenario('Bad', initial_ttcas=(3.0, (2.0, 3.0)),
                                       crossing_angle=bad_deg)
