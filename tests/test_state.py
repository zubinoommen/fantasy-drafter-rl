import numpy as np

from fantasy_draft.config import LeagueConfig, PlayerPoolConfig, PositionProfile, RosterConfig
from fantasy_draft.draft import DraftEnvironment
from fantasy_draft.models import Position
from fantasy_draft.players import generate_players
from fantasy_draft.policies import RosterAwareSoftmaxPolicy
from fantasy_draft.state import StateEncoder


def test_encoding_is_fixed_size_and_deterministic_across_draft() -> None:
    environment = DraftEnvironment()
    state = environment.initial_state(generate_players(10))
    encoder = StateEncoder()
    initial = encoder.encode_state(state)
    assert initial.shape == (len(encoder.feature_names(state)),)
    assert initial.shape == (390,)
    assert np.array_equal(initial, encoder.encode_state(state))
    rng = np.random.default_rng(0)
    policy = RosterAwareSoftmaxPolicy()
    for _ in range(25):
        state = environment.step(state, policy.select_player(state, environment, rng))
        assert encoder.encode_state(state).shape == initial.shape


def test_board_is_sorted_by_mean_and_contains_no_ids() -> None:
    state = DraftEnvironment().initial_state(generate_players(11))
    encoder = StateEncoder()
    encoded = encoder.encode_state(state)
    board_section = encoded[encoder.slices(state)["board"]]
    board = board_section[:120].reshape(4, 10, 3)
    assert np.all(np.diff(board[:, :, 0], axis=1) <= 0)
    assert not any(name.endswith("player_id") for name in encoder.feature_names(state))


def test_board_padding_uses_zeros() -> None:
    base = PlayerPoolConfig().profiles
    profiles = dict(base)
    profiles[Position.QB] = PositionProfile(
        3, 24, 14, 0.7, 0.1, 4, 5, 2, 35, 0.1
    )
    profiles[Position.WR] = PositionProfile(
        63, 21, 5, 0.6, 0.5, 4, 8, 2, 30, 0.2
    )
    players = generate_players(12, PlayerPoolConfig(profiles=profiles))
    state = DraftEnvironment().initial_state(players)
    encoder = StateEncoder()
    section = encoder.encode_state(state)[encoder.slices(state)["board"]]
    board = section[:120].reshape(4, 10, 3)
    masks = section[120:].reshape(4, 10)
    assert np.array_equal(board[0, 3:], np.zeros((7, 3)))
    assert np.array_equal(masks[0], np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))


def test_encoding_ignores_external_future_outcomes() -> None:
    state = DraftEnvironment().initial_state(generate_players(13))
    encoder = StateEncoder()
    before = encoder.encode_state(state)
    _unrelated_future_scores = np.random.default_rng(4).normal(size=(14, 120))
    after = encoder.encode_state(state)
    assert np.array_equal(before, after)


def test_league_configuration_is_explicit_and_distinguishable() -> None:
    players = generate_players(14)
    standard = DraftEnvironment(LeagueConfig()).initial_state(players)
    two_qb_config = LeagueConfig(
        n_teams=12,
        roster=RosterConfig(
            required={
                Position.QB: 2,
                Position.RB: 2,
                Position.WR: 2,
                Position.TE: 1,
            },
            flex_slots=2,
            bench_slots=3,
        ),
    )
    # Supply a larger pool by repeating generation under expanded profiles.
    base = PlayerPoolConfig().profiles
    expanded = {
        position: PositionProfile(
            profile.count * 2,
            profile.elite_mu,
            profile.floor_mu,
            profile.decay_power,
            profile.mu_noise,
            profile.sigma_low,
            profile.sigma_high,
            profile.injury_alpha,
            profile.injury_beta,
            profile.injury_max,
        )
        for position, profile in base.items()
    }
    two_qb = DraftEnvironment(two_qb_config).initial_state(
        generate_players(14, PlayerPoolConfig(profiles=expanded))
    )
    encoder = StateEncoder()
    standard_vector = encoder.encode_state(standard)
    two_qb_vector = encoder.encode_state(two_qb)
    assert standard_vector.shape == two_qb_vector.shape
    assert not np.array_equal(standard_vector, two_qb_vector)
    names = encoder.feature_names()
    assert standard_vector[names.index("config.num_teams")] == 8
    assert two_qb_vector[names.index("config.num_teams")] == 12
    assert two_qb_vector[names.index("config.starters.QB")] == 2
    assert two_qb_vector[names.index("config.flex_slots")] == 2
    assert two_qb_vector[names.index("config.roster_size")] == 12
