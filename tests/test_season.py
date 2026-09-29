import numpy as np

from fantasy_draft.config import RosterConfig
from fantasy_draft.models import Player, Position
from fantasy_draft.season import (
    ScoredPlayer,
    SeasonSimulator,
    mean_weekly_score,
    optimal_lineup,
)
from fantasy_draft.waivers import NoReplacementModel, PoolReplacementLevelModel


def player(
    player_id: int,
    position: Position,
    mu: float,
    injury: float = 0.0,
) -> Player:
    return Player(player_id, position, mu, 0.0, injury, player_id % 32)


def complete_roster(injury: float = 0.0) -> tuple[Player, ...]:
    positions = [
        Position.QB,
        Position.RB,
        Position.RB,
        Position.WR,
        Position.WR,
        Position.TE,
        Position.RB,
    ]
    return tuple(player(index, position, 10 + index, injury) for index, position in enumerate(positions))


def test_optimal_lineup_obeys_required_positions_and_flex() -> None:
    candidates = [
        ScoredPlayer(player(0, Position.QB, 0), 20),
        ScoredPlayer(player(1, Position.RB, 0), 10),
        ScoredPlayer(player(2, Position.RB, 0), 11),
        ScoredPlayer(player(3, Position.RB, 0), 30),
        ScoredPlayer(player(4, Position.WR, 0), 12),
        ScoredPlayer(player(5, Position.WR, 0), 13),
        ScoredPlayer(player(6, Position.TE, 0), 9),
    ]
    lineup = optimal_lineup(candidates, RosterConfig())
    assert len(lineup.selected_ids) == 7
    assert 3 in lineup.selected_ids  # highest RB is used, with another RB in FLEX
    assert lineup.score == 105


def test_injury_probability_extremes_and_season_length() -> None:
    healthy = complete_roster(0.0)
    simulator = SeasonSimulator(RosterConfig(), n_weeks=4, replacement_model=NoReplacementModel())
    healthy_result = simulator.simulate(healthy, (), seed=1)
    injured_result = simulator.simulate(complete_roster(1.0), (), seed=1)
    assert len(healthy_result.weekly_scores) == 4
    assert np.all(healthy_result.weekly_scores > 0)
    assert np.array_equal(injured_result.weekly_scores, np.zeros(4))


def test_injured_qb_without_backup_gets_waiver_points_not_automatic_zero() -> None:
    roster = (player(0, Position.QB, 22, 1.0),)
    waiver_qb = player(1, Position.QB, 14, 0.0)
    qb_only = RosterConfig(
        required={Position.QB: 1},
        flex_slots=0,
        flex_eligible=(),
        bench_slots=1,
    )
    result = SeasonSimulator(qb_only, 3, PoolReplacementLevelModel()).simulate(
        roster, (waiver_qb,), seed=3, player_universe=(*roster, waiver_qb)
    )
    assert np.array_equal(result.weekly_scores, np.full(3, 14.0))
    assert all(lineup.replacement_ids == (1,) for lineup in result.lineups)


def test_healthy_backup_qb_is_used_before_waiver() -> None:
    roster = (
        player(0, Position.QB, 22, 1.0),
        player(1, Position.QB, 12, 0.0),
    )
    waiver_qb = player(2, Position.QB, 18, 0.0)
    qb_only = RosterConfig(
        required={Position.QB: 1},
        flex_slots=0,
        flex_eligible=(),
        bench_slots=1,
    )
    result = SeasonSimulator(qb_only, 2).simulate(
        roster, (waiver_qb,), seed=4, player_universe=(*roster, waiver_qb)
    )
    assert np.array_equal(result.weekly_scores, np.full(2, 12.0))
    assert all(not lineup.replacement_ids for lineup in result.lineups)


def test_simultaneous_all_position_injuries_use_distinct_replacements() -> None:
    roster = complete_roster(1.0)
    free_agents = tuple(
        player(100 + index, position, 8.0)
        for index, position in enumerate(
            [
                Position.QB,
                Position.RB,
                Position.RB,
                Position.RB,
                Position.WR,
                Position.WR,
                Position.WR,
                Position.TE,
            ]
        )
    )
    result = SeasonSimulator(RosterConfig(), 1).simulate(
        roster, free_agents, seed=5, player_universe=(*roster, *free_agents)
    )
    used = result.lineups[0].replacement_ids
    assert len(used) == 7
    assert len(set(used)) == 7


def test_healthy_complete_roster_does_not_use_waivers_and_utility_is_separate() -> None:
    roster = complete_roster()
    free_agents = (player(100, Position.QB, 30),)
    result = SeasonSimulator(RosterConfig(), 2).simulate(
        roster, free_agents, seed=6, player_universe=(*roster, *free_agents)
    )
    assert all(not lineup.replacement_ids for lineup in result.lineups)
    assert mean_weekly_score(result.weekly_scores) == np.mean(result.weekly_scores)


def test_two_qb_and_two_flex_lineup_configuration() -> None:
    config = RosterConfig(
        required={
            Position.QB: 2,
            Position.RB: 2,
            Position.WR: 2,
            Position.TE: 1,
        },
        flex_slots=2,
        bench_slots=3,
    )
    positions = [
        Position.QB,
        Position.QB,
        Position.RB,
        Position.RB,
        Position.RB,
        Position.WR,
        Position.WR,
        Position.WR,
        Position.TE,
    ]
    candidates = [
        ScoredPlayer(player(index, position, 10 + index), 10 + index)
        for index, position in enumerate(positions)
    ]
    lineup = optimal_lineup(candidates, config)
    assert len(lineup.selected_ids) == 9
    selected_positions = [
        candidates[player_id].player.position for player_id in lineup.selected_ids
    ]
    assert selected_positions.count(Position.QB) == 2
    assert sum(
        position in config.flex_eligible for position in selected_positions
    ) == 7


def test_weekly_lineup_is_selected_ex_ante_without_realized_score_foreknowledge() -> None:
    qb_only = RosterConfig(
        required={Position.QB: 1},
        flex_slots=0,
        flex_eligible=(),
        bench_slots=1,
    )
    projected_starter = Player(0, Position.QB, 10.0, 5.0, 0.0, 0)
    projected_backup = Player(1, Position.QB, 9.0, 5.0, 0.0, 1)
    # With seed 5 the backup realizes 11.10 while the starter realizes 8.76.
    # The higher-mu starter must remain selected because lineup choice is ex ante.
    result = SeasonSimulator(qb_only, 1, NoReplacementModel()).simulate(
        (projected_starter, projected_backup),
        (),
        seed=5,
    )
    assert result.lineups[0].selected_ids == (projected_starter.player_id,)
    assert np.isclose(result.weekly_scores[0], 8.758191889523758)
