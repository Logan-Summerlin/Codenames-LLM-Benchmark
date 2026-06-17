#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/agentbot/workspace/codenames-llm-benchmark')
sys.path.insert(0, str(ROOT / 'src'))

from codenames_benchmark.llm.openrouter import OpenRouterClient
from codenames_benchmark.ratings import EloRatingSystem
from codenames_benchmark.tournament import OPENROUTER_CODENAMES_MODELS, TournamentGame
from codenames_benchmark.tournament_runner import (
    elo_rankings,
    model_lookup,
    play_game,
    provider_order_payload,
    reasoning_effort_payload,
    record_rankings,
    record_tally,
    tournament_manifest,
    update_record_tally,
    write_json,
    write_static_artifacts,
)

SOURCE_RUN = ROOT / 'runs' / 'openrouter-limited-coverage-paid-llama-20260609-212930'
SELECTED_GAME_NUMBERS = {5, 7, 8, 9, 10, 11}
MAX_TURNS = 30


def main() -> int:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = ROOT / 'runs' / f'openrouter-limited-coverage-paid-llama-rerun-failed-{stamp}'

    schedule_path = SOURCE_RUN / 'schedule.json'
    schedule_data = json.loads(schedule_path.read_text())
    selected_games = [TournamentGame(**game) for game in schedule_data if game['game_number'] in SELECTED_GAME_NUMBERS]
    if len(selected_games) != len(SELECTED_GAME_NUMBERS):
        missing = sorted(SELECTED_GAME_NUMBERS - {game.game_number for game in selected_games})
        raise RuntimeError(f'missing selected games from schedule: {missing}')

    labels = model_lookup(OPENROUTER_CODENAMES_MODELS)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_static_artifacts(
        output_dir=out_dir,
        models=OPENROUTER_CODENAMES_MODELS,
        schedule=selected_games,
        manifest=tournament_manifest(
            benchmark='codenames-openrouter-limited-coverage-rerun-failed',
            models=OPENROUTER_CODENAMES_MODELS,
            game_count=len(selected_games),
            max_turns=MAX_TURNS,
            seed_prefix='openrouter-codenames',
            elo_initial=1500.0,
            elo_k=32.0,
            schedule_mode='limited-coverage',
            round_size=len(selected_games),
            max_tokens=int(os.environ.get('OPENROUTER_MAX_TOKENS', '10000')),
            extra_fields={
                'rerun_of': str(SOURCE_RUN),
                'selected_game_numbers': sorted(SELECTED_GAME_NUMBERS),
                'rerun_mode': 'failed_matchups_only',
            },
        ),
    )

    client = OpenRouterClient()
    ratings = EloRatingSystem(models=[m.slug for m in OPENROUTER_CODENAMES_MODELS], initial=1500.0, k=32.0)
    records = record_tally([m.slug for m in OPENROUTER_CODENAMES_MODELS])

    round_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(selected_games)) as pool:
        futures = {
            pool.submit(play_game, game, output_dir=out_dir, max_turns=MAX_TURNS, labels=labels, client=client): game
            for game in selected_games
        }
        for future in as_completed(futures):
            game = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive artifact write
                result = {
                    'game': game.to_dict(),
                    'status': 'game_error',
                    'error': str(exc)[:1000],
                    'transcript': str(out_dir / f'game-{game.game_number:03d}' / 'transcript.json'),
                    'summary_path': str(out_dir / f'game-{game.game_number:03d}' / 'summary.json'),
                }
                write_json(Path(result['summary_path']), result)
            round_results.append(result)

    ordered_results = sorted(round_results, key=lambda item: item['game']['game_number'])
    completed_round_games = [
        {
            'red_model': result['game']['red_model'],
            'blue_model': result['game']['blue_model'],
            'winner_model': result.get('winner_model'),
            'game_number': result['game']['game_number'],
            'round_index': 1,
        }
        for result in ordered_results
        if result.get('status') == 'game_complete'
    ]
    elo_entries = ratings.record_round(completed_round_games)
    round_elo_by_game = {entry['game_number']: entry for entry in elo_entries}

    for result in ordered_results:
        if result.get('status') == 'game_complete':
            result['elo_update'] = round_elo_by_game[result['game']['game_number']]
            update_record_tally(records, result['game']['red_model'], result['game']['blue_model'], result.get('winner_model'))
        write_json(Path(result['summary_path']), result)
        with (out_dir / 'results.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(result, sort_keys=True) + '\n')
        print(json.dumps({
            'game_number': result['game']['game_number'],
            'status': result.get('status'),
            'winner': result.get('winner_model'),
            'reason': result.get('reason'),
        }, sort_keys=True), flush=True)

    standings = ratings.standings()
    write_json(out_dir / 'round_summaries.json', [{
        'round_index': 1,
        'game_numbers': [result['game']['game_number'] for result in ordered_results],
        'completed_games': len(completed_round_games),
        'error_games': sum(1 for result in ordered_results if result.get('status') != 'game_complete'),
        'standings': standings,
        'records': record_rankings(records, ratings, [m.slug for m in OPENROUTER_CODENAMES_MODELS], labels),
    }])
    write_json(out_dir / 'standings.json', standings)
    write_json(out_dir / 'elo_history.json', ratings.history)
    write_json(out_dir / 'records.json', records)
    write_json(out_dir / 'elo_rankings.json', elo_rankings(ratings, records, labels))
    write_json(out_dir / 'record_rankings.json', record_rankings(records, ratings, [m.slug for m in OPENROUTER_CODENAMES_MODELS], labels))
    write_json(out_dir / 'run_state.json', {
        'status': 'complete',
        'output_dir': str(out_dir),
        'scheduled_games': len(selected_games),
        'completed_or_attempted_games': len(selected_games),
        'round_count': 1,
        'round_size': len(selected_games),
        'elo_rankings': standings,
        'record_rankings': record_rankings(records, ratings, [m.slug for m in OPENROUTER_CODENAMES_MODELS], labels),
        'selected_game_numbers': sorted(SELECTED_GAME_NUMBERS),
        'rerun_of': str(SOURCE_RUN),
    })
    print(json.dumps({'status': 'complete', 'output_dir': str(out_dir), 'completed_games': len(completed_round_games)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
