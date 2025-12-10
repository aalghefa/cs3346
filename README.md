# cs3346

1. Install dependencies: `\n`
pip install -r requirements.txt

2. Run a quick environment test: `\n`
python -m npc_rl.env.run_env_test

3. This will create ppo_model/ppo_shootergrid.zip: `\n`
python -m npc_rl.ppo.train_ppo

4. evaluate ppo results: `\n`
python -m npc_rl.ppo.evaluate_ppo

5. Baseline evaluation: `\n`
python -m npc_rl.baseline.evaluate_baseline --outdir results_baseline --n_eval 200 --seed_start 10000

6. PPO evaluation with CSV output: `\n`
python -m npc_rl.ppo.evaluate_ppo --outdir results_ppo --n_eval 200 --seed_start 10000

7. Build Table I + paper figures: `\n`
python make_visuals.py --baseline_csv results_baseline/episodes_baseline.csv --ppo_csv results_ppo/episodes_ppo.csv --outdir paper_visuals

To remove the model and train from scratch: `\n`
rm -rf ppo_model
python -m npc_rl.ppo.train_ppo
python -m npc_rl.ppo.evaluate_ppo
