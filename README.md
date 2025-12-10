# cs3346

1. Install dependencies: <br />
pip install -r requirements.txt

2. Run a quick environment test: <br />
python -m npc_rl.env.run_env_test

3. This will create ppo_model/ppo_shootergrid.zip: <br />
python -m npc_rl.ppo.train_ppo

4. evaluate ppo results: <br />
python -m npc_rl.ppo.evaluate_ppo

5. Baseline evaluation: <br />
python -m npc_rl.baseline.evaluate_baseline --outdir results_baseline --n_eval 200 --seed_start 10000

6. PPO evaluation with CSV output: <br />
python -m npc_rl.ppo.evaluate_ppo --outdir results_ppo --n_eval 200 --seed_start 10000

7. Build Table I + paper figures: <br />
python make_visuals.py --baseline_csv results_baseline/episodes_baseline.csv --ppo_csv results_ppo/episodes_ppo.csv --outdir paper_visuals

To remove the model and train from scratch: <br />
rm -rf ppo_model<br />
python -m npc_rl.ppo.train_ppo<br />
python -m npc_rl.ppo.evaluate_ppo<br />
