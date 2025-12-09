# cs3346

1. Install dependencies:
pip install -r requirements.txt

2. Run a quick environment test:
python -m npc_rl.env.run_env_test

3. This will create ppo_model/ppo_shootergrid.zip:
python -m npc_rl.ppo.train_ppo

4. evaluate ppo results:
python -m npc_rl.ppo.evaluate_ppo


To remove the model and train from scratch:
rm -rf ppo_model
python -m npc_rl.ppo.train_ppo
python -m npc_rl.ppo.evaluate_ppo
