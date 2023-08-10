import argparse
import json
import shutil
import os
from config import GlobalConfig
from run_ddpg_in_env import AgentOperator


def modify_global_config(global_config: GlobalConfig, options):
    global_config.train_config.algorithm = options["algorithm"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="default")
    run_args = parser.parse_args()

    json_name = run_args.algorithm
    if json_name != "default":
        with open("./json_config/" + json_name + ".json", "r") as f:
            options = json.load(f)

    global_config = GlobalConfig(options["algorithm"])
    if not global_config.control_config.save_runs:
        if os.path.exists("runs"):
            shutil.rmtree("runs")
    if not global_config.control_config.save_save_model:
        if os.path.exists("save_model"):
            shutil.rmtree("save_model")

    agent_operator = AgentOperator(global_config, json_name)
    agent_operator.run_all_episode()


if __name__ == "__main__":
    main()
