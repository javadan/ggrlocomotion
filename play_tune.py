import os, glob, math
from ray.tune import ExperimentAnalysis, Analysis
import random
import pandas as pd
import ray
from ray import tune
from ray.tune.registry import register_env
import argparse
from ray.rllib.utils.framework import try_import_tf

#from gym import wrappers

import ray.rllib.agents.ars as ars

tf1 = try_import_tf()

from gym_robotable.envs.robotable_gym_env import RobotableEnv

from ray.tune.schedulers import PopulationBasedTraining


ray.init()

config = ars.DEFAULT_CONFIG.copy()

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=15,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        # "rollouts_used": lambda: random.randint(5, 20)  # number of perturbs to keep in gradient estimate
        "sgd_stepsize": lambda: random.uniform(0.005,0.05),
        "rollouts_used": [4,6,8,10,12]
        # "lambda": lambda: random.uniform(0.9, 1.0),
        # "clip_param": lambda: random.uniform(0.01, 0.5),
        # "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        # ,
        # "num_sgd_iter": lambda: random.randint(1, 30),
        # "sgd_minibatch_size": lambda: random.randint(128, 16384),
        # "train_batch_size": lambda: random.randint(2000, 160000),
    })



ppo_pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=15,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        # ,
        # "num_sgd_iter": lambda: random.randint(1, 30),
        # "sgd_minibatch_size": lambda: random.randint(128, 16384),
        # "train_batch_size": lambda: random.randint(2000, 160000),
    })




ray_results = '/media/chrx/0FEC49A4317DA4DA/PBT_ARS/PBT_ARS/PBT_ARS/' #f'{os.getenv("HOME")}/ray_results/'



def env_creator(env_config):
    return RobotableEnv()  # return an env instance
    #return wrappers.Monitor(RobotableEnv(), ray_results, force=True)

register_env("RobotableEnv-v0", env_creator)




# episode_reward_mean_goal = 4.5;


class CustomStopper(tune.Stopper):
   def __init__(self):
       self.should_stop = False

   def __call__(self, trial_id, result):
       if not self.should_stop and result["episode_reward_mean"] > episode_reward_mean_goal:
           self.should_stop = True
       return self.should_stop #or result["training_iteration"] >= max_iter

   def stop_all(self):
       return self.should_stop


stopper = CustomStopper()



def restore_for_ppo(experiment_name):
    analysis = tune.run(
        'PPO',
        name=experiment_name,
        scheduler=ppo_pbt,
        checkpoint_freq=1,
        local_dir="/media/chrx/TURBO/PBT_ARS/PBT_ARS/",
        restore=best_checkpoint if loaded else None,
        keep_checkpoints_num=10,  # Keep only the best checkpoints
        checkpoint_score_attr='episode_reward_mean',  # Metric used to compare checkpoints
        num_samples=2,
        # stop=stopper,
        # resume=True,
        stop={  # Stop a single trial if one of the conditions are met
            # "episode_reward_mean": episode_reward_mean_goal,
            "training_iteration": 500},
        config={
            "log_level": "DEBUG",
            "env": "RobotableEnv-v0",
            "action_noise_std": 0.0,
            "noise_stdev": 0.02,  # std deviation of parameter noise
            "num_rollouts": 12,  # number of perturbs to try
            "rollouts_used": 8,  # number of perturbs to keep in gradient estimate
            "sgd_stepsize": 0.01,  # sgd step-size
            "observation_filter": "MeanStdFilter",
            "noise_size": 25000000,
            "eval_prob": 0.05,  # probability of evaluating the parameter rewards
            "report_length": 1,  # how many of the last rewards we average over
            "offset": 0,
            "num_workers": 1,
            "num_gpus": 0
        })


def restore(experiment_name):

    # global episode_reward_mean_goal
    # episode_reward_mean_goal = 4.5

    best_checkpoint_tuple = None
    ndx = 1
    run = 1

    while (True):
        try:

            # absolute_exp_dir = ray_results
            # latest_log_dir = glob.glob(absolute_exp_dir + 'ARS_RobotableEnv-v0*')
            # log_dir = sorted(latest_log_dir, key=os.path.getmtime)[-2]  # need 2 back because new folder
            # print(log_dir)

            # Restore
            exp_dir = glob.glob(ray_results + 'ARS_RobotableEnv-v0*')
            best_score = 0
            for dir in exp_dir:

                analysis = Analysis(dir)

                print(analysis.get_best_config("episode_reward_mean", mode="max"))
                best_log_dir = analysis.get_best_logdir("episode_reward_mean", mode="max")
                if (best_log_dir is None):
                    continue
                print (best_log_dir)
                checkpoints = analysis.get_trial_checkpoints_paths(best_log_dir, "episode_reward_mean")
                # print(checkpoints)

                cleanedList = [x for x in checkpoints if str(x[1]) != 'nan']

                sorted_checkpoints = sorted(cleanedList, key=lambda x: x[1], reverse=True)
                # print(sorted_checkpoints)

                current_best_checkpoint = sorted_checkpoints[0][0]
                current_best_score = sorted_checkpoints[0][1]

                print("Checkpoint:", current_best_checkpoint)
                print("Score:", current_best_score)


                if (best_score < current_best_score):
                    best_checkpoint = current_best_checkpoint
                    best_score = current_best_score

            loaded = True

            print("Best Checkpoint:", best_checkpoint)
            print("Best Score:", best_score)

        except:
            print("Error finding/loading best checkpoint")
            loaded = False


        # experiment_name = experiment_name + str(run)
        analysis = tune.run(
            'ARS',
            name=experiment_name,
            scheduler=pbt,
            checkpoint_freq=1,
            local_dir="/media/chrx/TURBO/PBT_ARS/PBT_ARS/",
            restore=best_checkpoint if loaded else None,
            keep_checkpoints_num=10,  # Keep only the best checkpoints
            checkpoint_score_attr='episode_reward_mean',  # Metric used to compare checkpoints
            num_samples=2,
# stop=stopper,
            # resume=True,
            stop={  # Stop a single trial if one of the conditions are met
                # "episode_reward_mean": episode_reward_mean_goal,
                "training_iteration": 600},
            config={
                "log_level": "DEBUG",
                "env": "RobotableEnv-v0",
                "action_noise_std": 0.0,
                "noise_stdev": 0.02,  # std deviation of parameter noise
                "num_rollouts": 12,  # number of perturbs to try
                "rollouts_used": 8,  # number of perturbs to keep in gradient estimate
                "sgd_stepsize": 0.02,  # sgd step-size
                "observation_filter": "MeanStdFilter",
                "noise_size": 25000000,
                "eval_prob": 0.05,  # probability of evaluating the parameter rewards
                "report_length": 1,  # how many of the last rewards we average over
                "offset": 0,
                "num_workers": 1,
                "num_gpus": 0
            })

        try:
            all_config = analysis.get_all_configs()
            best_config = analysis.get_best_config()
            print("best config:" + best_config)
        except:
            print("Error printing best config")

        try:

            best_trial = analysis.get_best_trial("episode_reward_mean")
            best_checkpoint_tuple = max(
                analysis.get_trial_checkpoints_paths(best_trial, "episode_reward_mean"))
            print("best reward: ", best_checkpoint_tuple)
        except:
            print("Not enough checkpoints")

        # episode_reward_mean_goal += 0.5;
        # print("Next goal: ", episode_reward_mean_goal)


def replay(experiment_name, checkpoint_path):

    # checkpoint_path = "/media/chrx/TURBO/PBT_ARS/PBT_ARS/PBT_ARS/ARS_RobotableEnv-v0_0_2020-07-31_22-12-23hywaj5sl/checkpoint_436/checkpoint-436"

    tune.run(
        'ARS',

        # restore="/media/chrx/TURBO/PBT_ARS/PBT_ARS/PBT_ARS/ARS_RobotableEnv-v0_1_2020-08-03_13-32-54tu9rfjnw/checkpoint_455/checkpoint-455",
        name=experiment_name,
        local_dir=ray_results,
        scheduler=pbt,
        restore=checkpoint_path,
        stop={  # Stop a single trial if one of the conditions are met
            # "episode_reward_mean": episode_reward_mean_goal,
            "training_iteration": 1},
        config={
            "log_level": "DEBUG",
            "env": "RobotableEnv-v0",
            "action_noise_std": 0.0,
            "noise_stdev": 0.02,  # std deviation of parameter noise
            "num_rollouts": 5,  # number of perturbs to try
            "rollouts_used": 5,  # number of perturbs to keep in gradient estimate
            "sgd_stepsize": 0.02,  # sgd step-size
            "observation_filter": "MeanStdFilter",
            "noise_size": 25000000,
            "eval_prob": 0.05,  # probability of evaluating the parameter rewards
            "report_length": 2,  # how many of the last rewards we average over
            "offset": 0,
            "num_workers": 1,
            "num_gpus": 0
        })


if __name__ == "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', help='Train the robot', type=int, default=0)
    parser.add_argument('--replay', help='play the best', type=int, default=0)
    args = parser.parse_args()

    if (args.train == 1):
        restore("PBT_ARS")
    elif (args.replay == 1):

        exp_dir = glob.glob(ray_results + 'ARS_RobotableEnv-v0*')
        print(ray_results)
        best_score = 0
        worst_score = 9999999
        for dir in exp_dir:

            analysis = Analysis(dir)

            # print(analysis.get_best_config("episode_reward_mean", mode="max"))
            best_log_dir = analysis.get_best_logdir("episode_reward_mean", mode="max")
            if (best_log_dir is None):
                continue
            # print(best_log_dir)
            checkpoints = analysis.get_trial_checkpoints_paths(best_log_dir, "episode_reward_mean")
            # print(checkpoints)
            cleanedList = [x for x in checkpoints if str(x[1]) != 'nan']

            sorted_checkpoints = sorted(cleanedList, key=lambda x:x[1], reverse=True)
            # print(sorted_checkpoints)

            current_worst_checkpoint = sorted_checkpoints[-1][0]
            current_worst_score = sorted_checkpoints[-1][1]
            # print("Worst Checkpoint:", current_worst_checkpoint)
            # print("Worst Score:", current_worst_score)


            current_best_checkpoint = sorted_checkpoints[0][0]
            current_best_score = sorted_checkpoints[0][1]
            print("Best Checkpoint:", current_best_checkpoint)
            print("Best Score:", current_best_score)

            if (best_score < current_best_score):
                best_checkpoint = current_best_checkpoint
                best_score = current_best_score
                print("Best so far:", best_checkpoint)
                print("Best so far:", best_score)

            if (worst_score > current_worst_score):
                worst_checkpoint = current_worst_checkpoint
                worst_score = current_worst_score
                print("Worst so far:", worst_checkpoint)
                print("Worst so far:", worst_score)


        print("Best checkpoint: ", best_checkpoint)
        replay("PBT_TEST", best_checkpoint)

        print("Worst checkpoint: ", worst_checkpoint)
        replay("PBT_TEST", worst_checkpoint)



        # print(ray_results)
        # analysis = Analysis("/media/chrx/TURBO/PBT_ARS/PBT_ARS/PBT_ARS/ARS_RobotableEnv-v0_1_2020-07-30_08-25-18c0rlqk2g") #ray_results)
        # best_log_dir = analysis.get_best_logdir("episode_reward_mean", mode="max")
        # print (best_log_dir)
        # checkpoints = analysis.get_trial_checkpoints_paths(best_log_dir, "episode_reward_mean")
        # print(checkpoints)
        # sorted_checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        # print(sorted_checkpoints)
        #
        # for checkpoint in sorted_checkpoints:
        #    print("Score:", checkpoint[1])
        #    replay("PBT_TESTING", checkpoint[0])

#
# if (best_checkpoint_tuple is None):
#     try:
#         latest_log_dir = glob.glob(absolute_exp_dir + '/ARS_RobotableEnv-v0*')
#         log_dir = sorted(latest_log_dir, key=os.path.getmtime)[-ndx]  # cool python last element syntax
#         print(log_dir)
#
#         latest_checkpoint_dir = glob.glob(log_dir + '/checkpoint*')
#         check_dir = sorted(latest_checkpoint_dir, key=os.path.getmtime)[-1]
#         print(check_dir)
#
#         latest_checkpoint_files = glob.glob(check_dir + '/checkpoint*')
#         best_checkpoint_tuple = sorted(latest_checkpoint_files, key=os.path.getmtime)[-2]
#         print(best_checkpoint_tuple)
#         loaded = True
#     except:
#         print("No checkpoints")
#         loaded = False

# some weird shit cause of experiment directories being made
# ndx = 2
