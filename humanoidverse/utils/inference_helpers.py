import torch
from torch import nn
import os
import copy


def export_policy_as_jit(actor_critic, path, exported_policy_name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def export_decouple_policy_and_estimator_history_as_onnx(
    inference_model, path, exported_policy_name, example_obs_dict
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model["actor"]).to("cpu")
    left_ee_force_estimator = copy.deepcopy(
        inference_model["left_ee_force_estimator"]
    ).to("cpu")
    right_ee_force_estimator = copy.deepcopy(
        inference_model["right_ee_force_estimator"]
    ).to("cpu")

    class PPODecoupleForceEstimatorWrapper(nn.Module):
        def __init__(self, actor, left_ee_force_estimator, right_ee_force_estimator):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPODecoupleForceEstimatorWrapper, self).__init__()
            self.actor = actor
            self.left_ee_force_estimator = left_ee_force_estimator
            self.right_ee_force_estimator = right_ee_force_estimator

        def forward(self, inputs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs, estimator_obs, history_estimated_force = inputs
            left_ee_force_estimator_output = self.left_ee_force_estimator(estimator_obs)
            right_ee_force_estimator_output = self.right_ee_force_estimator(
                estimator_obs
            )
            input_for_actor = torch.cat(
                [
                    actor_obs,
                    left_ee_force_estimator_output,
                    right_ee_force_estimator_output,
                    history_estimated_force,
                ],
                dim=-1,
            )
            return (
                self.actor.act_inference(input_for_actor),
                left_ee_force_estimator_output,
                right_ee_force_estimator_output,
            )

    wrapper = PPODecoupleForceEstimatorWrapper(
        actor, left_ee_force_estimator, right_ee_force_estimator
    )
    example_input_list = [
        example_obs_dict["actor_obs"],
        example_obs_dict["estimator_obs"],
        example_obs_dict["estimated_force_history"],
    ]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=[
            "actor_obs",
            "estimator_obs",
            "estimated_force_history",
        ],  # Specify the input names
        output_names=[
            "action",
            "left_ee_force_estimator_output",
            "right_ee_force_estimator_output",
        ],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )


def export_decouple_policy_and_estimator_as_onnx(
    inference_model, path, exported_policy_name, example_obs_dict
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actor = copy.deepcopy(inference_model["actor"]).to("cpu")
    left_ee_force_estimator = copy.deepcopy(
        inference_model["left_ee_force_estimator"]
    ).to("cpu")
    right_ee_force_estimator = copy.deepcopy(
        inference_model["right_ee_force_estimator"]
    ).to("cpu")

    class PPODecoupleForceEstimatorWrapper(nn.Module):
        def __init__(self, actor, left_ee_force_estimator, right_ee_force_estimator):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPODecoupleForceEstimatorWrapper, self).__init__()
            self.actor = actor
            self.left_ee_force_estimator = left_ee_force_estimator
            self.right_ee_force_estimator = right_ee_force_estimator

        def forward(self, inputs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs, estimator_obs = inputs
            left_ee_force_estimator_output = self.left_ee_force_estimator(estimator_obs)
            right_ee_force_estimator_output = self.right_ee_force_estimator(
                estimator_obs
            )
            input_for_actor = torch.cat(
                [
                    actor_obs,
                    left_ee_force_estimator_output,
                    right_ee_force_estimator_output,
                ],
                dim=-1,
            )
            return (
                self.actor.act_inference(input_for_actor),
                left_ee_force_estimator_output,
                right_ee_force_estimator_output,
            )

    wrapper = PPODecoupleForceEstimatorWrapper(
        actor, left_ee_force_estimator, right_ee_force_estimator
    )
    example_input_list = [
        example_obs_dict["actor_obs"],
        example_obs_dict["estimator_obs"],
    ]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs", "estimator_obs"],  # Specify the input names
        output_names=[
            "action",
            "left_ee_force_estimator_output",
            "right_ee_force_estimator_output",
        ],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )


def export_multi_agent_decouple_policy_as_onnx(
    inference_model,
    path,
    exported_policy_name,
    example_obs_dict,
    body_keys=["lower_body", "upper_body"],
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)

    actors = {
        k: copy.deepcopy(inference_model["actors"][k]).to("cpu") for k in body_keys
    }

    class PPOMADecoupleWrapper(nn.Module):
        def __init__(self, actors, body_keys):
            super(PPOMADecoupleWrapper, self).__init__()
            self.actors = nn.ModuleDict(actors)
            self.body_keys = body_keys

        def forward(self, actor_obs):
            actions = [self.actors[k].act_inference(actor_obs) for k in self.body_keys]
            return torch.cat(actions, dim=-1)

    wrapper = PPOMADecoupleWrapper(actors, body_keys)
    example_input_list = example_obs_dict["actor_obs"]

    torch.onnx.export(
        wrapper,
        example_input_list,
        path,
        verbose=True,
        input_names=["actor_obs"],
        output_names=["action"],
        opset_version=13,
    )
