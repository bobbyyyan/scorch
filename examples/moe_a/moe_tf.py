import tensorflow as tf
import numpy as np
import torch


# Define the MoE model
class SparseMoE(tf.keras.Model):
    def __init__(self, in_features, num_experts, out_features):
        super(SparseMoE, self).__init__()
        self.gates = tf.keras.layers.Dense(num_experts, activation="softmax")
        self.experts = [tf.keras.layers.Dense(out_features) for _ in range(num_experts)]

    def call(self, inputs, training=False):
        gate_outputs = self.gates(inputs)
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=2)
        outputs = tf.reduce_sum(gate_outputs[:, :, tf.newaxis] * expert_outputs, axis=2)
        return outputs


# Load PyTorch weights
state_dict = torch.load("weights/moe_newsgroups_weights.pth")
weights = {k: v.numpy() for k, v in state_dict.items()}

# Initialize the model and load the weights
model = SparseMoE(X.shape[1], 10, len(set(Y)))
model.build((None, X.shape[1]))

# Assign the weights
for layer, (weight_name, bias_name) in zip(
    [model.gates] + model.experts,
    [("gates.weight", "gates.bias")]
    + [(f"experts.{i}.weight", f"experts.{i}.bias") for i in range(10)],
):
    layer.kernel.assign(weights[weight_name].T)
    layer.bias.assign(weights[bias_name])

# Perform inference
pred = model(X, training=False)
