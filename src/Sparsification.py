import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM , AutoModel

def pruning(model,layer_name, amount):
  parameters_to_prune = []

  for name, module in model.named_modules():
        if isinstance(module, layer_name):
            parameters_to_prune.append((module, 'weight'))
  prune.global_unstructured(
      parameters_to_prune,
      pruning_method=prune.L1Unstructured,
      amount=amount,
  )

  return model


def print_sparisity(model,layer_name):
  sparisity = 0.0
  nelement = 0.0
  for name, module in model.named_modules():
    if isinstance(module, layer_name):
      sparisity += torch.sum(module.weight == 0)
      nelement += module.weight.nelement()

  print(
    "Global sparsity: {:.2f}%".format(
        100. * float( sparisity
        )
        / float( nelement
        )
    )
)

model_bert = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
model_bart = AutoModel.from_pretrained("facebook/bart-base")

layer_type = [torch.nn.modules.linear.Linear,transformers.pytorch_utils.Conv1D]
models = {'bert': (model_bert, layer_type[0]), 'gpt-2':(model_gpt2, layer_type[1]), 'bart':(model_bart,layer_type[0])}
model_names = []

for key in models.keys():
  print('Starting pruning {}'.format(key))
  model = models[key][0]
  layer = models[key][1]
  for sparsity in [0.1, 0.5, 0.9, 0.95, 0.99]:
    pruned_model = pruning(model,layer,sparsity)
    print_sparisity(pruned_model,layer)
    pruned_model.save_pretrained('/content/models/{}-{}'.format(key,sparsity))
    model_names.append('/content/models/{}-{}'.format(key,sparsity))
    print('The size of the model {}-{} is: {}'.format(key,sparsity,pruned_model.num_parameters()))


