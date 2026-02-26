import torch
from torch import nn
import numpy as np

def transferWeightsAndBiases(nettype:str, source:nn.Module, target:nn.Module)->nn.Module:
    """
    Transfer source-->target
    """
    # source.state_dict().keys()
    # target.state_dict().keys()

    print("This is your source network:")
    print(source)

    print("This is your target network:")
    print(target)

    def transferBias(A, B):
        for i in range(len(A)):
            B[i] = A[i]

    def transferWeight(A, B):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[i, j] = A[i, j]

    if nettype == 'pi':
        transferWeight(source.state_dict()['predict.0.weight'], target.mlp_extractor.policy_net.state_dict()['0.weight'])
        transferWeight(source.state_dict()['predict.2.weight'], target.mlp_extractor.policy_net.state_dict()['2.weight'])
        transferWeight(source.state_dict()['predict.4.weight'], target.action_net.state_dict()['weight'])

        transferBias(source.state_dict()['predict.0.bias'], target.mlp_extractor.policy_net.state_dict()['0.bias'])
        transferBias(source.state_dict()['predict.2.bias'], target.mlp_extractor.policy_net.state_dict()['2.bias'])
        transferBias(source.state_dict()['predict.4.bias'], target.action_net.state_dict()['bias'])
    
    elif nettype == 'vf':
        transferWeight(source.state_dict()['predict.0.weight'], target.mlp_extractor.value_net.state_dict()['0.weight'])
        transferWeight(source.state_dict()['predict.2.weight'], target.mlp_extractor.value_net.state_dict()['2.weight'])
        transferWeight(source.state_dict()['predict.4.weight'], target.value_net.state_dict()['weight'])

        transferBias(source.state_dict()['predict.0.bias'], target.mlp_extractor.value_net.state_dict()['0.bias'])
        transferBias(source.state_dict()['predict.2.bias'], target.mlp_extractor.value_net.state_dict()['2.bias'])
        transferBias(source.state_dict()['predict.4.bias'], target.value_net.state_dict()['bias'])

    return target

def transferWeightsAndBiases_td3(nettype:str, source:nn.Module, target:nn.Module)->nn.Module:
    """
    Transfer source-->target
    """
    # source.state_dict().keys()
    # target.state_dict().keys()

    # print("This is your source network:")
    # print(source)

    # print("This is your target network:")
    # print(target)

    for name, param in target.named_parameters():
        if '0.weight' in name:
            for i in range(param.data.size(0)):
                for j in range(param.data.size(1)):
                    param.data[i, j] = source.state_dict()['predict.0.weight'][i, j]
    for name, param in target.named_parameters():
        if '2.weight' in name:
            for i in range(param.data.size(0)):
                for j in range(param.data.size(1)):
                    param.data[i, j] = source.state_dict()['predict.2.weight'][i, j]
    for name, param in target.named_parameters():
        if '4.weight' in name:
            for i in range(param.data.size(0)):
                for j in range(param.data.size(1)):
                    param.data[i, j] = source.state_dict()['predict.4.weight'][i, j]
    for name, param in target.named_parameters():
        if '0.bias' in name:
            for i in range(param.data.size(0)):
                param.data[i] = source.state_dict()['predict.0.bias'][i]
    for name, param in target.named_parameters():
        if '2.bias' in name:
            for i in range(param.data.size(0)):
                param.data[i] = source.state_dict()['predict.2.bias'][i]
    for name, param in target.named_parameters():
        if '4.bias' in name:
            for i in range(param.data.size(0)):
                param.data[i] = source.state_dict()['predict.4.bias'][i]

    return target

def transferWeightsAndBiases_sac(nettype:str, source:nn.Module, target:nn.Module)->nn.Module:
    """
    Transfer source-->target
    """


    # print("This is your source network:")
    # print(source)

    # print("This is your target network:")
    # print(target)

    # breakpoint()
    # print(source.state_dict().keys())
    # print(target.state_dict().keys())

    for name, param in target.named_parameters():
        if '0.weight' in name:
            for i in range(param.data.size(0)):
                for j in range(param.data.size(1)):
                    param.data[i, j] = source.state_dict()['predict.0.weight'][i, j]
        if '2.weight' in name:
            for i in range(param.data.size(0)):
                for j in range(param.data.size(1)):
                    param.data[i, j] = source.state_dict()['predict.2.weight'][i, j]
        if '4.weight' in name:
            for i in range(param.data.size(0)):
                for j in range(param.data.size(1)):
                    param.data[i, j] = source.state_dict()['predict.4.weight'][i, j]
        if ('mu.weight' in name): #or ('log_std.weight' in name):
            for i in range(param.data.size(0)):
                for j in range(param.data.size(1)):
                    param.data[i, j] = source.state_dict()['predict.4.weight'][i, j]
        if '0.bias' in name:
            for i in range(param.data.size(0)):
                param.data[i] = source.state_dict()['predict.0.bias'][i]
        if '2.bias' in name:
            for i in range(param.data.size(0)):
                param.data[i] = source.state_dict()['predict.2.bias'][i]
        if '4.bias' in name:
            for i in range(param.data.size(0)):
                param.data[i] = source.state_dict()['predict.4.bias'][i]
        if ('mu.bias' in name): #or ('log_std.bias' in name):
            for i in range(param.data.size(0)):
                param.data[i] = source.state_dict()['predict.4.bias'][i]


    return target