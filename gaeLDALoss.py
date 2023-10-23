# A script implementing the gae LDA loss
import numpy as np
import torch

# Using MSE Loss as the loss for this particular problem
import torch.nn as nn

mseLoss = nn.MSELoss()

# Function that takes in a batch with each element being:
# x_i (original images), 
# x`_i (reconstructed images), 
# y_i (labels)
# 
# It uses these to compute the recontruction loss
# between each x`_i and all other x_i's associated
# with the same label as x_i
def gae_lda_loss(inputs, outputs, labels):

    # Tracking the total loss across the entire batch
    totalLoss = 0

    # Looping over each sample, and finding out which images share a class
    for i in range(labels.shape[0]):
        targLabel = labels[i]
        sameLabelSampleInds = np.where(labels == targLabel)[0]
        numSameLabelSamples = sameLabelSampleInds.shape[0]
        inputsWeCareAbout = inputs[sameLabelSampleInds, :, :, :]
        targSampleReconstruction = outputs[i, :, :, :]
        for j in range(numSameLabelSamples):
            origImage = inputsWeCareAbout[j,:,:,:]
            reconLoss = (1/ numSameLabelSamples) * reconstructionLoss(origImage=origImage, reconImage=targSampleReconstruction)
            totalLoss = totalLoss + reconLoss

    return(totalLoss)


# A hacky way to make it easier to reimplement this 
# with another loss when needed
def reconstructionLoss(origImage, reconImage):

    reconLoss = mseLoss(origImage, reconImage)
    return(reconLoss)