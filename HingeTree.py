# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# November 2020
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import torch.nn as nn
import torch.autograd
import hingetree_cpp

class HingeTree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.tree_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

    @staticmethod
    def check_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_check_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def fix_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_fix_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def reachability(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_reachability(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.tree_speedtest(inData, False)

class DeterministicHingeTree(HingeTree):
    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_backward_deterministic(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.tree_speedtest(inData, True)

class HingeFern(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.fern_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad
     
    @staticmethod
    def check_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_check_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def fix_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_fix_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def reachability(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_reachability(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.fern_speedtest(inData, False)

class DeterministicHingeFern(HingeFern):
    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_backward_deterministic(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.fern_speedtest(inData, True)

