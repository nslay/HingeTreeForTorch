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
from HingeTree import HingeTree, HingeFern, DeterministicHingeTree, DeterministicHingeFern

class RandomHingeForest(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "deterministic", "init_type", "treeType" ]

    in_channels: int
    out_channels: int
    depth: int
    extra_outputs: int
    deterministic: bool
    init_type: str

    def __init__(self, in_channels: int, out_channels: int, depth: int, extra_outputs: int = 1, deterministic: bool = True, init_type: str = "random"):
        super(RandomHingeForest, self).__init__()

        self.treeType = HingeTree

        if deterministic:
            self.treeType = DeterministicHingeTree

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.deterministic = deterministic
        self.init_type = init_type

        thresholds = 6.0*torch.rand([out_channels, 2**depth - 1]) - 3.0

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=in_channels)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=thresholds.dtype)
            ordinals -= in_channels * (ordinals / in_channels).type(torch.int32)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, 2**depth])

        self.treeType.fix_thresholds(thresholds, ordinals, weights)

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return self.treeType.apply(x, self.thresholds, self.ordinals, self.weights)

    def reachability(self, x):
        return self.treeType.reachability(x, self.thresholds, self.ordinals, self.weights)

    def check_thresholds(self):
        return self.treeType.check_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def fix_thresholds(self):
        return self.treeType.fix_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def load_state_dict(self, *args, **kwargs):
        super(RandomHingeForest, self).load_state_dict(*args, **kwargs)

        # NOTE: All of this gets redundantly checked in the C++ code... but the C++ code provides no error messages!
        if self.thresholds.dim() != 2 or (self.thresholds.shape[1] & (self.thresholds.shape[1]+1)) != 0 or self.thresholds.shape != self.ordinals.shape:
            raise RuntimeError("Unexpected threshold/ordinal shapes.")

        if self.weights.dim() < 2 or self.weights.shape[0] != self.thresholds.shape[0] or self.weights.shape[1] != self.thresholds.shape[1]+1:
            raise RuntimeError("Unexpected weight shape.")

class RandomHingeFern(nn.Module):
    __constants__ = [ "in_channels", "out_channels", "depth", "extra_outputs", "deterministic", "init_type", "treeType" ]

    in_channels: int
    out_channels: int
    depth: int
    extra_outputs: int
    deterministic: bool
    init_type: str

    def __init__(self, in_channels: int, out_channels: int, depth: int, extra_outputs: int = 1, deterministic: bool = True, init_type: str = "random"):
        super(RandomHingeFern, self).__init__()

        self.treeType = HingeFern

        if deterministic:
            self.treeType = DeterministicHingeFern

        # Meta data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.extra_outputs = extra_outputs
        self.deterministic = deterministic
        self.init_type = init_type

        thresholds = 6.0*torch.rand([out_channels, depth]) - 3.0

        if init_type == "random":
            ordinals = torch.randint_like(thresholds, low=0, high=in_channels)
        elif init_type == "sequential":
            ordinals = torch.arange(thresholds.numel(), dtype=thresholds.dtype)
            ordinals -= in_channels * (ordinals / in_channels).type(torch.int32)
            ordinals = torch.reshape(ordinals, thresholds.shape)
        else:
            raise RuntimeError(f"Unknown init_type {init_type}. Must be one of 'random' or 'sequential'.")

        if extra_outputs > 1:
            weights = torch.randn([out_channels, 2**depth, extra_outputs])
        else:
            weights = torch.randn([out_channels, 2**depth])

        self.treeType.fix_thresholds(thresholds, ordinals, weights) # Doesn't do anything for ferns

        self.weights = nn.Parameter(weights, requires_grad=True)
        self.thresholds = nn.Parameter(thresholds, requires_grad=True)
        self.ordinals = nn.Parameter(ordinals, requires_grad=False)

    def forward(self, x):
        return self.treeType.apply(x, self.thresholds, self.ordinals, self.weights)

    def reachability(self, x):
        return self.treeType.reachability(x, self.thresholds, self.ordinals, self.weights)

    def check_thresholds(self):
        return self.treeType.check_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def fix_thresholds(self):
        return self.treeType.fix_thresholds(self.thresholds.data, self.ordinals.data, self.weights.data)

    def load_state_dict(self, *args, **kwargs):
        super(RandomHingeFern, self).load_state_dict(*args, **kwargs)

        # NOTE: All of this gets redundantly checked in the C++ code... but the C++ code provides no error messages!
        if self.thresholds.dim() != 2 or self.thresholds.shape != self.ordinals.shape:
            raise RuntimeError("Unexpected threshold/ordinal shapes.")

        if self.weights.dim() < 2 or self.weights.shape[0] != self.thresholds.shape[0] or self.weights.shape[1] != 2**self.thresholds.shape[1]:
            raise RuntimeError("Unexpected weight shape.")

