import torch
from torch import nn
from torch import distributed
from collections import deque


class PipelineBlock:
    """Base class to handle pipeline parallel.

    This class holds a sequential layers to execute forward and backward.

    After forward, it send the output to next block. And it send to grad to
    previous block after backward.
    """

    def __init__(self, layers : nn.Sequential, GPU_id : int) -> None:
        self.layers = layers.to(GPU_id)
        self.GPU_id = GPU_id

        # `inputs`, `outputs` and `grads` are dictionary: {idx : x}, where idx
        # is index of mini batch in the micro batch, x is corresponding tensor.
        #
        # Assume there are n blocks.
        #
        # For block 0, `inputs` are set by `ParallelModel` before pipeline
        # parallel. For other blocks, `inputs` are transferred from previous
        # block.
        #
        # For each block, `outputs` are computed by `inputs`. For block n - 1,
        # `outputs` holds loss.
        #
        # For block n - 1, `grads` holds labels set by `ParallelModel` before
        # pipeline paralle. For other blocks `grads` are transferred from next
        # block.
        #
        # Method `setInput` and `setGrad` are used to set inputs and grads.
        self.inputs : dict = {}
        self.outputs : dict = {}
        self.grads : dict = {}

        # `forward` and `backward` queue contains index of mini batch that
        # needed to be executed later. They are set by method `appendForward`
        # and `appendBackward`.
        self.forward_queue : deque = []
        self.backward_queue : deque = []

        # `forward_done` and `backward_done` means index of last mini batch that
        # have been execute.
        self.forward_done = -1
        self.backward_done = -1

    def resetIdxs(self):
        """reset variables before execution of a micro batch."""
        self.inputs.clear()
        self.outputs.clear()
        self.grads.clear()
        self.forward_queue.clear()
        self.backward_queue.clear()
        self.forward_done = -1
        self.backward_done = -1

    def setInput(self, idx, x):
        self.inputs[idx] =  x

    def setGrad(self, idx, grad):
        self.grads[idx] = grad

    def appendForward(self, idx):
        self.forward_queue.append(idx)

    def appendBackward(self, idx):
        self.backward_queue.append(idx)

    def readyForward(self):
        return len(self.forward_queue) != 0

    def readyBackward(self):
        return len(self.backward_queue) != 0

    def forward(self):
        """Forward of this block."""

        # Index of mini batch this forward executed.
        idx = self.forward_queue.popleft()

        # Transfer input to this device and set states.
        self.inputs[idx].to(self.getGPUid())
        self.inputs[idx].requires_grad_()
        self.inputs[idx].retain_grad()

        # Execute forward.
        output = self.layers(self.inputs[idx])

        # Set corresponding `outputs`.
        self.outputs[idx] = output

        # Record forward done of `idx`.
        self.setForwardDone(idx)
        return output, idx

    def backward(self):
        """Backward of this block"""

        # Index of mini batch this backward executed.
        idx = self.backward_queue.popleft()

        # Transfer grad to this device.
        self.grads[idx].to(self.getGPUid())

        # Execute backward.
        self.outputs[idx].backward(gradient=self.grads[idx], retain_graph=True)

        # Record backward done of `idx`.
        self.setBackwardDone(idx)
        return self.inputs[idx].grad, idx

    def setForwardDone(self, idx):
        self.forward_done = idx

    def setBackwardDone(self, idx):
        self.backward_done = idx

    def done(self, number_minibatches):
        """Return if all task of a micro batch are done."""
        return self.forward_done + 1 == number_minibatches and self.backward_done + 1 == number_minibatches

    def getGPUid(self):
        return self.GPU_id

    def parameters(self):
        return self.layers.parameters()


class PipelineBlockLast(PipelineBlock):
    """Derived class to handle pipeline parallel of last block.

    This revise `PipelineBlock`. `outputs` holds loss and `grads` holds label.
    """

    def __init__(self, layers : nn.Sequential, GPU_id : int, loss_fn) -> None:
        super().__init__(layers, GPU_id)
        self.loss_fn = loss_fn

    def forward(self):
        """Revised forward."""

        # Index of mini batch this forward execute.
        idx = self.forward_queue.popleft()

        # Transfer input to this device and set states.
        self.inputs[idx].to(self.getGPUid())
        self.inputs[idx].requires_grad_()
        self.inputs[idx].retain_grad()

        # Execute forward.
        output = self.layers(self.inputs[idx])

        # Execute loss function.
        loss = self.loss_fn(output, self.grads[idx].to(self.getGPUid()))

        # `outputs` contains loss instead of output.
        self.outputs[idx] = loss

        # Set backward queue.
        self.appendBackward(idx)

        # Record forward done of `idx`.
        self.setForwardDone(idx)
        return loss, idx

    def backward(self):
        """Revised backward."""

        # Index of mini batch this backward execute.
        idx = self.backward_queue.popleft()

        # Get loss.
        loss = self.outputs[idx]

        # Execute backward through loss.
        loss.backward(retain_graph=True)

        # Record backward one of `idx`.
        self.setBackwardDone(idx)
        return self.inputs[idx].grad, idx


class ParallelModel:
    """Parallel model.

    This class warp a model to train it using pipeline and data parallel.
    """

    def __init__(self, model : nn.Module, number_of_layers_each_blockL : list, loss_fn, optim, GPU_ids : list, group, lr=1e-3) -> None:
        """Initialize.

        `optim` should be the class type and it will be initialized in block.
        """

        # Each device holds a block.
        self.blocks : list[PipelineBlock] = []
        self.loss_fn = loss_fn
        len_blocks = len(number_of_layers_each_blockL)

        module_iter = model.children()
        for i in range(len_blocks):
            block = nn.Sequential()
            for _ in range(number_of_layers_each_blockL[i]):
                module = next(module_iter)
                block.append(module)
            if i < len_blocks - 1:
                self.blocks.append(PipelineBlock(block, GPU_ids[i]))
            else:
                self.blocks.append(PipelineBlockLast(block, GPU_ids[i], loss_fn))

        self.optimizer = optim([{"params": block.parameters()} for block in self.blocks], lr=lr)
        self.group = group

    def train(self, train_loader, number_minibatches):
        for x, y in train_loader:
            len_minibatch = x.shape[0] / number_minibatches

            # reset status of each block
            for i in range(len(self.blocks)):
                self.blocks[i].resetIdxs()

            # set input of first block and label of last block
            for i in range(len_minibatch):
                self.blocks[0].appendForward(i)

                left_idx = i * len_minibatch
                right_idx = (i + 1) * len_minibatch
                right_idx = min(right_idx, x.shape[0])

                self.blocks[0].setInput(i, x[left_idx : right_idx].detach())
                self.blocks[len(self.blocks) - 1].setGrad(i, y[left_idx : right_idx])

            done = False

            for _ in range(len(self.blocks)):
                for i in range(len(self.blocks), 0, -1):
                    i = i - 1
                    block = self.blocks[i]
                    if block.readyForward():
                        output, idx = block.forward()
                        if i < len(self.blocks) - 1:
                            self.blocks[i + 1].appendForward(idx)
                            self.blocks[i + 1].setInput(idx, output.clone().detach())

            while not done:
                for i in range(len(self.blocks)):
                    if self.blocks[i].readyBackward():
                        # forward
                        grad, idx = self.blocks[i].backward()
                        if i > 0:
                            self.blocks[i - 1].appendBackward(idx)
                            self.blocks[i - 1].setGrad(idx, grad)
                    else:
                        # backward``
                        output, idx = self.blocks[i].forward()
                        if i < len(self.blocks) - 1:
                            self.blocks[i + 1].appendForward(idx)
                            self.blocks[i + 1].setInput(idx, output.clone().detach())

                done = True
                for i in range(len(self.blocks)):
                    torch.cuda.synchronize(self.blocks[i].getGPUid())
                    done = done and self.blocks[i].done()

            for block in self.blocks:
                for para in block.parameters():
                    distributed.all_reduce(para, group=self.group)
                    para = para / distributed.get_world_size(self.group)

            self.optimizer.step()
