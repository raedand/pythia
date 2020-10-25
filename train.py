from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch
import os

from helpers import *
from model import *

# most of this is taken from https://github.com/thundercomb/pytorch-char-rnn/
# also https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
# and of course it wouldn't be a true char-rnn without citing andrej kaparthy
# https://karpathy.github.io/2015/05/21/rnn-effectiveness/

# i'm training on a macbook pro 2017 so u shouldn't need a gpu
# i'm working on optimizing this

# anyways if ur planning on using any of this (which like why...)
# i request that it only be trained on texts by people of color or lgbt people
# this model is anti-straight and anti-white
# i will sue you i need tuition money
# and read the license

# main training function

def train():
    # characters to indices
    batches = [char_to_index[char] for char in all_texts]

    # chunk to sequences
    batches = list(chunks(batches, seq_length + 1))

    # sequences to batches
    batches = list(chunks(batches, batch_size))

    # batches to tensors
    batches = [torch.LongTensor(batch).transpose_(0, 1) for batch in batches]

    # name the optimizer, hidden and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    hidden = model.init_hidden(batch_size)

    # initialize losses
    # i think
    all_losses = []
    best_ep_loss = float('inf')

# the if is so that it won't train when it's called in generate
# actually it's not called in generate but formatting this is a pain
    if __name__ == '__main__':
        try:
            epoch_progress = tqdm(range(1, epoch_count + 1))
            best_tl_loss = float('inf')

            for epoch in epoch_progress:
                random_state.shuffle(batches)
                batches_progress = batches
                best_loss = float('inf')

                for batch, batch_tensor in enumerate(batches_progress):
                    # i don't know if i'm supposed to
                    # zero out the model or the optimizer
                    model.zero_grad()

                    inp = Variable(batch_tensor[:-1])
                    target = Variable(batch_tensor[1:].contiguous().view(-1))

                    # calculate best_loss
                    output, _ = model(inp, hidden)
                    loss = criterion(output, target)

                    # very important part
                    loss.backward()
                    optimizer.step()

                    # do not fucking use data oh my god
                    loss = loss.item()
                    best_tl_loss = min(best_tl_loss, loss)
                    all_losses.append(loss)

                    # if you uncomment this its gonna print every batch progress
                    # so don't do it it's annoying
                    # batches_progress.set_postfix(loss='{:.03f}'.format(loss))

                    # i need to change this to stop making a million checkpoints

                    # if loss < 1.3 and loss == best_tl_loss:
                    #    checkpoint_path = os.path.join(args.output, 'checkpoint_tl_')
                    #    checkpoint_path = checkpoint_path + str('{:.03f}'.format(loss)) + '.cp'
                    #    torch.save({
                    #        'model': model.state_dict(),
                    #        'optimizer': optimizer.state_dict()
                    #        }, checkpoint_path)

                # print the progress
                epoch_progress.set_postfix(loss ='{:.03f}'.format(loss))
                best_ep_loss = min(best_ep_loss, loss)

                # if loss == best_ep_loss:
                #    checkpoint_path = os.path.join(args.output, 'checkpoint_ep_')
                #    checkpoint_path = checkpoint_path + str('{:.03f}'.format(loss)) + '.cp'
                #    torch.save({
                #        'model': model.state_dict(),
                #        'optimizer': optimizer.state_dict()
                #        }, checkpoint_path)

        except KeyboardInterrupt:
            pass

        # final save
        print('saving...')
        final_path = os.path.join(args.output, 'final_checkpoint_')
        final_path = final_path + str('{:.03f}'.format(loss)) + '.cp'
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, final_path)

# not sure why we are defining the model down here but whatever

model = RNN(chars_len, hidden_size, chars_len, n_layers, dropout)
train()
