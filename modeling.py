import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomCNN(nn.Module):
    def __init__(self, embedding_dim=32):
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # define cnn model

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # Reduce size by half
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # Reduce size by half
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # Reduce size by half
        )
        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim)
        )

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def minmax_map(x, H, W):
        with torch.no_grad():
            x = x.sum(1)
            max_for_batch = torch.amax(x, dim=[1, 2]).view(-1, 1, 1)
            min_for_batch = torch.amin(x, dim=[1, 2]).view(-1, 1, 1)
            x = (x - min_for_batch) / (max_for_batch - min_for_batch + 1e-8)
        return x.view(-1, 1, H, W)

    def forward(self, inputs):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        outputs: (Batch_size, Sequence_length, Hidden_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem 1: design CNN forward path
        batch_size, seq_len, H, W, C = inputs.size()
        x = inputs.view(batch_size * seq_len, C, H, W)
        x = CustomCNN.minmax_map(x, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(batch_size, seq_len, -1)  # x.shape = (B, T, 마지막 fc레이어 크기=256)
        outputs = x
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs


class Encoder(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, rnn_dropout=0.1, embedding_dim=32):
        super(Encoder, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.cnn = CustomCNN(embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(rnn_dropout)
        self.hidden_dim = hidden_dim
        
        #self.fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        #self.relu = nn.ReLU()
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs, lengths):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths (Batch_size) 배치의 각 성분 input시퀀스 하나하나의 길이 T
        output: (Batch_size, Sequence_length, Hidden_dim)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        embedded_img = self.cnn(inputs)  # x.shape = (B, T, 32)
        x = self.dropout(embedded_img)
        packed_input = pack_padded_sequence(  # packed_input.shape = (B*T, 256)
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.rnn(packed_input)
        # packed_output.shape == (B*T, 2_if_bidirect_else_1*hidden_dim), 만약 input이 (B, T, input_dim)이었다면 output(B, T, hidden_dim)
        # h_n.shape == c_n.shape == (num_layers*(2 if bidirectional else 1), B, hidden_dim)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output.shape == (B, T, 2*hidden_dim)

        # h_n = torch.cat((h_n[-1], h_n[-2]), dim=1)
        # shape: (B, 2*hidden_dim)
        # c_n = torch.cat((c_n[-1], c_n[-2]), dim=1)
        # shape: (B, 2*hidden_dim)

        # h_n = self.fc(h_n)
        # c_n = self.fc(c_n)  # TODO 이거 같은 fc 레이어 써도 괜찮나?
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

        return output, (h_n, c_n)




class Decoder_standard(nn.Module):
    def __init__(
        self, n_vocab=28, hidden_dim=64, num_layers=2, pad_idx=0, rnn_dropout=0.5
    ):
        super(Decoder_standard, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, hidden_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,  # it should be equal to embedding_dim, and also should be hidden_dim since output_of_decoder will become input seq when generating step
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(rnn_dropout)

        self.lm_head = nn.Linear(hidden_dim, n_vocab)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, input_seq, hidden_state, ):
        """
        input_seq: (Batch_size, Sequence_length)
        output: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        T = input_seq.shape[1]
        decoder_outputs = []
        for i in range(T):
            output, hidden_state = self.forward_step(
                input_seq[:, i].unsqueeze(1),
                hidden_state,
            )  # output: B, N_vocab
            decoder_outputs.append(output)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)  # B, T, N_vocab
        return decoder_outputs, hidden_state
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward_step(self, input, hidden_state):
        # input must (B, 1)
        embedded = self.embedding(input)  # B, 1, H
        output, hidden_state = self.rnn(embedded, hidden_state)  # output B, 1, H
        output = self.lm_head(output.view(output.shape[0], -1))  # output B, N_vocab
        return output, hidden_state
    




class Decoder(nn.Module):
    def __init__(
        self, n_vocab=28, hidden_dim=64, num_layers=2, pad_idx=0, rnn_dropout=0.5
    ):
        super(Decoder, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, hidden_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=2
            * hidden_dim,  # it should be equal to embedding_dim, and also should be hidden_dim since output_of_decoder will become input seq when generating step
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(rnn_dropout)

        self.lm_head = nn.Linear(hidden_dim, n_vocab)
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, input_seq, hidden_state, encoder_outputs):
        """
        input_seq: (Batch_size, Sequence_length)
        output: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        T = input_seq.shape[1]
        decoder_outputs = []
        for i in range(T):
            output, hidden_state = self.forward_step(
                input_seq[:, i].unsqueeze(1),
                hidden_state,
                encoder_outputs[:, i, :].unsqueeze(1),
            )  # output: B, N_vocab
            decoder_outputs.append(output)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)  # B, T, N_vocab
        return decoder_outputs, hidden_state
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward_step(self, input, hidden_state, encoder_output):
        # input must (B, 1)
        embedded = self.embedding(input)  # B, 1, H
        # print(encoder_output.shape, embedded.shape)
        rnn_input = torch.concat([encoder_output, embedded], 2)  # B, 1, 2H
        output, hidden_state = self.rnn(rnn_input, hidden_state)  # output B, 1, H
        output = self.lm_head(output.view(output.shape[0], -1))  # output B, N_vocab
        return output, hidden_state


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        num_classes=28,
        hidden_dim=64,
        n_rnn_layers=2,
        rnn_dropout=0.5,
        embedding_dim=32,
    ):
        super(Seq2SeqModel, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_layers=n_rnn_layers,
            rnn_dropout=rnn_dropout,
            embedding_dim=embedding_dim,
        )
        self.decoder = Decoder_standard(
            n_vocab=num_classes,
            hidden_dim=2 * hidden_dim,
            num_layers=n_rnn_layers,
            rnn_dropout=rnn_dropout,
        )
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs, lengths, inp_seq):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        inp_seq: (Batch_size, Sequence_length) - teacher forcing을 위한 정답시퀀스
        logits: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        encoder_outputs, hidden_state = self.encoder(inputs, lengths)

        hidden_state = Seq2SeqModel.hidden_shape_conversion(hidden_state)
        decoder_outputs, hidden_state = self.decoder(
            inp_seq, hidden_state
        )
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return decoder_outputs, hidden_state

    def hidden_shape_conversion(hidden_state):
        h_n, c_n = hidden_state
        bi_layers = h_n.shape[0]
        forward_h_n = h_n[: bi_layers // 2]
        backward_h_n = h_n[bi_layers // 2 :]
        h_n = torch.cat([forward_h_n, backward_h_n], dim=2)

        bi_layers = c_n.shape[0]
        forward_c_n = c_n[: bi_layers // 2]
        backward_c_n = c_n[bi_layers // 2 :]
        c_n = torch.cat([forward_c_n, backward_c_n], dim=2)

        hidden_state = (h_n, c_n) #nlay, B, 2H 
        return hidden_state

    def beam_search(self, inputs, lengths, start_token, max_length, beam_width=4):
        batch_size = inputs.size(0)
        encoder_outputs, hidden_state = self.encoder(inputs, lengths) #1,T,2H_enc
        hidden_state = Seq2SeqModel.hidden_shape_conversion(hidden_state)

        # Initialize the sequences with the start token
        sequences = [[[start_token], 0.0, hidden_state]]

        for _ in range(max_length):
            all_candidates = []
            for seq, score, hidden in sequences:
                decoder_input = seq[-1]
                output, hidden = self.decoder.forward_step(decoder_input, hidden)
                output = F.log_softmax(output, dim=1)  # 1, T
                topk_probs, topk_idx = torch.topk(output, beam_width)

                for i in range(beam_width):
                    candidate = [
                        seq + [topk_idx[:, i].unsqueeze(1)],
                        score - topk_probs[0][i].item(),
                        hidden,
                    ]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda x: x[1])
            sequences = ordered[:beam_width]
        # print("best_score: ", sequences[0][1])
        return torch.cat(
            sequences[0][0][1:], 1
        )  # Return the sequence with the highest score

    def generate(self, inputs, lengths, start_token, max_length, beam_width=4):
        generated_sequences = []
        for i in range(inputs.size(0)):
            seq = self.beam_search(
                inputs[i].unsqueeze(0),
                torch.unsqueeze(lengths[i], 0),
                torch.unsqueeze(start_token[i], 0),
                max_length,
                beam_width,
            )
            generated_sequences.append(seq)
        return torch.cat(generated_sequences, 0)
    











    
class Seq2SeqModelChallenge(nn.Module):
    def __init__(
        self,
        num_classes=28,
        hidden_dim=64,
        n_rnn_layers=2,
        rnn_dropout=0.5,
        embedding_dim=32,
    ):
        super(Seq2SeqModelChallenge, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_layers=n_rnn_layers,
            rnn_dropout=rnn_dropout,
            embedding_dim=embedding_dim,
        )
        self.decoder = Decoder(
            n_vocab=num_classes,
            hidden_dim=2 * hidden_dim,
            num_layers=n_rnn_layers,
            rnn_dropout=rnn_dropout,
        )
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs, lengths, inp_seq):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        inp_seq: (Batch_size, Sequence_length) - teacher forcing을 위한 정답시퀀스
        logits: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        encoder_outputs, hidden_state = self.encoder(inputs, lengths)

        hidden_state = Seq2SeqModelChallenge.hidden_shape_conversion(hidden_state)
        decoder_outputs, hidden_state = self.decoder(
            inp_seq, hidden_state, encoder_outputs
        )
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return decoder_outputs, hidden_state

    def hidden_shape_conversion(hidden_state):
        h_n, c_n = hidden_state
        bi_layers = h_n.shape[0]
        forward_h_n = h_n[: bi_layers // 2]
        backward_h_n = h_n[bi_layers // 2 :]
        h_n = torch.cat([forward_h_n, backward_h_n], dim=2)

        bi_layers = c_n.shape[0]
        forward_c_n = c_n[: bi_layers // 2]
        backward_c_n = c_n[bi_layers // 2 :]
        c_n = torch.cat([forward_c_n, backward_c_n], dim=2)

        hidden_state = (h_n, c_n) #nlay, B, 2H 
        return hidden_state

    def beam_search(self, inputs, lengths, start_token, max_length, beam_width=4):
        batch_size = inputs.size(0)
        encoder_outputs, hidden_state = self.encoder(inputs, lengths) #1,T,2H_enc
        hidden_state = Seq2SeqModelChallenge.hidden_shape_conversion(hidden_state)

        # Initialize the sequences with the start token
        sequences = [[[start_token], 0.0, hidden_state]]

        for _ in range(max_length):
            all_candidates = []
            for seq, score, hidden in sequences:
                decoder_input = seq[-1]
                output, hidden = self.decoder.forward_step(decoder_input, hidden, encoder_outputs[:, min(len(seq)-1, encoder_outputs.shape[1]-1), :].unsqueeze(1))
                output = F.log_softmax(output, dim=1)  # 1, T
                topk_probs, topk_idx = torch.topk(output, beam_width)

                for i in range(beam_width):
                    candidate = [
                        seq + [topk_idx[:, i].unsqueeze(1)],
                        score - topk_probs[0][i].item(),
                        hidden,
                    ]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda x: x[1])
            sequences = ordered[:beam_width]
        # print("best_score: ", sequences[0][1])
        return torch.cat(
            sequences[0][0][1:], 1
        )  # Return the sequence with the highest score

    def generate(self, inputs, lengths, start_token, max_length, beam_width=4):
        generated_sequences = []
        for i in range(inputs.size(0)):
            seq = self.beam_search(
                inputs[i].unsqueeze(0),
                torch.unsqueeze(lengths[i], 0),
                torch.unsqueeze(start_token[i], 0),
                max_length,
                beam_width,
            )
            generated_sequences.append(seq)
        return torch.cat(generated_sequences, 0)
