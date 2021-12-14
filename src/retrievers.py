import torch
import torch.nn as nn
import torch.nn.functional as F


#Bi-encoder
class RetrieverBiencoder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        
    def score(self, context, context_mask, responses, responses_mask):

        context_vec = self.bert(context, context_mask)[0][:,0,:]  # [bs,dim]

        batch_size, res_length = response.shape

        responses_vec = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:]  # [bs,dim]
        responses_vec = responses_vec.view(batch_size, 1, -1)

        responses_vec = responses_vec.squeeze(1)        
        context_vec = context_vec.unsqueeze(1)
        dot_product = torch.matmul(context_vec, responses_vec.permute(0, 2, 1)).squeeze()
        return dot_product
    
    def compute_loss(self, context, context_mask, response, response_mask):

        context_vec = self.bert(context, context_mask)[0]  # [bs,dim]

        batch_size, res_length = response.shape

        responses_vec = self.bert(response, response_mask)[0][:,0,:]  # [bs,dim]
        #responses_vec = responses_vec.view(batch_size, 1, -1)
        
        print(context_vec.shape)
        print(responses_vec.shape)

        responses_vec = responses_vec.squeeze(1)
        dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs]
        mask = torch.eye(context.size(0)).to(context_mask.device)
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss

    
#Single Bert Polyencoder 
class RetrieverPolyencoder(nn.Module):
    def __init__(self, bert, max_len = 300, hidden_dim = 768, out_dim = 64, num_layers = 2, dropout=0.1, device=None):
        super().__init__()
        if device==None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.out_dim = out_dim
        self.bert = bert
        
        # Context layers
        self.contextDropout = nn.Dropout(dropout)
        
        # Candidates layers
        self.pos_emb = nn.Embedding(self.max_len, self.hidden_dim)
        self.candidatesDropout = nn.Dropout(dropout)
        
        self.att_dropout = nn.Dropout(dropout)


    def attention(self, q, k, v, vMask=None):
        w = torch.matmul(q, k.transpose(-1, -2))
        if vMask is not None:
            w *= vMask.unsqueeze(1)
            w = F.softmax(w, -1)
        w = self.att_dropout(w)
        score = torch.matmul(w, v)
        return score

    def score(self, context, context_mask, responses, responses_mask):
        """Run the model on the source and compute the loss on the target.

        Args:
            source: An integer tensor with shape (max_source_sequence_length,
                batch_size) containing subword indices for the source sentences.
            target: An integer tensor with shape (max_target_sequence_length,
                batch_size) containing subword indices for the target sentences.

        Returns:
            A scalar float tensor representing cross-entropy loss on the current batch
            divided by the number of target tokens in the batch.
            Many of the target tokens will be pad tokens. You should mask the loss 
            from these tokens using appropriate mask on the target tokens loss.
        """
        batch_size, nb_cand, seq_len = responses.shape
        # Context
        context_encoded = self.bert(context,context_mask)[0][:,0,:]
        pos_emb = self.pos_emb(torch.arange(self.max_len).to(self.device))
        context_att = self.attention(pos_emb, context_encoded, context_encoded, context_mask)

        # Response
        responses_encoded = self.bert(responses.view(-1,responses.shape[2]), responses_mask.view(-1,responses.shape[2]))[0][:,0,:]
        responses_encoded = responses_encoded.view(batch_size,nb_cand,-1)
        response_encoded = self.candidatesFc(response_encoded)
        
        context_emb = self.attention(responses_encoded, context_att, context_att).squeeze() 
        dot_product = (context_emb*responses_encoded).sum(-1)
        
        return dot_product

    
    def compute_loss(self, context, context_mask, response, response_mask):
        """Run the model on the source and compute the loss on the target.

        Args:
            source: An integer tensor with shape (max_source_sequence_length,
                batch_size) containing subword indices for the source sentences.
            target: An integer tensor with shape (max_target_sequence_length,
                batch_size) containing subword indices for the target sentences.

        Returns:
            A scalar float tensor representing cross-entropy loss on the current batch
            divided by the number of target tokens in the batch.
            Many of the target tokens will be pad tokens. You should mask the loss 
            from these tokens using appropriate mask on the target tokens loss.
        """
        batch_size = context.shape[0]
        
        # Context
        context_encoded = self.bert(context,context_mask)[0]
        pos_emb = self.pos_emb(torch.arange(self.max_len).to(self.device))
        context_att = self.attention(pos_emb, context_encoded, context_encoded, context_mask)

        # Response
        response_encoded = self.bert(response, response_mask)[0][:,0,:]
        
        response_encoded = response_encoded.unsqueeze(0).expand(batch_size, batch_size, response_encoded.shape[1]) 
        context_emb = self.attention(response_encoded, context_att, context_att).squeeze() 
        dot_product = (context_emb*response_encoded).sum(-1)
        mask = torch.eye(batch_size).to(self.device)
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss
    
#Double Bert Polyencoder
class RetrieverPolyencoder_double(nn.Module):
    def __init__(self, contextBert, candidateBert, vocab, max_len = 300, hidden_dim = 768, out_dim = 64, num_layers = 2, dropout=0.1, device=None):
        super().__init__()

        if device==None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.out_dim = out_dim
        
        # Context layers
        self.contextBert = contextBert
        self.contextDropout = nn.Dropout(dropout)
        self.contextFc = nn.Linear(self.hidden_dim, self.out_dim)
        
        # Candidates layers
        self.candidatesBert = candidateBert
        self.pos_emb = nn.Embedding(self.max_len, self.hidden_dim)
        self.candidatesDropout = nn.Dropout(dropout)
        self.candidatesFc = nn.Linear(self.hidden_dim, self.out_dim)
        
        self.att_dropout = nn.Dropout(dropout)


    def attention(self, q, k, v, vMask=None):
        w = torch.matmul(q, k.transpose(-1, -2))
        if vMask is not None:
            w *= vMask.unsqueeze(1)
            w = F.softmax(w, -1)
        w = self.att_dropout(w)
        score = torch.matmul(w, v)
        return score

    def score(self, context, context_mask, responses, responses_mask):
        """Run the model on the source and compute the loss on the target.

        Args:
            source: An integer tensor with shape (max_source_sequence_length,
                batch_size) containing subword indices for the source sentences.
            target: An integer tensor with shape (max_target_sequence_length,
                batch_size) containing subword indices for the target sentences.

        Returns:
            A scalar float tensor representing cross-entropy loss on the current batch
            divided by the number of target tokens in the batch.
            Many of the target tokens will be pad tokens. You should mask the loss 
            from these tokens using appropriate mask on the target tokens loss.
        """
        batch_size, nb_cand, seq_len = responses.shape
        # Context
        context_encoded = self.contextBert(context,context_mask)[-1]
        pos_emb = self.pos_emb(torch.arange(self.max_len).to(self.device))
        context_att = self.attention(pos_emb, context_encoded, context_encoded, context_mask)

        # Response
        responses_encoded = self.candidatesBert(responses.view(-1,responses.shape[2]), responses_mask.view(-1,responses.shape[2]))[-1][:,0,:]
        responses_encoded = responses_encoded.view(batch_size,nb_cand,-1)
        
        context_emb = self.attention(responses_encoded, context_att, context_att).squeeze() 
        dot_product = (context_emb*responses_encoded).sum(-1)
        
        return dot_product

    
    def compute_loss(self, context, context_mask, response, response_mask):
        """Run the model on the source and compute the loss on the target.

        Args:
            source: An integer tensor with shape (max_source_sequence_length,
                batch_size) containing subword indices for the source sentences.
            target: An integer tensor with shape (max_target_sequence_length,
                batch_size) containing subword indices for the target sentences.

        Returns:
            A scalar float tensor representing cross-entropy loss on the current batch
            divided by the number of target tokens in the batch.
            Many of the target tokens will be pad tokens. You should mask the loss 
            from these tokens using appropriate mask on the target tokens loss.
        """
        batch_size = context.shape[0]
        
        # Context
        context_encoded = self.contextBert(context,context_mask)[-1]
        pos_emb = self.pos_emb(torch.arange(self.max_len).to(self.device))
        context_att = self.attention(pos_emb, context_encoded, context_encoded, context_mask)

        # Response
        response_encoded = self.candidatesBert(response, response_mask)[-1][:,0,:]
        
        response_encoded = response_encoded.unsqueeze(0).expand(batch_size, batch_size, response_encoded.shape[1]) 
        context_emb = self.attention(response_encoded, context_att, context_att).squeeze() 
        dot_product = (context_emb*response_encoded).sum(-1)
        mask = torch.eye(batch_size).to(self.device)
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss