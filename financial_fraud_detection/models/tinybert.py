import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class TinyBertForSequenceClassification(BertPreTrainedModel):
    """
    TinyBERT model for sequence classification
    Adapted from the TinyBERT paper: https://arxiv.org/pdf/1909.10351.pdf
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_embedding_output(self, **inputs):
        """Get embedding layer output for distillation"""
        return self.bert.embeddings(**inputs)

    def get_attention_outputs(self, **inputs):
        """Get attention matrices for distillation"""
        outputs = self.bert(
            **inputs,
            output_attentions=True,
        )
        return outputs.attentions

    def get_hidden_states(self, **inputs):
        """Get hidden states for distillation"""
        outputs = self.bert(
            **inputs,
            output_hidden_states=True,
        )
        return outputs.hidden_states


class TinyBertDistillationLoss(nn.Module):
    """
    Loss function for TinyBERT distillation
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def embedding_loss(self, student_embedding, teacher_embedding, attention_mask=None):
        """
        Calculate embedding layer distillation loss
        
        Args:
            student_embedding: Student model embedding output
            teacher_embedding: Teacher model embedding output
            attention_mask: Attention mask for padding
            
        Returns:
            torch.Tensor: Embedding distillation loss
        """
        # Check if dimensions match
        if student_embedding.size(-1) != teacher_embedding.size(-1):
            # Create a linear transformation to match dimensions
            device = student_embedding.device
            hidden_size_student = student_embedding.size(-1)
            hidden_size_teacher = teacher_embedding.size(-1)
            
            # Create a projection matrix
            projection = torch.nn.Linear(hidden_size_student, hidden_size_teacher, bias=False).to(device)
            
            # Project student embeddings to teacher dimension
            student_embedding = projection(student_embedding)
        
        if attention_mask is not None:
            # Apply attention mask to focus on non-padding tokens
            # Make sure to expand to the correct dimension
            student_mask = attention_mask.unsqueeze(-1).expand_as(student_embedding)
            teacher_mask = attention_mask.unsqueeze(-1).expand_as(teacher_embedding)
            
            student_embedding = student_embedding * student_mask
            teacher_embedding = teacher_embedding * teacher_mask
            
        return self.mse_loss(student_embedding, teacher_embedding)

    def attention_loss(self, student_attentions, teacher_attentions):
        """
        Calculate attention layer distillation loss
        
        Args:
            student_attentions: List of student model attention matrices
            teacher_attentions: List of teacher model attention matrices
            
        Returns:
            torch.Tensor: Attention distillation loss
        """
        loss = 0.0
        student_layers = len(student_attentions)
        teacher_layers = len(teacher_attentions)
        
        # Map teacher layers to student layers (uniform strategy)
        for student_layer, teacher_layer in enumerate(range(0, teacher_layers, teacher_layers // student_layers)):
            if teacher_layer < teacher_layers:
                student_attention = student_attentions[student_layer]
                teacher_attention = teacher_attentions[teacher_layer]
                
                # Calculate MSE loss for each attention head
                for i in range(student_attention.size(1)):
                    student_head = student_attention[:, i]
                    teacher_head = teacher_attention[:, i % teacher_attention.size(1)]
                    
                    # Ensure dimensions match for attention matrices
                    if student_head.size() != teacher_head.size():
                        # Resize student attention to match teacher attention
                        # This is typically needed when sequence lengths or batch sizes differ
                        # We'll use interpolation to resize
                        if len(student_head.size()) == 3:  # [batch_size, seq_len, seq_len]
                            # Resize each batch separately
                            resized_student_heads = []
                            for b in range(student_head.size(0)):
                                # Convert to 2D for interpolation
                                s_head = student_head[b].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                                # Use interpolate to resize
                                s_head = torch.nn.functional.interpolate(
                                    s_head,
                                    size=(teacher_head.size(1), teacher_head.size(2)),
                                    mode='bilinear',
                                    align_corners=False
                                )
                                resized_student_heads.append(s_head.squeeze(0).squeeze(0))  # [seq_len, seq_len]
                            
                            # Stack back to batch
                            student_head = torch.stack(resized_student_heads, dim=0)
                    
                    # Now calculate MSE loss
                    loss += self.mse_loss(student_head, teacher_head)
        
        return loss / student_layers

    def hidden_state_loss(self, student_hidden_states, teacher_hidden_states):
        """
        Calculate hidden state distillation loss
        
        Args:
            student_hidden_states: List of student model hidden states
            teacher_hidden_states: List of teacher model hidden states
            
        Returns:
            torch.Tensor: Hidden state distillation loss
        """
        loss = 0.0
        student_layers = len(student_hidden_states) - 1  # Exclude embedding layer
        teacher_layers = len(teacher_hidden_states) - 1  # Exclude embedding layer
        
        # Create projection layers for each student layer
        projections = {}
        
        # Map teacher layers to student layers (uniform strategy)
        for student_layer, teacher_layer in enumerate(range(0, teacher_layers, teacher_layers // student_layers)):
            if teacher_layer < teacher_layers:
                student_state = student_hidden_states[student_layer + 1]  # +1 to skip embedding layer
                teacher_state = teacher_hidden_states[teacher_layer + 1]  # +1 to skip embedding layer
                
                # Check if dimensions match
                if student_state.size(-1) != teacher_state.size(-1):
                    # Create a projection layer if not already created
                    if student_layer not in projections:
                        device = student_state.device
                        hidden_size_student = student_state.size(-1)
                        hidden_size_teacher = teacher_state.size(-1)
                        
                        # Create a projection matrix
                        projections[student_layer] = torch.nn.Linear(
                            hidden_size_student, 
                            hidden_size_teacher, 
                            bias=False
                        ).to(device)
                    
                    # Project student hidden state to teacher dimension
                    student_state = projections[student_layer](student_state)
                
                # Calculate MSE loss
                loss += self.mse_loss(student_state, teacher_state)
        
        return loss / student_layers

    def prediction_loss(self, student_logits, teacher_logits):
        """
        Calculate prediction layer distillation loss
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            
        Returns:
            torch.Tensor: Prediction distillation loss
        """
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)

    def forward(self, student_outputs, teacher_outputs, attention_mask=None):
        """
        Calculate total distillation loss
        
        Args:
            student_outputs: Dictionary containing student model outputs
            teacher_outputs: Dictionary containing teacher model outputs
            attention_mask: Attention mask for padding
            
        Returns:
            torch.Tensor: Total distillation loss
        """
        # Embedding layer distillation
        emb_loss = self.embedding_loss(
            student_outputs['embedding_output'],
            teacher_outputs['embedding_output'],
            attention_mask
        )
        
        # Attention layer distillation
        att_loss = self.attention_loss(
            student_outputs['attention_outputs'],
            teacher_outputs['attention_outputs']
        )
        
        # Hidden state distillation
        hid_loss = self.hidden_state_loss(
            student_outputs['hidden_states'],
            teacher_outputs['hidden_states']
        )
        
        # Prediction layer distillation
        pred_loss = self.prediction_loss(
            student_outputs['logits'],
            teacher_outputs['logits']
        )
        
        # Total loss (weighted sum)
        # Weights can be adjusted based on performance
        total_loss = 0.1 * emb_loss + 0.7 * att_loss + 0.1 * hid_loss + 0.1 * pred_loss
        
        return {
            'total_loss': total_loss,
            'embedding_loss': emb_loss,
            'attention_loss': att_loss,
            'hidden_loss': hid_loss,
            'prediction_loss': pred_loss
        }


def create_tinybert_config(teacher_config, num_hidden_layers=4, hidden_size=312, intermediate_size=1200):
    """
    Create TinyBERT configuration based on teacher configuration
    
    Args:
        teacher_config: Teacher model configuration
        num_hidden_layers (int): Number of hidden layers in TinyBERT
        hidden_size (int): Hidden size in TinyBERT
        intermediate_size (int): Intermediate size in TinyBERT
        
    Returns:
        BertConfig: TinyBERT configuration
    """
    return BertConfig(
        vocab_size=teacher_config.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=12,
        intermediate_size=intermediate_size,
        hidden_act=teacher_config.hidden_act,
        hidden_dropout_prob=teacher_config.hidden_dropout_prob,
        attention_probs_dropout_prob=teacher_config.attention_probs_dropout_prob,
        max_position_embeddings=teacher_config.max_position_embeddings,
        type_vocab_size=teacher_config.type_vocab_size,
        initializer_range=teacher_config.initializer_range,
        layer_norm_eps=teacher_config.layer_norm_eps,
        pad_token_id=teacher_config.pad_token_id,
        num_labels=teacher_config.num_labels,
    )
