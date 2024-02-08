import torch.nn as nn

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        """
        Initialize a one-layer fully connected neural network for embedding input features.

        Args:
        - input_dim (int): The dimensionality of the input features.
        - emb_dim (int): The dimensionality of the embedding space.

        """
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        # Define layers for the neural network
        # Linear transformation to embed input features into an 'emb_dim'-dimensional space
        self.embedding_layer = nn.Linear(input_dim, emb_dim)
        self.activation = nn.GELU()  # GELU activation function for non-linearity
        # Linear transformation for further embedding in the same 'emb_dim'-dimensional space
        self.fc_layer = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        """
        Perform forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
        - torch.Tensor: Output tensor after passing through the network.
        """
        # Reshape input tensor if necessary
        x = x.view(-1, self.input_dim)
        # Embed input features into 'emb_dim'-dimensional space
        embedded_x = self.embedding_layer(x)
        # Apply activation function
        embedded_x = self.activation(embedded_x)
        # Further embed the features
        output = self.fc_layer(embedded_x)
        return output
