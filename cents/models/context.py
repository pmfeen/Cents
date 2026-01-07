import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional
from .context_registry import register_context_module

class BaseContextModule(nn.Module):
    """
    Base class for context modules. Subclasses must implement the forward method.
    """
    @abstractmethod
    def forward(self, context_vars: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass

@register_context_module("default", "mlp")
class MLPContextModule(BaseContextModule):
    """
    Integrates multiple context variables into a single embedding and provides
    auxiliary classification logits for each variable.

    This module:
        - Learns separate embeddings for each context variable.
        - Concatenates embeddings and projects through an MLP to a shared embedding.
        - Outputs classification logits per context variable for auxiliary loss.

    Attributes:
        context_embeddings (nn.ModuleDict): Embedding layers for each variable.
        mlp (nn.Sequential): MLP to combine embeddings into a single vector.
        classification_heads (nn.ModuleDict): Linear heads for per-variable logits.
    """

    def __init__(self, context_vars: dict[str, int], embedding_dim: int):
        """
        Initialize the ContextModule.

        Args:
            context_vars (Dict[str, int]): Mapping of variable names to category counts.
            embedding_dim (int): Size of each variable's embedding vector.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_categories, embedding_dim)
                for name, num_categories in context_vars.items()
            }
        )

        total_dim = len(context_vars) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        self.classification_heads = nn.ModuleDict(
            {
                var_name: nn.Linear(embedding_dim, num_categories)
                for var_name, num_categories in context_vars.items()
            }
        )

    def forward(
        self, context_vars: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute a combined context embedding and classification logits.

        Args:
            context_vars (Dict[str, Tensor]): Tensors of category indices per variable.

        Returns:
            embedding (Tensor): Combined embedding of shape (batch_size, embedding_dim).
            classification_logits (Dict[str, Tensor]): Logits per variable,
                each of shape (batch_size, num_categories).
        """        
        embeddings = [
            layer(context_vars[name]) for name, layer in self.context_embeddings.items()
        ]

        context_matrix = torch.cat(embeddings, dim=1)
        embedding = self.mlp(context_matrix)

        classification_logits = {
            var_name: head(embedding)
            for var_name, head in self.classification_heads.items()
        }

        return embedding, classification_logits
@register_context_module("default", "sep_mlp")
class SepMLPContextModule(BaseContextModule):
    def __init__(
        self, 
        context_vars: dict[str, int], 
        embedding_dim: int, 
        init_depth: int = 1, 
        mixing_depth: int = 1, 
    ) -> None:
        """
        Initialize SepMLPContextModule.
        
        Args:
            context_vars: Mapping of variable names to category counts.
            embedding_dim: Size of embedding vectors.
            init_depth: Depth of initial MLPs.
            mixing_depth: Depth of mixing MLP.
            continuous_vars: List of continuous variable names.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.continuous_vars = [k for k, v in context_vars.items() if v[0] == "continuous"]
        self.categorical_vars = {k: v[1] for k, v in context_vars.items() if k not in self.continuous_vars}
        print(self.continuous_vars, "CONT VARS")
        print(self.categorical_vars, "CAT VARS")
        self.context_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_categories, embedding_dim)
                for name, num_categories in self.categorical_vars.items()
            }
        )

        # For continuous variables, use a simple linear projection
        self.continuous_projections = nn.ModuleDict(
            {
                name: nn.Linear(1, embedding_dim)
                for name in self.continuous_vars
            }
        )

        self.init_mlps = nn.ModuleDict({
            name: nn.Sequential(*[
                layer
                for _ in range(init_depth)
                for layer in (nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
            ])
            for name in self.categorical_vars.keys()
        })

        # Also create init MLPs for continuous variables
        self.continuous_init_mlps = nn.ModuleDict({
            name: nn.Sequential(*[
                layer
                for _ in range(init_depth)
                for layer in (nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
            ])
            for name in self.continuous_vars
        })

        total_dim = embedding_dim * (len(self.categorical_vars) + len(self.continuous_vars))

        self.mixing_mlp = nn.Sequential(            
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim))

        self.classification_heads = nn.ModuleDict(
            {
                var_name: nn.Linear(embedding_dim, num_categories)
                for var_name, num_categories in self.categorical_vars.items()
            }
        )
        
        # Regression heads for continuous variables (output single value for MSE loss)
        self.regression_heads = nn.ModuleDict(
            {
                var_name: nn.Linear(embedding_dim, 1)
                for var_name in self.continuous_vars
            }
        )

    def forward(self, context_vars):
        #print(self.continuous_vars, "CONT VARS")
        #print(context_vars, "VARS")
        encodings = {}
        
        # Process categorical variables (only those present in context_vars)
        for name, layer in self.context_embeddings.items():
            if name in context_vars:
                encodings[name] = layer(context_vars[name])
        
        #print(encodings, "ENCODINGS")
        # Process continuous variables (only those present in context_vars)
        for name, layer in self.continuous_projections.items():
            if name in context_vars:
                # Reshape to (batch_size, 1) for linear layer
                # Ensure proper shape and gradient flow
                continuous_val = context_vars[name]
                # Handle different input shapes
                if continuous_val.dim() == 0:
                    # Scalar: add batch dimension
                    continuous_val = continuous_val.unsqueeze(0)
                elif continuous_val.dim() == 1:
                    # 1D tensor: add feature dimension
                    continuous_val = continuous_val.unsqueeze(-1)
                # Ensure float type while preserving gradients
                if not continuous_val.is_floating_point():
                    continuous_val = continuous_val.float()    

                if continuous_val.dim() == 1:
                    continuous_val = continuous_val.unsqueeze(-1)
                encodings[name] = layer(continuous_val)

        embeddings = []        
        # Apply init MLPs to categorical variables
        for name, layer in self.init_mlps.items():
            embeddings.append(layer(encodings[name]))
        
        # Apply init MLPs to continuous variables
        for name, layer in self.continuous_init_mlps.items():
            if name in encodings:
                embedding_output = layer(encodings[name])
                # Check for NaN in embedding output
                if torch.isnan(embedding_output).any():
                    raise ValueError(
                        f"NaN detected in embedding output for continuous variable '{name}' "
                        f"after init MLP. This may indicate numerical instability in the MLP layers."
                    )
                embeddings.append(embedding_output)

        if not embeddings:
            raise ValueError("No context variables found in context_vars dict")

        context_matrix = torch.cat(embeddings, dim=1)
        
        # Check for NaN before mixing MLP
        if torch.isnan(context_matrix).any():
            raise ValueError(
                f"NaN detected in context_matrix before mixing MLP. "
                f"This suggests one of the context variable embeddings contains NaN."
            )
        
        embedding = self.mixing_mlp(context_matrix)
        
        # Check for NaN after mixing MLP
        if torch.isnan(embedding).any():
            raise ValueError(
                f"NaN detected in final embedding after mixing MLP. "
                f"Context matrix stats: mean={context_matrix.mean():.4f}, "
                f"std={context_matrix.std():.4f}, "
                f"min={context_matrix.min():.4f}, max={context_matrix.max():.4f}"
            )

        #print(embedding, "post mixing")
        classification_logits = {
            var_name: head(embedding)
            for var_name, head in self.classification_heads.items()
        }
        
        # Regression outputs for continuous variables
        regression_outputs = {
            var_name: head(embedding).squeeze(-1)  # Remove last dim to get (batch_size,)
            for var_name, head in self.regression_heads.items()
        }
        
        # Combine both into a single dict for backward compatibility
        # The training step will need to distinguish between them
        all_outputs = {**classification_logits, **regression_outputs}

        return embedding, all_outputs