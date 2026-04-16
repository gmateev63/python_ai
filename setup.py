# Option 1: Environment Variables
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"]       = "https://cloud.langfuse.com"  # or self-hosted URL

# Option 2: Direct client init
from langfuse import Langfuse

langfuse = Langfuse(
    public_key = "pk-lf-...",
    secret_key = "sk-lf-...",
    host       = "https://cloud.langfuse.com"
)