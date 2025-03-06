from datasets import load_dataset
from envyaml import EnvYAML

from common import get_vector_store

def main():
    env = EnvYAML("env.yaml", env_file=".env")

    # Create vector store
    vector_store = get_vector_store(env)
    vector_store


if __name__ == "__main__":
    main()
