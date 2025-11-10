# Standard imports
import argparse
import dataclasses
import hashlib
import logging
import os
import pathlib
import sys

# Add src directory to Python path for direct script execution
# This allows Python to find ollama_deep_researcher package
src_dir = pathlib.Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Application imports
from ollama_deep_researcher.graph import create_single_research_graph
from ollama_deep_researcher.configuration import Configuration
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


@dataclasses.dataclass
class Config:
    research_topic: str = dataclasses.field(metadata={
        "help": "String topic of research or a file path to a topic file",
        "short": "t"
    })
    checkpoint: pathlib.Path = dataclasses.field(metadata={
        "help": "Path to checkpoint file",
        "short": "c"
    })
    mode: str = dataclasses.field(default="single", metadata={
        "help": "Research mode: 'single' for single-path research or 'tree' for Deep Tree of Research",
        "choices": ["single", "tree"],
        "short": "m"
    })
    recursion_limit: int = dataclasses.field(default=500, metadata={
        "help": "Recursion limit for the research",
        "short": "r"
    })


def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Run Deep Tree of Research (DToR)")

    for field in dataclasses.fields(Config):
        # Build argument flags (long and short)
        long_flag = f"--{field.name.replace('_', '-')}"
        flags = [long_flag]

        # Add short flag if specified in metadata
        if "short" in field.metadata:
            short_flag = f"-{field.metadata['short']}"
            flags.append(short_flag)

        kwargs = {
            "help": field.metadata.get("help", ""),
            "default": field.default
        }

        # Handle choices if specified in metadata
        if "choices" in field.metadata:
            kwargs["choices"] = field.metadata["choices"]

        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type

        parser.add_argument(*flags, **kwargs)

    return parser

def get_args():
    """Get arguments from the command line and return a Config object for
    consistent parsing."""
    arg_parser = create_args()
    args = arg_parser.parse_args()
    for key,val in vars(args).items():
        logging.info(f"{key}: {val}")
    cfg = Config(**vars(args))
    return cfg

def run_single_mode(cfg: Config):
    """Run the single-path research mode."""
    with SqliteSaver.from_conn_string(cfg.checkpoint) as checkpointer:

        app = create_single_research_graph(checkpointer=checkpointer)

        # Check if local RAG should be enabled based on environment or config
        # If USE_LOCAL_RAG is not explicitly set, default to False for CLI
        use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"

        # Generate a unique thread_id for checkpointing based on research topic
        # This allows resuming the same research topic from checkpoints
        thread_id = f"research_{hashlib.md5(cfg.research_topic.encode()).hexdigest()[:12]}"

        research_config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "use_local_rag": use_local_rag,
            },
            recursion_limit=cfg.recursion_limit  # recursion_limit is a direct parameter, not in configurable
        )

        app.step_timeout = 86400  # 24 hours in seconds
        app.invoke({"research_topic": cfg.research_topic}, config=research_config)

def run_tree_mode(cfg: Config):
    logging.info("Running in tree mode")
    pass

def main():
    cfg = get_args()
    if cfg.mode == "single":
        run_single_mode(cfg)
    elif cfg.mode == "tree":
        run_tree_mode(cfg)
    else:
        logging.error(f"Invalid mode specified: {cfg.mode}. Must be 'single' or 'tree'")
        sys.exit(1)


if __name__ == '__main__':
    main()