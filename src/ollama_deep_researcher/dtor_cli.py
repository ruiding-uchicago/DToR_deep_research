# Standard imports
import argparse
import dataclasses
import datetime
import hashlib
import logging
import os
import pathlib
import sys
import typing

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
    out_dir: pathlib.Path = dataclasses.field(default=pathlib.Path("output"), metadata={
        "help": "Output directory for the research",
        "short": "o"
    })

    FINAL_REPORT_NAME: typing.ClassVar[str] = "final_report.md"
    INTERIM_REPORTS_NAME: typing.ClassVar[str] = "interim_reports"
    SOURCES_NAME: typing.ClassVar[str] = "sources.md"


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

def load_research_topic(research_topic: str) -> str:
    """Load the research topic from a file."""
    if pathlib.Path(research_topic).exists() and pathlib.Path(research_topic).is_file():
        with open(research_topic, 'r') as f:
            return f.read()
    else:
        return research_topic

def setup_output_dir(out_dir: pathlib.Path, research_topic: str,
                     interim_reports_name: str = Config.INTERIM_REPORTS_NAME):
    """Set the output directory for the research."""
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    research_topic = out_dir.joinpath(research_topic)
    research_topic.mkdir(parents=True, exist_ok=True)

    interim_reports = research_topic.joinpath(interim_reports_name)
    interim_reports.mkdir(parents=True, exist_ok=True)

    return research_topic

def truncate_string(value, max_length=200):
    """Truncate a string to a maximum length."""
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + f"... [truncated {len(value) - max_length} chars]"
    return value

def truncate_update(update, max_source_length=200):
    """Truncate long source fields in an update for cleaner logging."""
    if not isinstance(update, dict):
        return update

    source_fields = {'sources_gathered', 'web_research_results', 'complementary_sources_gathered',
                     'local_sources_gathered', 'complementary_web_research_results'}

    truncated = {}
    for key, value in update.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (e.g., {'web_research': {...}})
            nested_truncated = {}
            for nested_key, nested_value in value.items():
                if nested_key in source_fields:
                    # Truncate source fields in nested dicts
                    if isinstance(nested_value, list):
                        nested_truncated[nested_key] = [
                            truncate_string(item, max_source_length)
                            for item in nested_value
                        ]
                    else:
                        nested_truncated[nested_key] = truncate_string(nested_value, max_source_length)
                else:
                    nested_truncated[nested_key] = nested_value
            truncated[key] = nested_truncated
        elif key in source_fields:
            # Truncate source fields at top level
            if isinstance(value, list):
                truncated[key] = [
                    truncate_string(item, max_source_length)
                    for item in value
                ]
            else:
                truncated[key] = truncate_string(value, max_source_length)
        else:
            truncated[key] = value

    return truncated

def write_interim_report(update: dict, out_dir: pathlib.Path, research_topic: str):
    """Write an interim report to the output directory."""
    # Extract iteration number from summary_history
    iteration = update.get("summary_history", 0)[-1].get("iteration", 0)

    # Extract query from summary_history
    query = update.get("summary_history", 0)[-1].get("query", "")
    complementary_query = update.get("summary_history", 0)[-1].get("complementary_query", "")

    # Get the running summary content
    running_summary = update.get("running_summary", "")

    # Format markdown with iteration header
    # Use ## for level 2 header (you can change to # for level 1 if preferred)
    if complementary_query:
        markdown_content = f"## Iteration {iteration}\n\n{query}\n\n{complementary_query}\n\n{running_summary}"
    else:
        markdown_content = f"## Iteration {iteration}\n\n{query}\n\n{running_summary}"

    # Create filename with iteration number for easier tracking
    filename = f"report_iteration_{iteration:02d}.md"

    # Write to file
    report_path = out_dir / Config.INTERIM_REPORTS_NAME / filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logging.info(f"Wrote interim report for iteration {iteration} to {report_path}")

def write_final_report(update: dict, out_dir: pathlib.Path, research_topic: str):
    """Write a final report to the output directory."""
    final_summary = update.get("final_summary", "")
    filename = f"final_report.md"

    report_path = out_dir / Config.FINAL_REPORT_NAME / filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    logging.info(f"Wrote final report to {report_path}")

def write_sources(sources: list, out_dir: pathlib.Path):
    """Write sources to the output directory."""
    sources.sort()
    filename = f"sources.md"

    report_path = out_dir / Config.SOURCES_NAME / filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sources))

    logging.info(f"Wrote sources to {report_path}")

def run_single_mode(research_topic: str, cfg: Config, research_topic_dir: pathlib.Path):
    """Run the single-path research mode."""
    with SqliteSaver.from_conn_string(cfg.checkpoint) as checkpointer:

        app = create_single_research_graph(checkpointer=checkpointer)

        # Check if local RAG should be enabled based on environment or config
        # If USE_LOCAL_RAG is not explicitly set, default to False for CLI
        use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"

        # Generate a unique thread_id for checkpointing based on research topic
        # This allows resuming the same research topic from checkpoints
        thread_id = f"research_{hashlib.md5(research_topic.encode()).hexdigest()[:12]}"

        research_config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "use_local_rag": use_local_rag,
            },
            recursion_limit=cfg.recursion_limit  # recursion_limit is a direct parameter, not in configurable
        )

        app.step_timeout = 86400  # 24 hours in seconds

        sources = []
        for update in app.stream({"research_topic": research_topic}, config=research_config, stream_mode="updates"):
            logging.info(f"GRAPH UPDATE: {truncate_update(update)}")

            if update.get("summarize_sources"):
                write_interim_report(update.get("summarize_sources"), research_topic_dir, cfg.research_topic)
            if update.get("final_summary"):
                write_final_report(update.get("final_summary"), research_topic_dir, cfg.research_topic)

            if update.get("web_research") or update.get("complementary_web_research") or update.get("local_rag_research"):
                web_sources = update.get("web_research", {}).get("sources_gathered", [])
                if web_sources: sources.extend(web_sources)

                complementary_sources = update.get("complementary_web_research", {}).get("complementary_sources_gathered", [])
                if complementary_sources: sources.extend(complementary_sources)

                local_sources = update.get("local_rag_research", {}).get("local_sources_gathered", [])
                if local_sources: sources.extend(local_sources)

        write_sources(sources, research_topic_dir)
        logging.info(f"Total sources: {len(sources)}")

def run_tree_mode(cfg: Config):
    logging.info("Running in tree mode")
    pass

def main():
    cfg = get_args()
    research_topic = load_research_topic(cfg.research_topic)
    research_topic_dir = setup_output_dir(cfg.out_dir, cfg.research_topic)
    if cfg.mode == "single":
        run_single_mode(research_topic, cfg, research_topic_dir)
    elif cfg.mode == "tree":
        run_tree_mode(cfg)
    else:
        logging.error(f"Invalid mode specified: {cfg.mode}. Must be 'single' or 'tree'")
        sys.exit(1)


if __name__ == '__main__':
    main()