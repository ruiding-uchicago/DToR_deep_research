"""
Chainlit integration for the Deep Tree of Research (DToR) LangGraph.

This module provides a Chainlit UI for interacting with the DToR research graph.
Users can input research topics and select between single-path or DToR modes.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Set, Dict

import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig

from ollama_deep_researcher.dtor_graph import create_main_graph
from ollama_deep_researcher.dtor_state import DToRStateInput, DToRStateOutput

# Constants
RECURSION_LIMIT = 500
SUMMARY_NODES = ["finalize_summary", "synthesize_branch", "summarize_sources"]
FIRST_ITERATION_NODES = ["generate_query", "local_rag_research"]
SUBSEQUENT_ITERATION_NODES = ["local_rag_research", "generate_query"]
ROUTE_RESEARCH_NODE = "route_research"
CHANNELWRITE_PATTERNS = ["Channelwrite", "ChannelWrite"]
INTERNAL_NODES = ["__start__", "__end__", "write", "read", "", "...", "none"]

# Mapping of node names to user-friendly descriptions
NODE_DESCRIPTIONS = {
    # Main graph nodes
    "init_session": "🔧 Initializing research session",
    "single_path": "🔍 Starting single-path research",
    "dtor_mode": "🌳 Starting Deep Tree of Research",
    # DToR graph nodes
    "diversify_query": "💡 Diversifying initial research query",
    "select_next_branch": "🌿 Selecting next research branch",
    "analyze_research": "📊 Analyzing research findings",
    "generate_queries": "❓ Generating follow-up research queries",
    "research_node": "🔬 Conducting research on current topic",
    "synthesize_branch": "📝 Synthesizing branch findings",
    "synthesize_final": "✨ Creating final research synthesis",
    # Single research graph nodes
    "generate_query": "💭 Generating research query",
    "local_rag_research": "📚 Searching local knowledge base",
    "summarize_local_rag_results": "📄 Summarizing local research",
    "generate_complementary_query": "🔄 Generating complementary query",
    "web_research": "🌐 Searching the web",
    "complementary_web_research": "🌐 Conducting complementary web search",
    "summarize_sources": "📋 Summarizing research sources",
    "reflect_on_summary": "🤔 Reflecting on research quality",
    "finalize_summary": "✅ Finalizing research summary",
}


@dataclass
class ResearchSessionState:
    """Tracks the state of an ongoing research session."""
    iteration_count: int = 0
    current_iteration_nodes: list = field(default_factory=list)
    in_initialization_phase: bool = True
    initialization_header_sent: bool = False
    iteration_summaries: Dict[int, str] = field(default_factory=dict)
    current_iteration_summary: Optional[str] = None
    previous_node_name: Optional[str] = None
    route_research_completed: bool = False
    seen_node_starts: Set[str] = field(default_factory=set)


def extract_node_name_from_channelwrite(node_name: str) -> Optional[str]:
    """Extract node name from ChannelWrite format: 'Channelwrite<...,Node Name>'."""
    match = re.search(r'[,\s]+([A-Za-z_][A-Za-z0-9_\s]+)', node_name)
    if match:
        return match.group(1).strip().lower().replace(' ', '_')
    return None


def extract_node_name_from_event(event: dict) -> str:
    """Extract the actual node name from various event formats."""
    node_name = event.get("name", "")

    # Handle nested graph names like "single_path > generate_query"
    if " > " in node_name:
        node_name = node_name.split(" > ")[-1]

    # Handle ChannelWrite format
    if any(pattern in node_name for pattern in CHANNELWRITE_PATTERNS):
        extracted = extract_node_name_from_channelwrite(node_name)
        if extracted:
            return extracted

    return node_name


def get_friendly_name(node_name: str) -> str:
    """Get user-friendly name for a node, or return formatted version."""
    # Handle ChannelWrite format
    if any(pattern in node_name for pattern in CHANNELWRITE_PATTERNS):
        extracted = extract_node_name_from_channelwrite(node_name)
        if extracted:
            return NODE_DESCRIPTIONS.get(
                extracted,
                f"⚙️ {extracted.replace('_', ' ').title()}"
            )

    return NODE_DESCRIPTIONS.get(
        node_name,
        f"⚙️ {node_name.replace('_', ' ').title()}"
    )


def should_skip_node(node_name: str, actual_node_name: str) -> bool:
    """Check if a node should be skipped (internal/technical nodes)."""
    # Skip if empty
    if not node_name and not actual_node_name:
        return True

    # Skip internal LangGraph nodes
    if actual_node_name in INTERNAL_NODES:
        return True

    # Skip ChannelWrite operations
    node_lower = node_name.lower()
    actual_lower = actual_node_name.lower()
    if "channelwrite" in node_lower or "channelwrite" in actual_lower:
        return True

    # Skip if it's a ChannelWrite with incomplete info
    if "channelwrite" in node_lower and ("<..." in node_name or "..." in node_name):
        return True

    # Skip specific internal nodes
    if node_name in ["LangGraph", "_write"]:
        return True

    return False


def parse_user_input(user_input: str) -> tuple[str, str]:
    """Parse user input to extract research topic and mode."""
    user_input = user_input.strip()
    mode = "dtor" if any(keyword in user_input.lower() for keyword in ["dtor", "deep tree"]) else "single"

    # Extract research topic by removing mode keywords
    research_topic = user_input
    for keyword in ["dtor mode", "dtor", "deep tree", "single mode", "single"]:
        research_topic = research_topic.replace(keyword, "").strip()

    return research_topic, mode


def should_start_new_iteration(
    state: ResearchSessionState,
    node_name: str
) -> bool:
    """Determine if a new iteration should start based on the current node."""
    if state.iteration_count == 0:
        # First iteration: start when we see the first research node
        return node_name in FIRST_ITERATION_NODES
    else:
        # Subsequent iterations: start when we see research nodes after route_research completed
        return (
            state.route_research_completed and
            node_name in SUBSEQUENT_ITERATION_NODES
        )


async def display_iteration_summary(
    iteration_count: int,
    summary: str
) -> None:
    """Display the summary for a completed iteration."""
    summary_message = f"  📝 **Iteration Summary:**\n  {summary}"
    await cl.Message(content=summary_message).send()


async def start_new_iteration(state: ResearchSessionState) -> None:
    """Start a new research iteration and display the header."""
    state.iteration_count += 1
    state.current_iteration_nodes = []
    iteration_header = f"### 🔄 Research Iteration {state.iteration_count}"
    await cl.Message(content=iteration_header).send()


async def handle_initialization_phase(state: ResearchSessionState) -> None:
    """Handle initialization phase header display."""
    if state.in_initialization_phase and not state.initialization_header_sent:
        initialization_header = "### 🔧 Initialize Session"
        await cl.Message(content=initialization_header).send()
        state.initialization_header_sent = True


def extract_summary_from_output(output) -> Optional[str]:
    """Extract summary from various output formats."""
    if hasattr(output, 'running_summary') and output.running_summary:
        return output.running_summary

    if isinstance(output, dict):
        for key in ['running_summary', 'branch_summary', 'final_summary']:
            if key in output and output[key]:
                return output[key]

    return None


def extract_final_result(output) -> Optional[DToRStateOutput]:
    """Extract final result from output, handling various formats."""
    if isinstance(output, DToRStateOutput):
        return output

    if hasattr(output, 'final_summary'):
        return output

    if isinstance(output, dict) and 'final_summary' in output:
        return DToRStateOutput(
            final_summary=output.get('final_summary', ''),
            all_sources=output.get('all_sources', [])
        )

    return None


async def display_final_results(
    final_result: DToRStateOutput,
    final_answer: cl.Message
) -> None:
    """Display the final research summary and sources."""
    if hasattr(final_result, 'final_summary') and final_result.final_summary:
        await final_answer.stream_token(
            f"## Final Research Summary\n\n{final_result.final_summary}"
        )

    if hasattr(final_result, 'all_sources') and final_result.all_sources:
        sources_text = "\n\n## Sources\n\n"
        for i, source in enumerate(final_result.all_sources[:10], 1):
            sources_text += f"{i}. {source}\n"
        await final_answer.stream_token(sources_text)

    await final_answer.send()


async def process_chain_start_event(
    event: dict,
    state: ResearchSessionState
) -> None:
    """Process a chain start event to track node execution."""
    node_name = event.get("name", "")
    actual_node_name = extract_node_name_from_event(event)

    # Skip internal/technical nodes
    if should_skip_node(node_name, actual_node_name):
        return

    # Use run_id for deduplication
    run_id = event.get("run_id", "")
    event_key = f"{run_id}:{actual_node_name}:start"
    if event_key in state.seen_node_starts:
        return
    state.seen_node_starts.add(event_key)

    friendly_name = get_friendly_name(actual_node_name)

    # Skip if not a valid friendly name
    if not friendly_name or friendly_name.startswith("⚙️ Channelwrite"):
        return

    # Check if we should start a new iteration
    should_start = should_start_new_iteration(state, actual_node_name)

    # Display previous iteration summary if starting new iteration
    if should_start and state.iteration_count > 0 and state.current_iteration_summary:
        await display_iteration_summary(
            state.iteration_count,
            state.current_iteration_summary
        )
        state.iteration_summaries[state.iteration_count] = state.current_iteration_summary
        state.current_iteration_summary = None
        state.route_research_completed = False

    # Handle initialization phase
    await handle_initialization_phase(state)

    # Start new iteration if needed
    if should_start:
        if state.in_initialization_phase:
            state.in_initialization_phase = False
        await start_new_iteration(state)

    # Display node message
    indented_node = f"  • {friendly_name}"
    await cl.Message(content=indented_node).send()
    state.current_iteration_nodes.append(actual_node_name)
    state.previous_node_name = actual_node_name


async def process_chain_end_event(
    event: dict,
    state: ResearchSessionState
) -> Optional[DToRStateOutput]:
    """Process a chain end event to handle summaries and final results."""
    actual_node_name = extract_node_name_from_event(event)

    # Track when route_research completes
    if actual_node_name == ROUTE_RESEARCH_NODE:
        state.route_research_completed = True

    # Extract summary from output
    data = event.get("data", {})
    output = data.get("output")
    summary = extract_summary_from_output(output)

    # Update current iteration summary if found
    if summary and actual_node_name in SUMMARY_NODES:
        state.current_iteration_summary = summary

    # Extract final result
    final_result = extract_final_result(output)
    if final_result and state.current_iteration_summary:
        await display_iteration_summary(
            state.iteration_count,
            state.current_iteration_summary
        )

    return final_result


@cl.on_chat_start
async def on_chat_start():
    """Initialize the graph when a new chat session starts."""
    graph = create_main_graph(checkpointer=None)
    cl.user_session.set("graph", graph)

    await cl.Message(
        content=(
            "Welcome to the Deep Tree of Research assistant! 🚀\n\n"
            "I can help you conduct deep research on any topic. You can choose between:\n"
            "- **Single-path mode**: Fast, linear research\n"
            "- **DToR mode**: Deep Tree of Research with multiple branches and perspectives\n\n"
            "Just send me a research topic and optionally specify the mode "
            "(e.g., 'Research quantum computing in dtor mode')."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and execute the research graph."""
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(
            content="Error: Graph not initialized. Please refresh the page."
        ).send()
        return

    # Parse user input
    research_topic, mode = parse_user_input(message.content)
    if not research_topic:
        await cl.Message(
            content=(
                "Please provide a research topic. For example: "
                "'Research quantum computing' or 'Research quantum computing in dtor mode'"
            )
        ).send()
        return

    # Create input state and config
    input_state = DToRStateInput(research_topic=research_topic, mode=mode)
    research_config = RunnableConfig(
        configurable={"thread_id": cl.context.session.id},
        recursion_limit=RECURSION_LIMIT
    )

    final_answer = cl.Message(content="")
    state = ResearchSessionState()

    try:
        final_result = None

        async for event in graph.astream_events(
            input_state,
            config=research_config,
            version="v2"
        ):
            event_type = event.get("event")

            if event_type == "on_chain_start":
                await process_chain_start_event(event, state)
            elif event_type == "on_chain_end":
                result = await process_chain_end_event(event, state)
                if result:
                    final_result = result

        # Display final results
        if final_result:
            await display_final_results(final_result, final_answer)
        else:
            await cl.Message(
                content="Research completed, but no final summary was generated."
            ).send()

    except Exception as e:
        error_msg = f"Error during research execution: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise
