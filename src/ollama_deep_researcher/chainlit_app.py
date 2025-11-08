"""
Chainlit integration for the Deep Tree of Research (DToR) LangGraph.

This module provides a Chainlit UI for interacting with the DToR research graph.
Users can input research topics and select between single-path or DToR modes.
"""

import chainlit as cl
import re
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage

from ollama_deep_researcher.dtor_graph import create_main_graph
from ollama_deep_researcher.dtor_state import DToRStateInput, DToRStateOutput

# Set recursion limit constant
RECURSION_LIMIT = 500

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


def get_friendly_name(node_name: str) -> str:
    """Get user-friendly name for a node, or return formatted version of the name."""
    # Clean up node name - handle cases like "Channelwrite<...,Node Name>"
    if "Channelwrite" in node_name or "ChannelWrite" in node_name:
        # Extract node name from ChannelWrite format: "Channelwrite<...,Node Name>"
        match = re.search(r'[,\s]+([A-Za-z_][A-Za-z0-9_\s]+)', node_name)
        if match:
            extracted_name = match.group(1).strip().lower().replace(' ', '_')
            return NODE_DESCRIPTIONS.get(extracted_name, f"⚙️ {extracted_name.replace('_', ' ').title()}")

    return NODE_DESCRIPTIONS.get(node_name, f"⚙️ {node_name.replace('_', ' ').title()}")


def extract_node_name_from_event(event: dict) -> str:
    """Extract the actual node name from various event formats."""
    node_name = event.get("name", "")

    # Handle nested graph names like "single_path > generate_query"
    if " > " in node_name:
        parts = node_name.split(" > ")
        node_name = parts[-1]  # Take the leaf node name

    # Handle ChannelWrite format: "Channelwrite<...,Node Name>"
    if "Channelwrite" in node_name or "ChannelWrite" in node_name:
        match = re.search(r'[,\s]+([A-Za-z_][A-Za-z0-9_\s]+)', node_name)
        if match:
            return match.group(1).strip().lower().replace(' ', '_')

    return node_name


def should_skip_node(node_name: str, actual_node_name: str) -> bool:
    """Check if a node should be skipped (internal/technical nodes)."""
    # Skip if empty
    if not node_name and not actual_node_name:
        return True

    # Skip internal LangGraph nodes
    if actual_node_name in ["__start__", "__end__"]:
        return True

    # Skip ChannelWrite operations (internal state management)
    if "channelwrite" in node_name.lower() or "channelwrite" in actual_node_name.lower():
        return True

    # Skip generic "Write" operations
    if actual_node_name.lower() in ["write", "read"]:
        return True

    # Skip if the node name is just "..." or empty after extraction
    if actual_node_name in ["", "...", "none"]:
        return True

    # Skip if it's a ChannelWrite with incomplete info (like "Channelwrite<...>")
    if "channelwrite" in node_name.lower() and ("<..." in node_name or "..." in node_name):
        return True

    # Skip for LangGraph and write
    if "LangGraph" == node_name or "_write" == node_name:
        return True

    return False


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the graph when a new chat session starts.
    """
    # Create the graph instance
    graph = create_main_graph(checkpointer=None)

    # Store the graph in the user session
    cl.user_session.set("graph", graph)

    # Welcome message
    await cl.Message(
        content="Welcome to the Deep Tree of Research assistant! 🚀\n\n"
                "I can help you conduct deep research on any topic. You can choose between:\n"
                "- **Single-path mode**: Fast, linear research\n"
                "- **DToR mode**: Deep Tree of Research with multiple branches and perspectives\n\n"
                "Just send me a research topic and optionally specify the mode (e.g., 'Research quantum computing in dtor mode')."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and execute the research graph.
    """
    # Get the graph from the session
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="Error: Graph not initialized. Please refresh the page.").send()
        return

    # Parse the user's message to extract research topic and mode
    user_input = message.content.strip()

    # Default to single mode unless user specifies "dtor"
    mode = "dtor" if "dtor" in user_input.lower() or "deep tree" in user_input.lower() else "single"

    # Extract the research topic (remove mode keywords)
    research_topic = user_input
    for keyword in ["dtor mode", "dtor", "deep tree", "single mode", "single"]:
        research_topic = research_topic.replace(keyword, "").strip()

    if not research_topic:
        await cl.Message(
            content="Please provide a research topic. For example: 'Research quantum computing' or 'Research quantum computing in dtor mode'"
        ).send()
        return

    # Create the input state
    input_state = DToRStateInput(
        research_topic=research_topic,
        mode=mode
    )

    # Configure the runnable config - no callback handler to avoid LangSmith conflicts
    research_config = RunnableConfig(
        configurable={"thread_id": cl.context.session.id},
        recursion_limit=RECURSION_LIMIT
    )

    # Create a message to stream the final answer
    final_answer = cl.Message(content="")

    # Stream the graph execution using astream_events
    try:
        final_result = None
        seen_node_starts = set()  # Track node starts using run_id to prevent duplicates
        iteration_count = 0  # Track iteration number
        current_iteration_nodes = []  # Track nodes in current iteration
        in_initialization_phase = True  # Track if we're in the initialization phase
        initialization_header_sent = False  # Track if we've sent the initialization header
        iteration_summaries = {}  # Track summaries for each iteration
        current_iteration_summary = None  # Track current iteration's summary
        previous_node_name = None  # Track previous node to detect transitions
        route_research_completed = False  # Track when route_research completes
        research_iteration_nodes = ["generate_query", "local_rag_research", "research_node", "diversify_query"]  # Nodes that indicate new iteration

        # Use astream_events with version="v2" to get structured events
        async for event in graph.astream_events(
            input_state,
            config=research_config,
            version="v2"
        ):
            event_type = event.get("event")
            node_name = event.get("name", "")

            # Extract the actual node name (handles ChannelWrite format and nested graphs)
            actual_node_name = extract_node_name_from_event(event)

            # Filter for node start events to show progress
            if event_type == "on_chain_start":
                # Skip internal/technical nodes
                if should_skip_node(node_name, actual_node_name):
                    continue

                # Use run_id for deduplication to handle nodes that appear multiple times
                run_id = event.get("run_id", "")
                event_key = f"{run_id}:{actual_node_name}:start"
                if event_key in seen_node_starts:
                    continue
                seen_node_starts.add(event_key)

                friendly_name = get_friendly_name(actual_node_name)

                # Only show if we have a valid friendly name (not the default fallback for skipped nodes)
                if friendly_name and not friendly_name.startswith("⚙️ Channelwrite"):
                    # Detect when we should start a new iteration
                    # New iteration starts when:
                    # 1. First iteration: when we see generate_query or local_rag_research for the first time
                    # 2. Subsequent iterations: when we see local_rag_research or generate_query after route_research has completed
                    should_start_iteration = False
                    if iteration_count == 0:
                        # First iteration: start when we see the first research node
                        should_start_iteration = actual_node_name in ["generate_query", "local_rag_research"]
                    else:
                        # Subsequent iterations: start when we see research nodes after route_research completed
                        should_start_iteration = (
                            route_research_completed and
                            actual_node_name in ["local_rag_research", "generate_query"]
                        )

                    # If we're starting a new iteration, display the previous iteration's summary
                    if should_start_iteration and iteration_count > 0 and current_iteration_summary:
                        summary_message = f"  📝 **Iteration Summary:**\n  {current_iteration_summary}"
                        await cl.Message(content=summary_message).send()
                        iteration_summaries[iteration_count] = current_iteration_summary
                        current_iteration_summary = None
                        route_research_completed = False  # Reset flag after starting new iteration

                    # Send initialization header if this is the first node and we're still in initialization
                    if in_initialization_phase and not initialization_header_sent:
                        initialization_header = "### 🔧 Initialize Session"
                        await cl.Message(content=initialization_header).send()
                        initialization_header_sent = True

                    # If we're starting an iteration, we've left the initialization phase
                    if should_start_iteration and in_initialization_phase:
                        in_initialization_phase = False
                        current_iteration_nodes = []

                        # Start a new iteration
                        iteration_count += 1
                        iteration_header = f"### 🔄 Research Iteration {iteration_count}"
                        await cl.Message(content=iteration_header).send()
                        current_iteration_nodes = []
                    elif should_start_iteration:
                        # Start a new iteration (not the first one)
                        current_iteration_nodes = []

                        iteration_count += 1
                        iteration_header = f"### 🔄 Research Iteration {iteration_count}"
                        await cl.Message(content=iteration_header).send()
                        current_iteration_nodes = []

                    # Send node message - nested under initialization or iteration
                    # Add indentation to show it's a child of the section
                    indented_node = f"  • {friendly_name}"
                    await cl.Message(content=indented_node).send()
                    current_iteration_nodes.append(actual_node_name)
                    previous_node_name = actual_node_name

            # Filter for node end events to get final result and capture summaries
            elif event_type == "on_chain_end":
                # Track when route_research completes - this indicates a new iteration may start
                if actual_node_name == "route_research":
                    route_research_completed = True

                # Extract state data to get summaries
                data = event.get("data", {})
                output = data.get("output")

                # Try to extract summary from various state formats
                summary = None

                # Check for running_summary in SummaryState
                if hasattr(output, 'running_summary') and output.running_summary:
                    summary = output.running_summary
                elif isinstance(output, dict):
                    if 'running_summary' in output and output['running_summary']:
                        summary = output['running_summary']
                    elif 'branch_summary' in output and output['branch_summary']:
                        summary = output['branch_summary']
                    elif 'final_summary' in output and output['final_summary']:
                        summary = output['final_summary']

                # Update current iteration summary if we found one
                # Priority: finalize_summary > synthesize_branch > summarize_sources
                if summary and actual_node_name in ["finalize_summary", "synthesize_branch", "summarize_sources"]:
                    current_iteration_summary = summary

                # Check if this is the final output (DToRStateOutput)
                if isinstance(output, DToRStateOutput):
                    final_result = output
                    # Display last iteration summary if available
                    if current_iteration_summary:
                        summary_message = f"  📝 **Iteration Summary:**\n  {current_iteration_summary[:500]}..." if len(current_iteration_summary) > 500 else f"  📝 **Iteration Summary:**\n  {current_iteration_summary}"
                        await cl.Message(content=summary_message).send()
                elif hasattr(output, 'final_summary'):
                    final_result = output
                elif isinstance(output, dict) and 'final_summary' in output:
                    # Only extract fields that DToRStateOutput accepts
                    final_result = DToRStateOutput(
                        final_summary=output.get('final_summary', ''),
                        all_sources=output.get('all_sources', [])
                    )

        # Display the final summary
        if final_result:
            if hasattr(final_result, 'final_summary') and final_result.final_summary:
                await final_answer.stream_token(f"## Final Research Summary\n\n{final_result.final_summary}")

            # Display sources if available
            if hasattr(final_result, 'all_sources') and final_result.all_sources:
                sources_text = "\n\n## Sources\n\n"
                for i, source in enumerate(final_result.all_sources[:10], 1):
                    sources_text += f"{i}. {source}\n"
                await final_answer.stream_token(sources_text)

            await final_answer.send()
        else:
            await cl.Message(content="Research completed, but no final summary was generated.").send()

    except Exception as e:
        error_msg = f"Error during research execution: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise
