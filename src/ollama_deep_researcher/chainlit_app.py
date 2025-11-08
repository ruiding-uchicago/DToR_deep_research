"""
Chainlit integration for the Deep Tree of Research (DToR) LangGraph.

This module provides a Chainlit UI for interacting with the DToR research graph.
Users can input research topics and select between single-path or DToR modes.
"""

import chainlit as cl
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
    return NODE_DESCRIPTIONS.get(node_name, f"⚙️ {node_name.replace('_', ' ').title()}")


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
        current_step = None

        # Use astream_events with version="v2" to get structured events
        async for event in graph.astream_events(
            input_state,
            config=research_config,
            version="v2"
        ):
            event_type = event.get("event")
            node_name = event.get("name", "")

            # Filter for node start events to show progress
            if event_type == "on_chain_start":
                if node_name and node_name not in ["__start__", "__end__"]:
                    # Only show if it's a new step
                    if current_step != node_name:
                        current_step = node_name
                        friendly_name = get_friendly_name(node_name)

                        # Send a progress message
                        await cl.Message(content=f"**{friendly_name}**").send()

            # Filter for node end events to get final result
            elif event_type == "on_chain_end":
                # Check if this is the final output
                data = event.get("data", {})
                output = data.get("output")

                # Check if this is the final output (DToRStateOutput)
                if isinstance(output, DToRStateOutput):
                    final_result = output
                elif hasattr(output, 'final_summary'):
                    final_result = output
                elif isinstance(output, dict) and 'final_summary' in output:
                    final_result = DToRStateOutput(**output) if isinstance(output, dict) else output

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
