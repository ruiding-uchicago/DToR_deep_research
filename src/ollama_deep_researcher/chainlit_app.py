"""
Chainlit integration for the Deep Tree of Research (DToR) LangGraph.

This module provides a Chainlit UI for interacting with the DToR research graph.
Users can input research topics and select between single-path or DToR modes.
"""

import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage

from ollama_deep_researcher.dtor_graph import create_main_graph
from ollama_deep_researcher.dtor_state import DToRStateInput


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

    # Create callback handler for streaming
    cb = cl.LangchainCallbackHandler()

    # Create a message to stream the final answer
    final_answer = cl.Message(content="")

    # Configure the runnable config with callbacks
    config = RunnableConfig(
        callbacks=[cb],
        configurable={"thread_id": cl.context.session.id}
    )

    # Stream the graph execution
    try:
        # Stream messages from the graph
        async for event in graph.astream(
            input_state,
            config=config,
            stream_mode="messages"
        ):
            # Check if this is a final output message
            if hasattr(event, 'content') and event.content:
                # Only stream content from final nodes to avoid duplication
                # The callback handler will show intermediate steps
                pass

        # Get the final result by invoking the graph
        final_result = await graph.ainvoke(
            input_state,
            config=config
        )

        # Display the final summary
        if hasattr(final_result, 'final_summary') and final_result.final_summary:
            await final_answer.stream_token(f"\n\n## Final Research Summary\n\n{final_result.final_summary}")

        # Display sources if available
        if hasattr(final_result, 'all_sources') and final_result.all_sources:
            sources_text = "\n\n## Sources\n\n"
            for i, source in enumerate(final_result.all_sources[:10], 1):  # Limit to first 10 sources
                sources_text += f"{i}. {source}\n"
            await final_answer.stream_token(sources_text)

        await final_answer.send()

    except Exception as e:
        error_msg = f"Error during research execution: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise
