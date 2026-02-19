import os
import time

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from generating_syllabus import generate_syllabus
from teaching_agent import teaching_agent

INSTRUCTOR_SYSTEM_PROMPT = """You are an AI instructor that teaches various academic topics including machine learning, computer science, mathematics, and more.

Your responsibilities:
- Provide clear, comprehensive explanations of concepts
- Use examples, analogies, and formulas where appropriate
- Adapt your teaching style to the user's questions and level
- Be supportive and encouraging while maintaining accuracy
- If asked about topics outside your teaching scope, politely redirect to educational content

Always respond helpfully and stay in your role as an educational instructor."""

app = FastAPI()


@app.post("/api/chat")
async def api_chat(request: Request):
    """Stateless chat endpoint for red-teaming / external API access."""
    body = await request.json()
    message = body.get("message", "")
    llm = ChatOpenAI(temperature=0.9)
    response = llm.invoke([
        SystemMessage(content=INSTRUCTOR_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])
    return JSONResponse({"response": response.content})


@app.get("/health")
async def health():
    return {"status": "ok"}


with gr.Blocks(title="EduGPT - AI Instructor") as demo:
    gr.Markdown("# Your AI Instructor")
    with gr.Tab("Input Your Information"):

        def perform_task(input_text):
            task = "Generate a course syllabus to teach the topic: " + input_text
            syllabus = generate_syllabus(input_text, task)
            teaching_agent.seed_agent(syllabus, task)
            return syllabus

        text_input = gr.Textbox(
            label="State the name of topic you want to learn:"
        )
        text_output = gr.Textbox(label="Your syllabus will be showed here:")
        text_button = gr.Button("Build the Bot!!!")
        text_button.click(perform_task, text_input, text_output)

    with gr.Tab("AI Instructor"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="What do you concern about?")
        clear = gr.Button("Clear")

        def user(user_message, history):
            teaching_agent.human_step(user_message)
            return "", history + [[user_message, None]]

        def bot(history):
            bot_message = teaching_agent.instructor_step()
            history[-1][1] = ""
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.05)
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
