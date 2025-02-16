import json
import asyncio
import time
from openai import AsyncOpenAI
from e2b_code_interpreter import AsyncSandbox

tools = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute python code in a sandbox and return result, good for simple python code like calculations and counting, and other basic math tasks",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute in a single cell"
                }
            },
            "required": ["code"]
        }
    }
}]


sandbox = None

async def _init_sandbox():
    global sandbox
    if sandbox is None:
        print("create sandbox")
        sandbox = await AsyncSandbox.create()

async def _kill_sandbox():
    global sandbox
    if sandbox is not None:
        print("kill sandbox")
        await sandbox.kill()

async def _execute_python(code = ""):
    global sandbox
    print("Execute SANDBOX")
    execution = await sandbox.run_code(code)
    # format res to string
    print(execution)
    return str(execution)

NAME_TO_FUNCTION = {
    'execute_python' : _execute_python
}

# pass in completion from client.chat.completions.create()
# returns true if tool used and appends the tool response to messages
async def _apply_tool(completion, messages):
    if completion.choices[0].message.tool_calls != None:
        for tool_call in completion.choices[0].message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            function = NAME_TO_FUNCTION[tool_call.function.name]
            res = await function(**args)
            messages.append(completion.choices[0].message)
            messages.append({
                "role": "tool",
                "name": tool_call.function.name,
                "content": res,
                "tool_call_id": tool_call.id,
            })
        return True
    return False

async def _cleanup():
    await _kill_sandbox()

async def _init():
    await _init_sandbox()


def chat_completion_with_tool(client: AsyncOpenAI, messages_list, model = "gpt-4o"):
    async def tool_call_flow(messages):
        completion = await client.chat.completions.create(
            model = model,
            messages = messages,
            tools = tools,
        )
        if await _apply_tool(completion, messages):
            completion_final = await client.chat.completions.create(
                model = model,
                messages = messages,
                tools = tools
            )
            return completion_final.choices[0].message.content
        
        return completion.choices[0].message.content
    
    async def run_batch():
        await _init()
        tasks = [tool_call_flow(messages) for messages in messages_list]
        result = await asyncio.gather(*tasks)
        await _cleanup()

        return result
    

    return asyncio.run(run_batch())


if __name__ == '__main__':
    messages = [
        [{"role": "user", "content": "How many r's are there in strawberry?"}],
        [{"role": "user", "content": "How what is the square root of 2?"}],
        [{"role": "user", "content": "which is larger, 9.9 or 9.11?"}]
    ]

    client = AsyncOpenAI()
    
    print(chat_completion_with_tool(client, messages))