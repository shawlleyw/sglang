import aiohttp
import asyncio

port = 30000
url = f"http://localhost:{port}/generate"
data = {
    "text": "The capital of France is",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 32,
    },
    "stream": True,
}

async def main():
    bs = 1
    async def gao():
        async with aiohttp.request("POST", url, json=data) as r:
            response = await r.text()
            print(response)
    tasks = [
        asyncio.create_task(gao()) for i in range(bs)
    ]
    await asyncio.gather(*tasks)

asyncio.run(main())