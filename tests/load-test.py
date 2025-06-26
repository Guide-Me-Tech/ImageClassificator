import aiohttp
import aiofiles
import asyncio
import argparse
import time

PUBLIC_URL = "https://smarty-test.smartbank.uz/image/classification"


async def send_request(session: aiohttp.ClientSession, image: aiofiles.AsyncFile):
    start_time = time.time()
    form = aiohttp.FormData()
    form.add_field("image", image)
    resp = await session.post(PUBLIC_URL, form)
    end_time = time.time()
    return end_time - start_time, resp


async def start_load_test(
    concurrent_requests: int, requests_per_second: int, duration: int
):
    files = [
        "tests/test_1.webp",
        "tests/test_2.webp",
        "tests/test_3.jpeg",
        "tests/test_4.jpeg",
        "tests/test_5.jpeg",
    ]
    images = [aiofiles.open(file, "rb") for file in files]
    image_index = 0
    while True:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(concurrent_requests):
                form = aiohttp.FormData()
                form.add_field("image", images[image_index])
                tasks.append(session.post(PUBLIC_URL, form))
                image_index = (image_index + 1) % len(images)
            latencies, responses = await asyncio.gather(*tasks)
            average_latency = sum(latencies) / len(latencies)
            print(f"Average latency: {average_latency} seconds")
            time.sleep(4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrent-requests", type=int, default=1)
    parser.add_argument("--requests-per-second", type=int, default=1)
    parser.add_argument("--duration", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(
        start_load_test(
            args.concurrent_requests, args.requests_per_second, args.duration
        )
    )
