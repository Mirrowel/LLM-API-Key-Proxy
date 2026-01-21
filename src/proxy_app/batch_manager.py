# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

import asyncio
from typing import List, Dict, Any, Tuple
import time
from rotator_library import RotatingClient

class EmbeddingBatcher:
    def __init__(self, client: RotatingClient, batch_size: int = 64, timeout: float = 0.1):
        self.client = client
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._batch_worker())

    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        future = asyncio.Future()
        await self.queue.put((request_data, future))
        return await future

    async def _batch_worker(self):
        while True:
            batch, futures = await self._gather_batch()
            if not batch:
                continue

            try:
                # Assume all requests in a batch use the same model and other settings
                model = batch[0]["model"]
                inputs = [item["input"][0] for item in batch] # Extract single string input

                batched_request = {
                    "model": model,
                    "input": inputs
                }
                
                # Pass through any other relevant parameters from the first request
                for key in ["input_type", "dimensions", "user"]:
                    if key in batch[0]:
                        batched_request[key] = batch[0][key]

                response = await self.client.aembedding(**batched_request)
                
                # Distribute results back to the original requesters
                for i, future in enumerate(futures):
                    # Create a new response object for each item in the batch
                    single_response_data = {
                        "object": response.object,
                        "model": response.model,
                        "data": [response.data[i]],
                        "usage": response.usage # Usage is for the whole batch
                    }
                    future.set_result(single_response_data)

            except Exception as e:
                for future in futures:
                    future.set_exception(e)

    async def _gather_batch(self) -> Tuple[List[Dict[str, Any]], List[asyncio.Future]]:
        batch = []
        futures = []
        start_time = time.time()

        while len(batch) < self.batch_size and (time.time() - start_time) < self.timeout:
            try:
                # Wait for an item with a timeout
                timeout = self.timeout - (time.time() - start_time)
                if timeout <= 0:
                    break
                request, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                batch.append(request)
                futures.append(future)
            except asyncio.TimeoutError:
                break
        
        return batch, futures

    async def stop(self):
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass